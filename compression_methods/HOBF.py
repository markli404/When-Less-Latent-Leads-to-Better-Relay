import time
from typing import Any, Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from compression_methods.BaseKVCompressor import BaseKVCompressor


class HOBF(BaseKVCompressor):
    """
    Head-wise prompt slicing + head-wise compensation on V.

    KV layout (shared across batch):
      [ history | prompt(padded) | latent_tail ]
    - history length is shared across batch
    - latent_tail length == latent_steps (shared)
    - prompt is right padded, described by prompt_mask (B, L_prompt)

    Per layer, per sample, per KV head:
      1) Select exactly k_eff prompt tokens by aggregated attention (exclude padding, exclude sink).
      2) Physically slice K and V (no masks, no zeroing):
           keep all history + (optional sink once) + selected prompt tokens (head-wise) + all latent_tail
      3) Compensation on V (head-wise):
           - Build span(V_keep) from kept prompt tokens (QR)
           - Compute residual R = V_disc - Proj_keep(V_disc)
           - PCA/SVD on R, keep top pca_rank residual directions
           - Build an attention-weighted residual mean vector and project onto top residual subspace
           - Inject the resulting delta into the kept prompt V in the NEW cache (uniform or attention-distributed)

    Notes / Assumptions:
      - all_steps_attentions[step][layer] is expected to be shaped like (B, H_q, K_len)
        (i.e., already reduced over query length if your model returns (B,H,Q,K)).
      - This edits V only; K is never numerically modified (only sliced).
      - Sink tokens are kept only once per reset (first compress call after reset), consistent with BaseKVCompressor behavior.
      - All comments are in English (per your preference).
    """

    def __init__(
        self,
        sink_size: int = 4,
        kv_budget: int = 32,
        pca_rank: int = 8,
        eps: float = 1e-12,
        inject_mode: str = "uniform",     # {"uniform", "attn"}
        scale_mode: str = "ad_over_as",   # {"none", "ad_over_as", "ad_frac", "ad"}
        center: bool = False,             # whether to mean-center V before subspace ops
    ):
        super().__init__(sink_size=int(sink_size), kv_budget=int(kv_budget))
        self.pca_rank = max(0, int(pca_rank))
        self.eps = float(eps)
        self.inject_mode = str(inject_mode)
        self.scale_mode = str(scale_mode)
        self.center = bool(center)

        if self.inject_mode not in {"uniform", "attn"}:
            raise ValueError(f"inject_mode must be 'uniform' or 'attn', got {self.inject_mode}")
        if self.scale_mode not in {"none", "ad_over_as", "ad_frac", "ad"}:
            raise ValueError(f"scale_mode invalid: {self.scale_mode}")

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _past_length(past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "key_cache"):
            return int(past_key_values.key_cache[0].shape[-2]) if len(past_key_values.key_cache) > 0 else 0
        return int(past_key_values[0][0].shape[-2]) if len(past_key_values) > 0 else 0

    @staticmethod
    def _as_legacy_tuple(past_key_values: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        if past_key_values is None:
            return tuple()
        if hasattr(past_key_values, "key_cache"):
            return tuple((k, v) for k, v in zip(past_key_values.key_cache, past_key_values.value_cache))
        return past_key_values

    @staticmethod
    def _nanmin_mean_max(x: torch.Tensor) -> Tuple[float, float, float]:
        xf = x.reshape(-1)
        mask = torch.isfinite(xf)
        if mask.sum().item() == 0:
            return float("nan"), float("nan"), float("nan")
        vals = xf[mask]
        return float(vals.min().item()), float(vals.mean().item()), float(vals.max().item())

    def _compute_scale(self, Ad: torch.Tensor, As: torch.Tensor) -> torch.Tensor:
        """
        Ad, As: (B, H_kv) float tensors
        Returns: (B, H_kv) float tensor
        """
        eps = self.eps
        if self.scale_mode == "none":
            return torch.ones_like(Ad)
        if self.scale_mode == "ad_over_as":
            return Ad / (As + eps)
        if self.scale_mode == "ad_frac":
            return Ad / (Ad + As + eps)
        # "ad"
        return Ad

    def _aggregate_prompt_scores_kv(
        self,
        *,
        device: torch.device,
        B: int,
        H_kv: int,
        L_total: int,
        L_history: int,
        L_prompt: int,
        all_steps_attentions: List[List[torch.Tensor]],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Returns:
          prompt_scores_kv: (B, H_kv, L_prompt) float32
          H_q: query heads observed (int)
        """
        if len(all_steps_attentions) == 0:
            return torch.zeros((B, H_kv, L_prompt), device=device, dtype=torch.float32), H_kv

        a0 = all_steps_attentions[0][layer_idx]
        H_q = int(a0.shape[1])

        max_klen = 0
        for step_data in all_steps_attentions:
            a = step_data[layer_idx]
            max_klen = max(max_klen, int(a.shape[-1]))

        agg = torch.zeros((B, H_q, max_klen), device=device, dtype=torch.float32)
        for step_data in all_steps_attentions:
            a = step_data[layer_idx].to(device=device, dtype=torch.float32, non_blocking=True)
            if int(a.shape[-1]) < max_klen:
                a = F.pad(a, (0, max_klen - int(a.shape[-1])))
            agg += a

        if max_klen < L_total:
            agg_full = F.pad(agg, (0, L_total - max_klen))  # (B, H_q, L_total)
        else:
            agg_full = agg[:, :, :L_total]

        prompt_scores_q = agg_full[:, :, L_history:L_history + L_prompt]  # (B, H_q, L_prompt)

        # Reduce query heads -> kv heads (GQA)
        if H_q == H_kv:
            prompt_scores_kv = prompt_scores_q
        elif (H_q % H_kv) == 0:
            group = H_q // H_kv
            prompt_scores_kv = prompt_scores_q.view(B, H_kv, group, L_prompt).sum(dim=2)
        else:
            # Fallback (rare, but keeps code robust)
            if prompt_scores_q.shape[1] >= H_kv:
                prompt_scores_kv = prompt_scores_q[:, :H_kv, :]
            else:
                prompt_scores_kv = F.pad(prompt_scores_q, (0, 0, 0, H_kv - prompt_scores_q.shape[1]))

        return prompt_scores_kv.to(dtype=torch.float32), H_q

    # -----------------------------
    # Main compression
    # -----------------------------
    @torch.no_grad()
    def compress(
        self,
        *,
        past_key_values: Any,
        latent_steps: int,
        all_steps_attentions: List[List[torch.Tensor]],
        prompt_mask: torch.Tensor,                 # (B, L_prompt) 0/1, right padded
        current_full_mask: Optional[Any] = None,   # ignored
        debug: bool = False,
        **kwargs,
    ) -> Tuple[Any, float]:

        t0 = time.time()

        if past_key_values is None:
            return None, 0.0

        layers = self._as_legacy_tuple(past_key_values)
        if len(layers) == 0:
            return past_key_values, 0.0

        k0, _ = layers[0]
        B = int(k0.shape[0])
        num_layers = len(layers)
        H_kv = int(k0.shape[1])

        if prompt_mask.dim() != 2:
            raise ValueError(f"prompt_mask must be (B, L_prompt), got {prompt_mask.shape}")
        if int(prompt_mask.shape[0]) != B:
            raise ValueError(f"Batch mismatch: KV B={B}, prompt_mask B={prompt_mask.shape[0]}")

        L_prompt = int(prompt_mask.shape[1])

        steps = int(latent_steps) if latent_steps is not None else 0
        steps = max(0, steps)

        L_total = self._past_length(past_key_values)
        if steps > L_total:
            steps = L_total

        L_latent = steps
        L_history = L_total - L_prompt - L_latent
        if L_history < 0:
            raise ValueError(f"Invalid layout: L_total={L_total}, L_prompt={L_prompt}, L_latent={L_latent}")

        # Keep sink only once per reset
        apply_sink = (not getattr(self, "_has_kept_sink", False)) and (self.sink_size > 0)

        new_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(num_layers):
            k, v = layers[layer_idx]
            device = k.device
            D = int(k.shape[-1])

            # Move prompt_mask onto this layer device
            pm = (prompt_mask > 0).to(dtype=torch.long, device=device)  # (B, L_prompt)
            valid_prompt_len = pm.sum(dim=-1).to(torch.long)            # (B,)

            # Sink length common across batch (avoid per-sample length mismatch)
            if apply_sink:
                min_valid = int(valid_prompt_len.min().item())
                sink_len_common = min(int(self.sink_size), min_valid)
            else:
                sink_len_common = 0

            # k_eff is per-head prompt keep budget (non-sink portion), shared across batch
            available = (valid_prompt_len - sink_len_common).clamp(min=0)
            k_eff = min(int(self.kv_budget), int(available.min().item()))
            k_eff = max(0, k_eff)

            # Aggregate attention -> prompt_scores_kv: (B, H_kv, L_prompt)
            prompt_scores_kv, H_q = self._aggregate_prompt_scores_kv(
                device=device,
                B=B,
                H_kv=H_kv,
                L_total=L_total,
                L_history=L_history,
                L_prompt=L_prompt,
                all_steps_attentions=all_steps_attentions,
                layer_idx=layer_idx,
            )

            # Mask padding and sink region for selection
            prompt_scores_kv = prompt_scores_kv.masked_fill(pm.unsqueeze(1) == 0, -float("inf"))
            if sink_len_common > 0:
                prompt_scores_kv[:, :, :sink_len_common] = -float("inf")

            # Build per-(b, head) selected prompt indices (within prompt coordinate)
            # selected_prompt_idx[b]: (H_kv, k_eff)
            selected_prompt_idx: List[torch.Tensor] = []
            for b in range(B):
                vp = int(valid_prompt_len[b].item())
                if k_eff <= 0 or vp <= sink_len_common:
                    selected_prompt_idx.append(torch.zeros((H_kv, 0), device=device, dtype=torch.long))
                    continue

                scores_b = prompt_scores_kv[b, :, :vp]                     # (H_kv, vp)
                seg = scores_b[:, sink_len_common:vp]                      # (H_kv, vp-sink)
                kk = min(k_eff, int(seg.shape[-1]))
                if kk <= 0:
                    selected_prompt_idx.append(torch.zeros((H_kv, 0), device=device, dtype=torch.long))
                    continue

                topk_rel = torch.topk(seg, k=kk, dim=-1).indices            # (H_kv, kk), per head
                topk_rel, _ = torch.sort(topk_rel, dim=-1)                  # keep original order
                topk = topk_rel + sink_len_common                           # shift back into [0, vp)
                selected_prompt_idx.append(topk.to(dtype=torch.long))

            # Build head-wise full indices in ORIGINAL KV coordinate:
            # idx shape: (B, H_kv, L_new)
            # L_new = L_history + sink_len_common + k_eff + L_latent
            L_new = L_history + sink_len_common + k_eff + L_latent
            idx = torch.empty((B, H_kv, L_new), device=device, dtype=torch.long)

            # History part (shared)
            if L_history > 0:
                hist = torch.arange(0, L_history, device=device, dtype=torch.long)
                idx[:, :, :L_history] = hist.view(1, 1, -1).expand(B, H_kv, -1)

            # Sink part (shared)
            if sink_len_common > 0:
                sink = torch.arange(0, sink_len_common, device=device, dtype=torch.long) + L_history
                s0 = L_history
                s1 = L_history + sink_len_common
                idx[:, :, s0:s1] = sink.view(1, 1, -1).expand(B, H_kv, -1)

            # Prompt selected part (head-wise)
            p0 = L_history + sink_len_common
            p1 = p0 + k_eff
            if k_eff > 0:
                for b in range(B):
                    topk = selected_prompt_idx[b]  # (H_kv, k_eff) or empty
                    if topk.numel() == 0:
                        # Safe fallback; should not happen when k_eff is computed from batch min available.
                        idx[b, :, p0:p1] = (L_history + sink_len_common)
                    else:
                        idx[b, :, p0:p1] = topk + L_history

            # Latent tail (shared)
            if L_latent > 0:
                lat = torch.arange(L_total - L_latent, L_total, device=device, dtype=torch.long)
                l0 = L_new - L_latent
                idx[:, :, l0:] = lat.view(1, 1, -1).expand(B, H_kv, -1)

            # Gather new_k/new_v (no masks, no zeroing)
            idx_exp = idx.unsqueeze(-1).expand(B, H_kv, L_new, D)  # (B,H,L_new,D)
            new_k = torch.gather(k, dim=2, index=idx_exp)
            new_v = torch.gather(v, dim=2, index=idx_exp)

            # -----------------------------
            # Head-wise compensation on V (prompt selected segment only)
            # -----------------------------
            # Diagnostics (optional)
            if debug:
                r_perp = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
                pca_evr = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
                ad_over_as = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
                inj_norm = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)

            if self.pca_rank > 0 and k_eff > 0:
                for b in range(B):
                    vp = int(valid_prompt_len[b].item())
                    if vp <= sink_len_common:
                        continue

                    candidates = torch.arange(sink_len_common, vp, device=device, dtype=torch.long)

                    topk_b = selected_prompt_idx[b]  # (H_kv, k_eff)
                    if topk_b.numel() == 0:
                        continue

                    for h in range(H_kv):
                        kept = topk_b[h]  # (k_eff,)
                        if kept.numel() == 0:
                            continue

                        # Build discard indices (prompt coordinate)
                        selected = torch.zeros((vp,), device=device, dtype=torch.bool)
                        selected[kept] = True
                        disc = candidates[~selected[candidates]]
                        if disc.numel() == 0:
                            # Still log Ad/As if asked
                            if debug:
                                wk = prompt_scores_kv[b, h, kept].to(torch.float32)
                                As = wk.sum()
                                ad_over_as[b, h] = float("nan") if As.item() == 0 else 0.0
                            continue

                        # Attention weights (KV-head space)
                        w_keep = prompt_scores_kv[b, h, kept].to(torch.float32)  # (k,)
                        w_disc = prompt_scores_kv[b, h, disc].to(torch.float32)  # (nd,)
                        As = w_keep.sum()
                        Ad = w_disc.sum()

                        # Per-head scale (scalar)
                        scale = self._compute_scale(Ad.view(1, 1), As.view(1, 1)).view(1)[0]

                        if debug:
                            ad_over_as[b, h] = float((Ad / (As + self.eps)).item())

                        # Absolute indices in KV cache for V slicing
                        kept_abs = kept + L_history
                        disc_abs = disc + L_history

                        X = v[b, h].index_select(dim=0, index=kept_abs).to(torch.float32)  # (k, D)
                        Y = v[b, h].index_select(dim=0, index=disc_abs).to(torch.float32)  # (nd, D)

                        # Optional centering (using kept mean)
                        if self.center:
                            mu = X.mean(dim=0, keepdim=True)
                            Xc = X - mu
                            Yc = Y - mu
                        else:
                            Xc = X
                            Yc = Y

                        # Build orthonormal basis for span(Xc) via QR on Xc^T
                        try:
                            Q, _ = torch.linalg.qr(Xc.t(), mode="reduced")  # Q: (D, r)
                        except RuntimeError:
                            continue

                        # Project Yc onto span(Xc)
                        Yproj = (Yc @ Q) @ Q.t()
                        R = Yc - Yproj  # (nd, D)

                        disc_energy = torch.linalg.norm(Yc, ord="fro")
                        resid_energy = torch.linalg.norm(R, ord="fro")

                        if debug:
                            r_perp[b, h] = resid_energy / (disc_energy + self.eps)

                        # If residual is tiny, nothing meaningful to inject
                        if resid_energy.item() <= 1e-10:
                            if debug:
                                pca_evr[b, h] = 0.0
                                inj_norm[b, h] = 0.0
                            continue

                        # PCA/SVD on residual
                        try:
                            _, S, Vh = torch.linalg.svd(R, full_matrices=False)
                        except RuntimeError:
                            continue

                        total_var = (S * S).sum() + self.eps
                        p = min(self.pca_rank, int(Vh.shape[0]))
                        if p <= 0:
                            continue

                        top_var = (S[:p] * S[:p]).sum()
                        evr = top_var / total_var
                        if debug:
                            pca_evr[b, h] = evr

                        C = Vh[:p, :]  # (p, D), orthonormal rows

                        # Attention-weighted mean of residual vectors
                        wd_sum = w_disc.sum()
                        if wd_sum.item() > self.eps:
                            wd_norm = w_disc / (wd_sum + self.eps)
                        else:
                            wd_norm = torch.full_like(w_disc, 1.0 / float(max(int(w_disc.numel()), 1)))

                        r_mean = (wd_norm.unsqueeze(0) @ R).squeeze(0)  # (D,)

                        # Restrict r_mean to top-PC residual subspace
                        coeff = r_mean @ C.t()  # (p,)
                        delta = coeff @ C       # (D,)
                        delta = delta * scale

                        # Inject into kept prompt V in the NEW cache segment [p0:p1]
                        if self.inject_mode == "uniform":
                            patch = new_v[b, h, p0:p1, :].to(torch.float32)  # (k_eff, D)
                            patch = patch + delta.unsqueeze(0)
                            new_v[b, h, p0:p1, :] = patch.to(dtype=new_v.dtype)
                        else:
                            # Distribute delta across kept prompt tokens by kept attention weights
                            wk_sum = w_keep.sum()
                            if wk_sum.item() > self.eps:
                                wk_norm = w_keep / (wk_sum + self.eps)
                            else:
                                wk_norm = torch.full_like(w_keep, 1.0 / float(max(int(w_keep.numel()), 1)))

                            add = wk_norm.unsqueeze(-1) * delta.unsqueeze(0)  # (k_eff, D)
                            patch = new_v[b, h, p0:p1, :].to(torch.float32)
                            patch = patch + add
                            new_v[b, h, p0:p1, :] = patch.to(dtype=new_v.dtype)

                        if debug:
                            inj_norm[b, h] = torch.linalg.norm(delta)

            new_layers.append((new_k, new_v))

            # Compact debug log per layer
            if debug:
                rmin, rmean, rmax = self._nanmin_mean_max(r_perp)
                evmin, evmean, evmax = self._nanmin_mean_max(pca_evr)
                admin, admean, admax = self._nanmin_mean_max(ad_over_as)
                inmin, inmean, inmax = self._nanmin_mean_max(inj_norm)

                print(
                    f"[headwise-obf] layer={layer_idx} sink={sink_len_common} k_eff={k_eff} "
                    f"L_hist={L_history} L_lat={L_latent} L_new={L_new} steps={steps} "
                    f"H_kv={H_kv} H_q={int(H_q)} "
                    f"Ad/As(min/mean/max)={admin:.4f}/{admean:.4f}/{admax:.4f} "
                    f"r_perp(min/mean/max)={rmin:.4f}/{rmean:.4f}/{rmax:.4f} "
                    f"pca_evr(min/mean/max)={evmin:.4f}/{evmean:.4f}/{evmax:.4f} "
                    f"inj_norm(min/mean/max)={inmin:.4f}/{inmean:.4f}/{inmax:.4f} "
                    f"mode={self.inject_mode} scale={self.scale_mode} center={int(self.center)}"
                )

        if apply_sink:
            self._has_kept_sink = True

        new_cache = DynamicCache.from_legacy_cache(tuple(new_layers))
        return new_cache, time.time() - t0
