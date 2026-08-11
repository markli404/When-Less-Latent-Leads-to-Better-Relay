import time
from typing import Any, Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from compression_methods.BaseKVCompressor import BaseKVCompressor


class LOBFEvr(BaseKVCompressor):
    """
    EVR-adaptive-rank variant of LOBF (verbatim copy of LOBF.py except for the
    per-(sample, head) rank selection — repo convention is one file per variant).

    Fixed-rank LOBF uses the same pca_rank for every layer/head. Here, each
    (sample, head) instead picks the SMALLEST rank p whose cumulative explained-
    variance ratio (EVR) of the orthogonal residual reaches a target threshold
    evr_tau, capped at max_rank. Heads whose residual energy is concentrated get
    a small rank; diffuse ones get more — i.e. layer/head-dependent rank
    allocation driven by the residual spectrum itself.

    CLI convention: registered as "lobf_evr" in run.py and REUSES the --pca_rank
    flag to carry the threshold as a percentage (pca_rank=90 -> evr_tau=0.90).
    This keeps run.py / sweep scripts / wandb config plumbing unchanged. The
    per-head ranks actually chosen are logged via rank_used.
    """

    def __init__(
        self,
        sink_size: int = 4,
        kv_budget: int = 32,
        pca_rank: int = 90,               # interpreted as evr_tau percent (90 -> tau=0.90)
        eps: float = 1e-12,
        inject_mode: str = "uniform",     # {"uniform", "attn"}
        scale_mode: str = "ad_over_as",   # {"none", "ad_over_as", "ad_frac", "ad"}
        center: bool = False,             # whether to mean-center V before subspace ops
        debug_topk_heads: int = 0,        # 0 => do not print per-head topk info (keeps logs compact)
        max_rank: int = 32,               # hard cap on the adaptive rank (matches the sweep max)
    ):
        super().__init__(sink_size=int(sink_size), kv_budget=int(kv_budget))
        self.pca_rank = max(0, int(pca_rank))        # raw value kept for config logging
        self.evr_tau = float(self.pca_rank) / 100.0  # e.g. pca_rank=90 -> tau=0.90
        self.max_rank = max(1, int(max_rank))
        self.eps = float(eps)
        self.inject_mode = str(inject_mode)
        self.scale_mode = str(scale_mode)
        self.center = bool(center)
        self.debug_topk_heads = int(debug_topk_heads)

        if not (0.0 < self.evr_tau < 1.0):
            raise ValueError(
                f"lobf_evr expects --pca_rank in (0, 100) as an EVR percentage, got {self.pca_rank}"
            )
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
        # x is 1D or ND; flatten and ignore non-finite
        xf = x.reshape(-1)
        mask = torch.isfinite(xf)
        if mask.sum().item() == 0:
            return float("nan"), float("nan"), float("nan")
        vals = xf[mask]
        return float(vals.min().item()), float(vals.mean().item()), float(vals.max().item())

    def _compute_scale(self, Ad: torch.Tensor, As: torch.Tensor) -> torch.Tensor:
        """
        Ad, As are (B, H_kv) float tensors.
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

    def _aggregate_attention(
        self,
        *,
        layer_device: torch.device,
        B: int,
        H_kv: int,
        L_total: int,
        L_history: int,
        L_prompt: int,
        all_steps_attentions: List[List[torch.Tensor]],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          prompt_scores_kv: (B, H_kv, L_prompt) aggregated over steps (float32)
          full_scores_kv:   (B, H_kv, L_total)  aggregated over steps (float32), for optional diagnostics
        """
        if len(all_steps_attentions) == 0:
            prompt_scores_kv = torch.zeros((B, H_kv, L_prompt), device=layer_device, dtype=torch.float32)
            full_scores_kv = torch.zeros((B, H_kv, L_total), device=layer_device, dtype=torch.float32)
            return prompt_scores_kv, full_scores_kv

        # Query heads H_q from attention tensors
        H_q = int(all_steps_attentions[0][layer_idx].shape[1])

        # Determine maximum key length observed across steps
        max_klen = 0
        for step_data in all_steps_attentions:
            a = step_data[layer_idx]
            max_klen = max(max_klen, int(a.shape[-1]))

        # Aggregate attentions across steps into (B, H_q, max_klen)
        agg = torch.zeros((B, H_q, max_klen), device=layer_device, dtype=torch.float32)
        for step_data in all_steps_attentions:
            a = step_data[layer_idx].to(device=layer_device, dtype=torch.float32, non_blocking=True)
            if int(a.shape[-1]) < max_klen:
                a = F.pad(a, (0, max_klen - int(a.shape[-1])))
            agg += a

        # Pad/crop to L_total
        if max_klen < L_total:
            agg_full = F.pad(agg, (0, L_total - max_klen))
        else:
            agg_full = agg[:, :, :L_total]

        # Reduce query heads -> KV heads
        if H_q == H_kv:
            full_scores_kv = agg_full
        elif (H_q % H_kv) == 0:
            group = H_q // H_kv
            full_scores_kv = agg_full.view(B, H_kv, group, L_total).sum(dim=2)
        else:
            # Fallback: truncate or pad heads (rare but keeps code robust)
            if H_q >= H_kv:
                full_scores_kv = agg_full[:, :H_kv, :]
            else:
                full_scores_kv = F.pad(agg_full, (0, 0, 0, H_kv - H_q))

        prompt_scores_kv = full_scores_kv[:, :, L_history:L_history + L_prompt]
        return prompt_scores_kv, full_scores_kv

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
        prompt_mask: torch.Tensor,  # (B, L_prompt) right padded 0/1
        debug: bool = False,
        **kwargs,
    ) -> Tuple[Any, float, dict]:

        debug_store = {
            "r_perp": [],
            "recovery_R": [],
            "recovery_cos": [],
            "pca_evr": [],
            "ad_over_as": [],
            "inj_norm": [],
            "pca_cum_evr": [],
        }

        t0 = time.time()
        # Accumulator for time spent on pure-diagnostic metric computation.
        # Subtracted from the total at the end so the returned compressor time
        # reflects only the real compression work (selection + SVD + injection).
        metric_time = 0.0
        if past_key_values is None:
            return None, 0.0, None

        layers = self._as_legacy_tuple(past_key_values)
        if len(layers) == 0:
            return past_key_values, 0.0, None

        k0, _ = layers[0]
        B = int(k0.shape[0])
        num_layers = len(layers)
        H_kv = int(k0.shape[1])
        D = int(k0.shape[-1])

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

        apply_sink = (not self._has_kept_sink) and (self.sink_size > 0)

        new_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(num_layers):
            k, v = layers[layer_idx]
            layer_device = k.device

            pm = (prompt_mask > 0).to(dtype=torch.long, device=layer_device)  # (B, L_prompt)
            valid_prompt_len = pm.sum(dim=-1).to(torch.long)                  # (B,)

            # Common sink length across batch (avoid per-sample length mismatch)
            if apply_sink:
                min_valid = int(valid_prompt_len.min().item())
                sink_len_common = min(int(self.sink_size), min_valid)
            else:
                sink_len_common = 0

            # k_eff is budget for NON-sink prompt tokens (consistent with your previous setup)
            available = (valid_prompt_len - sink_len_common).clamp(min=0)
            k_eff = min(int(self.kv_budget), int(available.min().item()))
            k_eff = max(0, k_eff)

            # Aggregate attention (KV-head space)
            prompt_scores_kv, _ = self._aggregate_attention(
                layer_device=layer_device,
                B=B,
                H_kv=H_kv,
                L_total=L_total,
                L_history=L_history,
                L_prompt=L_prompt,
                all_steps_attentions=all_steps_attentions,
                layer_idx=layer_idx,
            )

            # Token scores for selection: sum over KV heads => (B, L_prompt)
            token_scores = prompt_scores_kv.sum(dim=1)  # (B, L_prompt)
            token_scores = token_scores.masked_fill(pm == 0, -float("inf"))
            if sink_len_common > 0:
                token_scores[:, :sink_len_common] = -float("inf")

            # Per-sample kept/discard prompt indices
            kept_prompt_idx: List[torch.Tensor] = []
            discard_prompt_idx: List[torch.Tensor] = []
            discard_mask_batch = torch.zeros((B, L_prompt), device=layer_device, dtype=torch.bool)

            for b in range(B):
                vp = int(valid_prompt_len[b].item())
                if k_eff <= 0 or vp <= sink_len_common:
                    kept_prompt_idx.append(torch.zeros((0,), device=layer_device, dtype=torch.long))
                    discard_prompt_idx.append(torch.zeros((0,), device=layer_device, dtype=torch.long))
                    continue

                candidates = torch.arange(sink_len_common, vp, device=layer_device, dtype=torch.long)
                scores_cand = token_scores[b, candidates]  # (n_cand,)

                k_keep = min(k_eff, int(candidates.numel()))
                if k_keep <= 0:
                    kept = torch.zeros((0,), device=layer_device, dtype=torch.long)
                else:
                    top_rel = torch.topk(scores_cand, k=k_keep, dim=-1).indices
                    kept = candidates[top_rel].to(dtype=torch.long)
                    kept, _ = torch.sort(kept)

                selected = torch.zeros((vp,), device=layer_device, dtype=torch.bool)
                if kept.numel() > 0:
                    selected[kept] = True

                discard = candidates[~selected[candidates]]
                discard = discard[token_scores[b, discard] > -float("inf")]

                kept_prompt_idx.append(kept)
                discard_prompt_idx.append(discard)
                if discard.numel() > 0:
                    discard_mask_batch[b, discard] = True

            # Build full indices in original KV coordinate: history + sink + kept + latent
            history_idx = torch.arange(0, L_history, device=layer_device, dtype=torch.long) if L_history > 0 else None
            sink_idx = torch.arange(0, sink_len_common, device=layer_device, dtype=torch.long) if sink_len_common > 0 else None
            latent_idx = (
                torch.arange(L_total - L_latent, L_total, device=layer_device, dtype=torch.long) if L_latent > 0 else None
            )

            full_indices: List[torch.Tensor] = []
            for b in range(B):
                parts = []
                if history_idx is not None:
                    parts.append(history_idx)
                if sink_idx is not None:
                    parts.append(sink_idx + L_history)
                if kept_prompt_idx[b].numel() > 0:
                    parts.append(kept_prompt_idx[b] + L_history)
                if latent_idx is not None:
                    parts.append(latent_idx)
                if len(parts) == 0:
                    full_idx = torch.zeros((0,), device=layer_device, dtype=torch.long)
                else:
                    full_idx = torch.cat(parts, dim=0).to(dtype=torch.long)
                full_indices.append(full_idx)

            kept_lens = [int(ix.numel()) for ix in full_indices]
            if len(set(kept_lens)) != 1:
                raise RuntimeError(f"Kept length is not constant across batch: {kept_lens}")

            L_new = kept_lens[0]

            # Allocate new KV
            new_k = torch.empty((B, H_kv, L_new, D), device=layer_device, dtype=k.dtype)
            new_v = torch.empty((B, H_kv, L_new, D), device=layer_device, dtype=v.dtype)

            # Diagnostics tensors per (B, H_kv)
            r_perp = torch.full((B, H_kv), float("nan"), device=layer_device, dtype=torch.float32)
            recovery_R = torch.full((B, H_kv), float("nan"), device=layer_device, dtype=torch.float32)
            recovery_cos = torch.full((B, H_kv), float("nan"), device=layer_device, dtype=torch.float32)
            pca_evr = torch.full((B, H_kv), float("nan"), device=layer_device, dtype=torch.float32)
            ub_res_pca = torch.full((B, H_kv), float("nan"), device=layer_device, dtype=torch.float32)
            ad_over_as = torch.full((B, H_kv), float("nan"), device=layer_device, dtype=torch.float32)
            inj_norm = torch.full((B, H_kv), float("nan"), device=layer_device, dtype=torch.float32)
            rank_used = torch.zeros((B, H_kv), device=layer_device, dtype=torch.float32)
            pca_cum_evr = torch.full((B, H_kv, D), float("nan"), device=layer_device, dtype=torch.float32)

            # Slice KV, then inject into kept prompt V
            prompt_kept_offset = L_history + sink_len_common
            # kept prompt count is k_eff (same across batch by construction)
            kept_count = k_eff

            for b in range(B):
                ix = full_indices[b].to(device=layer_device, dtype=torch.long, non_blocking=True)
                new_k[b] = k[b].index_select(dim=-2, index=ix)
                new_v[b] = v[b].index_select(dim=-2, index=ix)

                kept = kept_prompt_idx[b]
                disc = discard_prompt_idx[b]
                vp = int(valid_prompt_len[b].item())

                # If no disc or no kept (within the non-sink portion), skip injection
                if kept.numel() == 0 or disc.numel() == 0 or self.pca_rank <= 0:
                    # Still compute Ad/As for logging if possible
                    if kept.numel() > 0 and disc.numel() > 0:
                        w_keep = prompt_scores_kv[b, :, kept].to(torch.float32)  # (H_kv, k)
                        w_disc = prompt_scores_kv[b, :, disc].to(torch.float32)  # (H_kv, nd)
                        As = w_keep.sum(dim=-1)
                        Ad = w_disc.sum(dim=-1)
                        ad_over_as[b] = Ad / (As + self.eps)
                    continue

                # Global absolute indices
                kept_abs = kept + L_history
                disc_abs = disc + L_history

                V_keep = v[b].index_select(dim=-2, index=kept_abs.to(torch.long)).to(torch.float32)  # (H_kv, k, D)
                V_disc = v[b].index_select(dim=-2, index=disc_abs.to(torch.long)).to(torch.float32)  # (H_kv, nd, D)

                # Attention weights in KV-head space
                w_keep = prompt_scores_kv[b, :, kept].to(torch.float32)  # (H_kv, k)
                w_disc = prompt_scores_kv[b, :, disc].to(torch.float32)  # (H_kv, nd)
                As = w_keep.sum(dim=-1)  # (H_kv,)
                Ad = w_disc.sum(dim=-1)  # (H_kv,)
                _mt = time.time()
                ad_over_as[b] = Ad / (As + self.eps)
                metric_time += time.time() - _mt

                # Scale per head
                scale = self._compute_scale(Ad.unsqueeze(0), As.unsqueeze(0)).squeeze(0)  # (H_kv,)

                # Inject per head
                # We will build delta_all (H_kv, D) then add to new_v[b, :, prompt_kept_offset:prompt_kept_offset+k, :]
                delta_all = torch.zeros((H_kv, D), device=layer_device, dtype=torch.float32)

                for h in range(H_kv):
                    X = V_keep[h]  # (k, D)
                    Y = V_disc[h]  # (nd, D)

                    # Optional centering (often helps with cosine pathologies, but can change semantics)
                    if self.center:
                        mu = X.mean(dim=0, keepdim=True)
                        Xc = X - mu
                        Yc = Y - mu
                    else:
                        Xc = X
                        Yc = Y

                    # Build orthonormal basis for span(Xc) via QR on Xc^T
                    # Q: (D, r) where r <= k
                    try:
                        Q, _ = torch.linalg.qr(Xc.t(), mode="reduced")
                    except RuntimeError:
                        # Extremely rare numeric failure; skip this head
                        continue

                    # Project Yc onto span(Xc)
                    Yproj = (Yc @ Q) @ Q.t()
                    R = Yc - Yproj  # (nd, D)

                    # resid_energy is used both for the r_perp metric and the
                    # downstream edge-case gate, so keep it on the real path.
                    resid_energy = torch.linalg.norm(R, ord="fro")

                    _mt = time.time()
                    disc_energy = torch.linalg.norm(Yc, ord="fro")
                    proj_energy = torch.linalg.norm(Yproj, ord="fro")
                    r_perp[b, h] = resid_energy / (disc_energy + self.eps)
                    recovery_R[b, h] = proj_energy / (disc_energy + self.eps)
                    # Cosine between mean(Yc) and mean(Yproj)
                    my = Yc.mean(dim=0)
                    mp = Yproj.mean(dim=0)
                    denom = (torch.linalg.norm(my) * torch.linalg.norm(mp) + self.eps)
                    recovery_cos[b, h] = torch.dot(my, mp) / denom
                    metric_time += time.time() - _mt

                    # If residual is tiny, nothing meaningful to inject
                    if resid_energy.item() <= 1e-10:
                        _mt = time.time()
                        pca_evr[b, h] = 0.0
                        ub_res_pca[b, h] = 0.0
                        inj_norm[b, h] = 0.0
                        pca_cum_evr[b, h, :] = 1.0
                        metric_time += time.time() - _mt
                        continue

                    # PCA/SVD on residual
                    # R = U S Vh, principal directions are rows of Vh
                    try:
                        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
                    except RuntimeError:
                        continue

                    # EVR-adaptive rank (the only change vs LOBF): smallest p
                    # whose cumulative EVR reaches evr_tau, capped at max_rank.
                    # (cum_sel[0] >= tau -> p=1; monotone cumsum ends at ~1.0.)
                    S2_sel = S * S
                    cum_sel = torch.cumsum(S2_sel, dim=0) / (S2_sel.sum() + self.eps)
                    p = int((cum_sel < self.evr_tau).sum().item()) + 1
                    p = min(p, self.max_rank, int(Vh.shape[0]))
                    if p <= 0:
                        continue

                    # Pure-diagnostic PCA statistics. Injection only needs Vh[:p, :].
                    _mt = time.time()
                    S2 = S * S
                    total_var = S2.sum() + self.eps
                    rank_used[b, h] = float(p)
                    # Explained variance ratio for top-p components
                    top_var = S2[:p].sum()
                    evr = top_var / total_var
                    pca_evr[b, h] = evr
                    ub_res_pca[b, h] = torch.sqrt(torch.clamp(1.0 - evr, min=0.0))
                    # Cumulative EVR curve (per-rank). Pad tail with 1.0 if fewer
                    # singular values than head_dim D so the stored length is fixed.
                    cum_ratio = torch.cumsum(S2, dim=0) / total_var
                    k_sv = cum_ratio.numel()
                    pca_cum_evr[b, h, :k_sv] = cum_ratio.to(torch.float32)
                    if k_sv < D:
                        pca_cum_evr[b, h, k_sv:] = 1.0
                    metric_time += time.time() - _mt

                    C = Vh[:p, :]  # (p, D), orthonormal rows

                    # Attention-weighted mean of residual vectors
                    wd = w_disc[h]  # (nd,)
                    wd_sum = wd.sum()
                    if wd_sum.item() > self.eps:
                        wd_norm = wd / (wd_sum + self.eps)
                    else:
                        wd_norm = torch.full_like(wd, 1.0 / float(max(int(wd.numel()), 1)))

                    r_mean = (wd_norm.unsqueeze(0) @ R).squeeze(0)  # (D,)

                    # Restrict r_mean to top-PC subspace (low-rank residual content)
                    coeff = r_mean @ C.t()     # (p,)
                    delta = coeff @ C          # (D,)

                    # Apply per-head scale (Ad/As or other)
                    delta = delta * scale[h]

                    delta_all[h] = delta
                    _mt = time.time()
                    inj_norm[b, h] = torch.linalg.norm(delta)
                    metric_time += time.time() - _mt

                # Inject into kept prompt V in the NEW cache
                # NOTE: kept prompt tokens are placed contiguously after history+sink in our construction.
                if kept_count > 0:
                    # Ensure slice exists (it should, by construction)
                    start = prompt_kept_offset
                    end = prompt_kept_offset + kept_count
                    if end <= L_new:
                        if self.inject_mode == "uniform":
                            # Add the same delta vector to every kept prompt token
                            new_v[b, :, start:end, :] = new_v[b, :, start:end, :].to(torch.float32) + delta_all.unsqueeze(1)
                            new_v[b, :, start:end, :] = new_v[b, :, start:end, :].to(v.dtype)
                        else:
                            # Distribute delta across kept prompt tokens by kept attention weights (per head)
                            wk = w_keep.to(torch.float32)  # (H_kv, k)
                            wk_sum = wk.sum(dim=-1, keepdim=True)  # (H_kv, 1)
                            wk_norm = torch.where(wk_sum > self.eps, wk / (wk_sum + self.eps), torch.full_like(wk, 1.0 / float(max(int(wk.shape[-1]), 1))))
                            # Add wk_norm[h, j] * delta_all[h] to token j
                            add = wk_norm.unsqueeze(-1) * delta_all.unsqueeze(1)  # (H_kv, k, D)
                            new_v[b, :, start:end, :] = new_v[b, :, start:end, :].to(torch.float32) + add
                            new_v[b, :, start:end, :] = new_v[b, :, start:end, :].to(v.dtype)

            new_layers.append((new_k, new_v))

            # -----------------------------
            # Logging stats to WanDB
            # -----------------------------
            _mt = time.time()
            debug_store["r_perp"].append(r_perp)
            debug_store["recovery_R"].append(recovery_R)
            debug_store["recovery_cos"].append(recovery_cos)
            debug_store["pca_evr"].append(pca_evr)
            debug_store["ad_over_as"].append(ad_over_as)
            debug_store["inj_norm"].append(inj_norm)
            debug_store["pca_cum_evr"].append(pca_cum_evr)
            metric_time += time.time() - _mt

            # -----------------------------
            # Debug log (compact, decision-driving)
            # -----------------------------
            if debug:
                _mt = time.time()
                rmin, rmean, rmax = self._nanmin_mean_max(r_perp)
                rrmin, rrmean, rrmax = self._nanmin_mean_max(recovery_R)
                rcmin, rcmean, rcmax = self._nanmin_mean_max(recovery_cos)
                evmin, evmean, evmax = self._nanmin_mean_max(pca_evr)
                ubmin, ubmean, ubmax = self._nanmin_mean_max(ub_res_pca)
                admin, admean, admax = self._nanmin_mean_max(ad_over_as)
                inmin, inmean, inmax = self._nanmin_mean_max(inj_norm)

                rkmin, rkmean, rkmax = self._nanmin_mean_max(rank_used)

                print(
                    f"[pca-extra-v1] layer={layer_idx} sink_len={sink_len_common} k_eff={k_eff} "
                    f"L_hist={L_history} L_lat={L_latent} L_new={L_new} steps={steps} "
                    f"Ad/As(min/mean/max)={admin:.4f}/{admean:.4f}/{admax:.4f} "
                    f"r_perp(min/mean/max)={rmin:.4f}/{rmean:.4f}/{rmax:.4f} "
                    f"recovery_R(min/mean/max)={rrmin:.4f}/{rrmean:.4f}/{rrmax:.4f} "
                    f"recovery_cos(min/mean/max)={rcmin:.4f}/{rcmean:.4f}/{rcmax:.4f} "
                    f"pca_evr(min/mean/max)={evmin:.4f}/{evmean:.4f}/{evmax:.4f} "
                    f"ub_res_pca(min/mean/max)={ubmin:.4f}/{ubmean:.4f}/{ubmax:.4f} "
                    f"rank_used(min/mean/max)={rkmin:.1f}/{rkmean:.1f}/{rkmax:.1f} "
                    f"inj_norm(min/mean/max)={inmin:.4f}/{inmean:.4f}/{inmax:.4f} "
                    f"mode={self.inject_mode} scale={self.scale_mode} center={int(self.center)}"
                )
                metric_time += time.time() - _mt

        if apply_sink:
            self._has_kept_sink = True

        _mt = time.time()
        metadata = {
            "r_perp": torch.stack([x.detach().cpu() for x in debug_store["r_perp"]], dim=1),
            "recovery_R": torch.stack([x.detach().cpu() for x in debug_store["recovery_R"]], dim=1),
            "recovery_cos": torch.stack([x.detach().cpu() for x in debug_store["recovery_cos"]], dim=1),
            "pca_evr": torch.stack([x.detach().cpu() for x in debug_store["pca_evr"]], dim=1),
            "ad_over_as": torch.stack([x.detach().cpu() for x in debug_store["ad_over_as"]], dim=1),
            "inj_norm": torch.stack([x.detach().cpu() for x in debug_store["inj_norm"]], dim=1),
            "pca_cum_evr": torch.stack([x.detach().cpu() for x in debug_store["pca_cum_evr"]], dim=1),
        }
        metric_time += time.time() - _mt
        new_cache = DynamicCache.from_legacy_cache(tuple(new_layers))
        return new_cache, max(0.0, (time.time() - t0) - metric_time), metadata
