import time
from typing import Any, Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from compression_methods.BaseKVCompressor import BaseKVCompressor


class HOBFMaxP(BaseKVCompressor):
    """
    HOBF max-p variant with metric collection.

    This mirrors HOBFMetric but uses the maximum available PCA rank
    instead of truncating to self.pca_rank.
    """

    def __init__(
        self,
        sink_size: int = 4,
        kv_budget: int = 32,
        pca_rank: int = 8,
        eps: float = 1e-12,
        inject_mode: str = "uniform",
        scale_mode: str = "ad_over_as",
        center: bool = False,
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
        eps = self.eps
        if self.scale_mode == "none":
            return torch.ones_like(Ad)
        if self.scale_mode == "ad_over_as":
            return Ad / (As + eps)
        if self.scale_mode == "ad_frac":
            return Ad / (Ad + As + eps)
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
            agg_full = F.pad(agg, (0, L_total - max_klen))
        else:
            agg_full = agg[:, :, :L_total]

        prompt_scores_q = agg_full[:, :, L_history:L_history + L_prompt]

        if H_q == H_kv:
            prompt_scores_kv = prompt_scores_q
        elif (H_q % H_kv) == 0:
            group = H_q // H_kv
            prompt_scores_kv = prompt_scores_q.view(B, H_kv, group, L_prompt).sum(dim=2)
        else:
            if prompt_scores_q.shape[1] >= H_kv:
                prompt_scores_kv = prompt_scores_q[:, :H_kv, :]
            else:
                prompt_scores_kv = F.pad(prompt_scores_q, (0, 0, 0, H_kv - prompt_scores_q.shape[1]))

        return prompt_scores_kv.to(dtype=torch.float32), H_q

    @torch.no_grad()
    def compress(
        self,
        *,
        past_key_values: Any,
        latent_steps: int,
        all_steps_attentions: List[List[torch.Tensor]],
        prompt_mask: torch.Tensor,
        current_full_mask: Optional[Any] = None,
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

        apply_sink = (not getattr(self, "_has_kept_sink", False)) and (self.sink_size > 0)

        new_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(num_layers):
            k, v = layers[layer_idx]
            device = k.device
            D = int(k.shape[-1])

            pm = (prompt_mask > 0).to(dtype=torch.long, device=device)
            valid_prompt_len = pm.sum(dim=-1).to(torch.long)

            if apply_sink:
                min_valid = int(valid_prompt_len.min().item())
                sink_len_common = min(int(self.sink_size), min_valid)
            else:
                sink_len_common = 0

            available = (valid_prompt_len - sink_len_common).clamp(min=0)
            k_eff = min(int(self.kv_budget), int(available.min().item()))
            k_eff = max(0, k_eff)

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

            prompt_scores_kv = prompt_scores_kv.masked_fill(pm.unsqueeze(1) == 0, -float("inf"))
            if sink_len_common > 0:
                prompt_scores_kv[:, :, :sink_len_common] = -float("inf")

            selected_prompt_idx: List[torch.Tensor] = []
            for b in range(B):
                vp = int(valid_prompt_len[b].item())
                if k_eff <= 0 or vp <= sink_len_common:
                    selected_prompt_idx.append(torch.zeros((H_kv, 0), device=device, dtype=torch.long))
                    continue

                scores_b = prompt_scores_kv[b, :, :vp]
                seg = scores_b[:, sink_len_common:vp]
                kk = min(k_eff, int(seg.shape[-1]))
                if kk <= 0:
                    selected_prompt_idx.append(torch.zeros((H_kv, 0), device=device, dtype=torch.long))
                    continue

                topk_rel = torch.topk(seg, k=kk, dim=-1).indices
                topk_rel, _ = torch.sort(topk_rel, dim=-1)
                topk = topk_rel + sink_len_common
                selected_prompt_idx.append(topk.to(dtype=torch.long))

            L_new = L_history + sink_len_common + k_eff + L_latent
            idx = torch.empty((B, H_kv, L_new), device=device, dtype=torch.long)

            if L_history > 0:
                hist = torch.arange(0, L_history, device=device, dtype=torch.long)
                idx[:, :, :L_history] = hist.view(1, 1, -1).expand(B, H_kv, -1)

            if sink_len_common > 0:
                sink = torch.arange(0, sink_len_common, device=device, dtype=torch.long) + L_history
                s0 = L_history
                s1 = L_history + sink_len_common
                idx[:, :, s0:s1] = sink.view(1, 1, -1).expand(B, H_kv, -1)

            p0 = L_history + sink_len_common
            p1 = p0 + k_eff
            if k_eff > 0:
                for b in range(B):
                    topk = selected_prompt_idx[b]
                    if topk.numel() == 0:
                        idx[b, :, p0:p1] = (L_history + sink_len_common)
                    else:
                        idx[b, :, p0:p1] = topk + L_history

            if L_latent > 0:
                lat = torch.arange(L_total - L_latent, L_total, device=device, dtype=torch.long)
                l0 = L_new - L_latent
                idx[:, :, l0:] = lat.view(1, 1, -1).expand(B, H_kv, -1)

            idx_exp = idx.unsqueeze(-1).expand(B, H_kv, L_new, D)
            new_k = torch.gather(k, dim=2, index=idx_exp)
            new_v = torch.gather(v, dim=2, index=idx_exp)

            r_perp = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
            recovery_R = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
            recovery_cos = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
            pca_evr = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
            ad_over_as = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
            inj_norm = torch.full((B, H_kv), float("nan"), device=device, dtype=torch.float32)
            pca_cum_evr = torch.full((B, H_kv, D), float("nan"), device=device, dtype=torch.float32)

            if self.pca_rank > 0 and k_eff > 0:
                for b in range(B):
                    vp = int(valid_prompt_len[b].item())
                    if vp <= sink_len_common:
                        continue

                    candidates = torch.arange(sink_len_common, vp, device=device, dtype=torch.long)

                    topk_b = selected_prompt_idx[b]
                    if topk_b.numel() == 0:
                        continue

                    for h in range(H_kv):
                        kept = topk_b[h]
                        if kept.numel() == 0:
                            continue

                        selected = torch.zeros((vp,), device=device, dtype=torch.bool)
                        selected[kept] = True
                        disc = candidates[~selected[candidates]]
                        if disc.numel() == 0:
                            wk = prompt_scores_kv[b, h, kept].to(torch.float32)
                            As = wk.sum()
                            ad_over_as[b, h] = float("nan") if As.item() == 0 else 0.0
                            continue

                        w_keep = prompt_scores_kv[b, h, kept].to(torch.float32)
                        w_disc = prompt_scores_kv[b, h, disc].to(torch.float32)
                        As = w_keep.sum()
                        Ad = w_disc.sum()
                        _mt = time.time()
                        ad_over_as[b, h] = Ad / (As + self.eps)
                        metric_time += time.time() - _mt

                        scale = self._compute_scale(Ad.view(1, 1), As.view(1, 1)).view(1)[0]

                        kept_abs = kept + L_history
                        disc_abs = disc + L_history

                        X = v[b, h].index_select(dim=0, index=kept_abs).to(torch.float32)
                        Y = v[b, h].index_select(dim=0, index=disc_abs).to(torch.float32)

                        if self.center:
                            mu = X.mean(dim=0, keepdim=True)
                            Xc = X - mu
                            Yc = Y - mu
                        else:
                            Xc = X
                            Yc = Y

                        try:
                            Q, _ = torch.linalg.qr(Xc.t(), mode="reduced")
                        except RuntimeError:
                            continue

                        Yproj = (Yc @ Q) @ Q.t()
                        R = Yc - Yproj

                        resid_energy = torch.linalg.norm(R, ord="fro")

                        _mt = time.time()
                        disc_energy = torch.linalg.norm(Yc, ord="fro")
                        proj_energy = torch.linalg.norm(Yproj, ord="fro")
                        r_perp[b, h] = resid_energy / (disc_energy + self.eps)
                        recovery_R[b, h] = proj_energy / (disc_energy + self.eps)
                        my = Yc.mean(dim=0)
                        mp = Yproj.mean(dim=0)
                        denom = (torch.linalg.norm(my) * torch.linalg.norm(mp) + self.eps)
                        recovery_cos[b, h] = torch.dot(my, mp) / denom
                        metric_time += time.time() - _mt

                        if resid_energy.item() <= 1e-10:
                            _mt = time.time()
                            pca_evr[b, h] = 0.0
                            pca_cum_evr[b, h, :] = 1.0
                            inj_norm[b, h] = 0.0
                            metric_time += time.time() - _mt
                            continue

                        try:
                            _, S, Vh = torch.linalg.svd(R, full_matrices=False)
                        except RuntimeError:
                            continue

                        p = int(Vh.shape[0])
                        if p <= 0:
                            continue

                        _mt = time.time()
                        total_var = (S * S).sum() + self.eps
                        top_var = (S[:p] * S[:p]).sum()
                        pca_evr[b, h] = top_var / total_var
                        # Cumulative EVR curve (per-rank). Pad tail with 1.0 if fewer
                        # singular values than head_dim D so the stored length is fixed.
                        cum_ratio = torch.cumsum(S * S, dim=0) / total_var
                        k_sv = cum_ratio.numel()
                        pca_cum_evr[b, h, :k_sv] = cum_ratio.to(torch.float32)
                        if k_sv < D:
                            pca_cum_evr[b, h, k_sv:] = 1.0
                        metric_time += time.time() - _mt

                        C = Vh[:p, :]

                        wd_sum = w_disc.sum()
                        if wd_sum.item() > self.eps:
                            wd_norm = w_disc / (wd_sum + self.eps)
                        else:
                            wd_norm = torch.full_like(w_disc, 1.0 / float(max(int(w_disc.numel()), 1)))

                        r_mean = (wd_norm.unsqueeze(0) @ R).squeeze(0)
                        coeff = r_mean @ C.t()
                        delta = coeff @ C
                        delta = delta * scale

                        if self.inject_mode == "uniform":
                            patch = new_v[b, h, p0:p1, :].to(torch.float32)
                            patch = patch + delta.unsqueeze(0)
                            new_v[b, h, p0:p1, :] = patch.to(dtype=new_v.dtype)
                        else:
                            wk_sum = w_keep.sum()
                            if wk_sum.item() > self.eps:
                                wk_norm = w_keep / (wk_sum + self.eps)
                            else:
                                wk_norm = torch.full_like(w_keep, 1.0 / float(max(int(w_keep.numel()), 1)))

                            add = wk_norm.unsqueeze(-1) * delta.unsqueeze(0)
                            patch = new_v[b, h, p0:p1, :].to(torch.float32)
                            patch = patch + add
                            new_v[b, h, p0:p1, :] = patch.to(dtype=new_v.dtype)

                        _mt = time.time()
                        inj_norm[b, h] = torch.linalg.norm(delta)
                        metric_time += time.time() - _mt

            new_layers.append((new_k, new_v))

            _mt = time.time()
            debug_store["r_perp"].append(r_perp)
            debug_store["recovery_R"].append(recovery_R)
            debug_store["recovery_cos"].append(recovery_cos)
            debug_store["pca_evr"].append(pca_evr)
            debug_store["ad_over_as"].append(ad_over_as)
            debug_store["inj_norm"].append(inj_norm)
            debug_store["pca_cum_evr"].append(pca_cum_evr)
            metric_time += time.time() - _mt

            if debug:
                _mt = time.time()
                rmin, rmean, rmax = self._nanmin_mean_max(r_perp)
                rrmin, rrmean, rrmax = self._nanmin_mean_max(recovery_R)
                rcmin, rcmean, rcmax = self._nanmin_mean_max(recovery_cos)
                evmin, evmean, evmax = self._nanmin_mean_max(pca_evr)
                admin, admean, admax = self._nanmin_mean_max(ad_over_as)
                inmin, inmean, inmax = self._nanmin_mean_max(inj_norm)

                print(
                    f"[headwise-obf-metric] layer={layer_idx} sink={sink_len_common} k_eff={k_eff} "
                    f"L_hist={L_history} L_lat={L_latent} L_new={L_new} steps={steps} "
                    f"H_kv={H_kv} H_q={int(H_q)} "
                    f"Ad/As(min/mean/max)={admin:.4f}/{admean:.4f}/{admax:.4f} "
                    f"r_perp(min/mean/max)={rmin:.4f}/{rmean:.4f}/{rmax:.4f} "
                    f"recovery_R(min/mean/max)={rrmin:.4f}/{rrmean:.4f}/{rrmax:.4f} "
                    f"recovery_cos(min/mean/max)={rcmin:.4f}/{rcmean:.4f}/{rcmax:.4f} "
                    f"pca_evr(min/mean/max)={evmin:.4f}/{evmean:.4f}/{evmax:.4f} "
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
