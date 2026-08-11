import time
from typing import Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from compression_methods.BaseKVCompressor import BaseKVCompressor


class Headwise(BaseKVCompressor):
    """
    Head-wise slicing KV compressor (no masks, no physical zeroing).

    KV layout (shared across batch):
      [ history | prompt(padded) | latent_tail ]
    - history length is shared across batch
    - latent_tail length == latent_steps (shared)
    - prompt is right padded, described by prompt_mask (B, L_prompt)

    Keep per layer:
      - keep all history
      - keep sink tokens only once (first compress call after reset) in the prompt region
      - keep exactly k_eff prompt tokens per head, per sample (exclude padding, exclude sink region)
      - keep all latent_tail

    Output:
      - new DynamicCache with shorter length (same L_new for all samples and heads)
      - elapsed time
    """

    def __init__(self, sink_size: int = 4, kv_budget: int = 32):
        super().__init__(sink_size=int(sink_size), kv_budget=int(kv_budget))

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
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()

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

        # Keep sink only once per reset
        apply_sink = (not getattr(self, "_has_kept_sink", False)) and (self.sink_size > 0)

        new_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        selected_position_matrices: List[List[List[List[int]]]] = []

        for layer_idx in range(num_layers):
            k, v = layers[layer_idx]
            device = k.device
            D = int(k.shape[-1])

            # Move prompt_mask onto this layer device
            pm = (prompt_mask > 0).to(dtype=torch.long, device=device)  # (B, L_prompt)
            valid_prompt_len = pm.sum(dim=-1).to(torch.long)            # (B,)

            if apply_sink:
                min_valid = int(valid_prompt_len.min().item())
                sink_len_common = min(int(self.sink_size), min_valid)
            else:
                sink_len_common = 0

            available = (valid_prompt_len - sink_len_common).clamp(min=0)
            k_eff = min(int(self.kv_budget), int(available.min().item()))
            k_eff = max(0, k_eff)

            # -----------------------------
            # Aggregate attention scores -> prompt_scores_kv: (B, H_kv, L_prompt)
            # -----------------------------
            if len(all_steps_attentions) == 0:
                prompt_scores_kv = torch.zeros((B, H_kv, L_prompt), device=device, dtype=torch.float32)
                H_q = H_kv
            else:
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
                    # Fallback (rare)
                    if prompt_scores_q.shape[1] >= H_kv:
                        prompt_scores_kv = prompt_scores_q[:, :H_kv, :]
                    else:
                        prompt_scores_kv = F.pad(prompt_scores_q, (0, 0, 0, H_kv - prompt_scores_q.shape[1]))

            # Exclude padding globally
            prompt_scores_kv = prompt_scores_kv.masked_fill(pm.unsqueeze(1) == 0, -float("inf"))

            # Exclude sink region from top-k selection
            if sink_len_common > 0:
                prompt_scores_kv[:, :, :sink_len_common] = -float("inf")

            # -----------------------------
            # Build per-(b, head) selected prompt indices (within prompt coordinate)
            # We keep exactly k_eff indices from the non-sink region per head.
            # -----------------------------
            # selected_prompt_idx[b]: (H_kv, k_eff)
            selected_prompt_idx: List[torch.Tensor] = []

            for b in range(B):
                vp = int(valid_prompt_len[b].item())
                if k_eff <= 0 or vp <= 0 or (vp - sink_len_common) <= 0:
                    selected_prompt_idx.append(torch.zeros((H_kv, 0), device=device, dtype=torch.long))
                    continue

                # Restrict to valid prompt length for safety
                scores_b = prompt_scores_kv[b, :, :vp]  # (H_kv, vp)

                # Now selection region is [sink_len_common : vp)
                seg = scores_b[:, sink_len_common:vp]   # (H_kv, vp-sink)
                kk = min(k_eff, int(seg.shape[-1]))
                if kk <= 0:
                    selected_prompt_idx.append(torch.zeros((H_kv, 0), device=device, dtype=torch.long))
                    continue

                topk = torch.topk(seg, k=kk, dim=-1).indices  # (H_kv, kk), per head
                topk, _ = torch.sort(topk, dim=-1)            # keep original order
                topk = topk + sink_len_common                 # shift back into [0, vp)
                selected_prompt_idx.append(topk.to(dtype=torch.long))

            # -----------------------------
            # Build head-wise full indices in ORIGINAL KV coordinate:
            # idx shape: (B, H_kv, L_new)
            # L_new = L_history + sink_len_common + k_eff + L_latent
            # -----------------------------
            L_new = L_history + sink_len_common + k_eff + L_latent
            idx = torch.empty((B, H_kv, L_new), device=device, dtype=torch.long)

            # History part (shared)
            if L_history > 0:
                hist = torch.arange(0, L_history, device=device, dtype=torch.long)  # (L_history,)
                idx[:, :, :L_history] = hist.view(1, 1, -1).expand(B, H_kv, -1)

            # Sink part (shared)
            if sink_len_common > 0:
                sink = torch.arange(0, sink_len_common, device=device, dtype=torch.long) + L_history
                s0 = L_history
                s1 = L_history + sink_len_common
                idx[:, :, s0:s1] = sink.view(1, 1, -1).expand(B, H_kv, -1)

            # Prompt selected part (head-wise)
            if k_eff > 0:
                p0 = L_history + sink_len_common
                p1 = p0 + k_eff
                for b in range(B):
                    topk = selected_prompt_idx[b]  # (H_kv, k_eff) maybe empty
                    if topk.numel() == 0:
                        # Should not happen if your "available > k_eff" assumption holds,
                        # but keep it safe.
                        idx[b, :, p0:p1] = (L_history + sink_len_common)
                    else:
                        idx[b, :, p0:p1] = topk + L_history

            # Latent tail (shared)
            if L_latent > 0:
                lat = torch.arange(L_total - L_latent, L_total, device=device, dtype=torch.long)
                l0 = L_new - L_latent
                idx[:, :, l0:] = lat.view(1, 1, -1).expand(B, H_kv, -1)

            # -----------------------------
            # Gather new_k/new_v (no masks, no zeroing)
            # k[b]: (H_kv, L_total, D)
            # We need per-head indices -> use torch.gather on dim=1 (sequence dim).
            # -----------------------------
            new_k = torch.empty((B, H_kv, L_new, D), device=device, dtype=k.dtype)
            new_v = torch.empty((B, H_kv, L_new, D), device=device, dtype=v.dtype)

            idx_exp = idx.unsqueeze(-1).expand(B, H_kv, L_new, D)  # (B,H,L_new,D)

            # Gather along sequence dimension (dim=2 for (B,H,L,D))
            new_k = torch.gather(k, dim=2, index=idx_exp)
            new_v = torch.gather(v, dim=2, index=idx_exp)

            layer_selection: List[List[List[int]]] = []
            for b in range(B):
                layer_selection.append([
                    [int(x) for x in selected_prompt_idx[b][h].detach().cpu().tolist()]
                    for h in range(H_kv)
                ])

            selected_position_matrices.append(layer_selection)
            new_layers.append((new_k, new_v))

            if debug and layer_idx == 0:
                print(
                    f"[headwise-ascore-slice] sink={sink_len_common}, k_eff={k_eff}, "
                    f"L_history={L_history}, L_prompt={L_prompt}, L_latent={L_latent}, "
                    f"L_new={L_new}, H_kv={H_kv}, H_q={int(H_q)}, device={device}"
                )

        if apply_sink:
            self._has_kept_sink = True

        selection_metadata = {
            "selected_prompt_positions_matrix": [
                [selected_position_matrices[layer_idx][b] for layer_idx in range(num_layers)]
                for b in range(B)
            ]
        }

        new_cache = DynamicCache.from_legacy_cache(tuple(new_layers))
        return new_cache, time.time() - t0, selection_metadata
