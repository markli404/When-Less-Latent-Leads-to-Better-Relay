import time
from typing import Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from compression_methods.BaseKVCompressor import BaseKVCompressor


class Layerwise(BaseKVCompressor):
    """
    Physical KV slicing with a fixed per-sample budget (no cache padding, no masks).

    KV layout (shared across batch):
      [ history | prompt(padded) | latent_tail ]
    - history length is shared across batch
    - latent_tail length == latent_steps (shared)
    - prompt is right padded, described by prompt_mask (B, L_prompt)

    Per layer keep:
      - keep all history
      - keep sink tokens only once (first compress call after reset) in the prompt region
      - keep exactly k_eff prompt tokens per sample (exclude padding, exclude sink region)
      - keep all latent_tail

    Note:
      - k_eff is batch-constant so the kept KV length is identical across samples.
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
    ) -> Tuple[Any, float]:
        t0 = time.time()

        # Do NOT create a CPU DynamicCache here.
        if past_key_values is None:
            return None, 0.0

        layers = self._as_legacy_tuple(past_key_values)
        if len(layers) == 0:
            # Keep it as an empty DynamicCache on the same device is tricky; just return as-is.
            # Upstream should treat this as "no past".
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

        # Head count in attention (query heads)
        H_q = None
        if len(all_steps_attentions) > 0 and len(all_steps_attentions[0]) > 0:
            H_q = int(all_steps_attentions[0][0].shape[1])

        # Sink decision (batch-constant)
        # valid_prompt_len excludes padding.
        # We compute it per layer device later to avoid CPU fallback issues.
        apply_sink = (not self._has_kept_sink) and (self.sink_size > 0)

        new_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(num_layers):
            k, v = layers[layer_idx]
            layer_device = k.device

            # Move prompt_mask onto this layer device
            pm = (prompt_mask > 0).to(dtype=torch.long, device=layer_device)

            valid_prompt_len = pm.sum(dim=-1).to(torch.long)  # (B,)

            if apply_sink:
                min_valid = int(valid_prompt_len.min().item())
                sink_len_common = min(int(self.sink_size), min_valid)
            else:
                sink_len_common = 0

            available = (valid_prompt_len - sink_len_common).clamp(min=0)
            k_eff = min(int(self.kv_budget), int(available.min().item()))
            k_eff = max(0, k_eff)

            # Aggregate attention scores for this layer over latent steps
            if len(all_steps_attentions) == 0:
                prompt_scores_kv = torch.zeros((B, H_kv, L_prompt), device=layer_device, dtype=torch.float32)
            else:
                if H_q is None:
                    raise RuntimeError("all_steps_attentions provided but cannot infer H_q")

                max_klen = 0
                for step_data in all_steps_attentions:
                    a = step_data[layer_idx]
                    max_klen = max(max_klen, int(a.shape[-1]))

                agg = torch.zeros((B, H_q, max_klen), device=layer_device, dtype=torch.float32)

                for step_data in all_steps_attentions:
                    a = step_data[layer_idx]
                    # Force attention onto the same device as KV
                    a = a.to(device=layer_device, dtype=torch.float32, non_blocking=True)

                    if int(a.shape[-1]) < max_klen:
                        a = F.pad(a, (0, max_klen - int(a.shape[-1])))
                    agg += a

                if max_klen < L_total:
                    agg_full = F.pad(agg, (0, L_total - max_klen))  # (B,H_q,L_total)
                else:
                    agg_full = agg[:, :, :L_total]

                prompt_scores_q = agg_full[:, :, L_history:L_history + L_prompt]  # (B,H_q,L_prompt)

                # Reduce query heads -> kv heads if needed (GQA)
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

            # Token scores for selecting a single index set per sample
            token_scores = prompt_scores_kv.sum(dim=1)  # (B,L_prompt)

            # Exclude padding
            token_scores = token_scores.masked_fill(pm == 0, -float("inf"))

            # Exclude sink region
            if sink_len_common > 0:
                token_scores[:, :sink_len_common] = -float("inf")

            # Per-sample top-k within valid region
            kept_prompt_idx: List[torch.Tensor] = []
            for b in range(B):
                vp = int(valid_prompt_len[b].item())
                if k_eff <= 0 or vp <= 0:
                    idx = torch.zeros((0,), device=layer_device, dtype=torch.long)
                else:
                    scores_b = token_scores[b, :vp]
                    topk = torch.topk(scores_b, k=k_eff, dim=-1).indices.to(dtype=torch.long)
                    topk, _ = torch.sort(topk)
                    idx = topk
                kept_prompt_idx.append(idx)

            # Build full indices in the ORIGINAL KV coordinate
            history_idx = torch.arange(0, L_history, device=layer_device, dtype=torch.long) if L_history > 0 else None
            sink_idx = torch.arange(0, sink_len_common, device=layer_device, dtype=torch.long) if sink_len_common > 0 else None
            latent_idx = torch.arange(L_total - L_latent, L_total, device=layer_device, dtype=torch.long) if L_latent > 0 else None

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

            # Slice KV
            L_new = kept_lens[0]
            D = int(k.shape[-1])

            new_k = torch.empty((B, H_kv, L_new, D), device=layer_device, dtype=k.dtype)
            new_v = torch.empty((B, H_kv, L_new, D), device=layer_device, dtype=v.dtype)

            for b in range(B):
                ix = full_indices[b]
                # Ensure index on the same device
                ix = ix.to(device=layer_device, dtype=torch.long, non_blocking=True)
                new_k[b] = k[b].index_select(dim=-2, index=ix)
                new_v[b] = v[b].index_select(dim=-2, index=ix)

            new_layers.append((new_k, new_v))

            if debug and layer_idx == 0:
                print(
                    f"[ascore] sink={sink_len_common}, k_eff={k_eff}, "
                    f"L_history={L_history}, L_latent={L_latent}, L_new={L_new}, device={layer_device}"
                )

        if apply_sink:
            self._has_kept_sink = True

        new_cache = DynamicCache.from_legacy_cache(tuple(new_layers))
        return new_cache, time.time() - t0
