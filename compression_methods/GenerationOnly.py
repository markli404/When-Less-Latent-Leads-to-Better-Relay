import time
from typing import Any, Optional, Tuple, List

import torch
from torch import Tensor
from transformers import DynamicCache

from compression_methods.BaseKVCompressor import BaseKVCompressor


def _past_length(past_key_values: Any) -> int:
    """Return KV sequence length L from the cache."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "key_cache"):
        if past_key_values.key_cache is None or len(past_key_values.key_cache) == 0:
            return 0
        return int(past_key_values.key_cache[0].shape[-2])  # (B, H, L, D)
    try:
        return int(past_key_values[0][0].shape[-2])  # legacy (k, v)
    except Exception:
        return 0


def _get_cache_device(past_key_values: Any) -> torch.device:
    """Get device of cache tensors."""
    if past_key_values is None:
        return torch.device("cpu")
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache is not None and len(past_key_values.key_cache) > 0:
        return past_key_values.key_cache[0].device
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        return past_key_values[0][0].device
    return torch.device("cpu")


def _concat_slices(past_key_values: Any, intervals: List[Tuple[int, int]]) -> Any:
    """
    Physically slice KV along the token axis (L axis) and concatenate selected intervals.
    Assumes key/value tensors are (B, H, L, D).
    """
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "key_cache"):
        legacy = tuple((k, v) for k, v in zip(past_key_values.key_cache, past_key_values.value_cache))
    else:
        legacy = past_key_values

    new_layers = []
    for (k, v) in legacy:
        chunks_k = [k[:, :, s:e, :] for (s, e) in intervals if e > s]
        chunks_v = [v[:, :, s:e, :] for (s, e) in intervals if e > s]

        if len(chunks_k) == 0:
            new_k = k[:, :, 0:0, :]
            new_v = v[:, :, 0:0, :]
        else:
            new_k = torch.cat(chunks_k, dim=-2) if len(chunks_k) > 1 else chunks_k[0]
            new_v = torch.cat(chunks_v, dim=-2) if len(chunks_v) > 1 else chunks_v[0]

        new_layers.append((new_k, new_v))

    return DynamicCache.from_legacy_cache(tuple(new_layers))


def _merge_intervals(intervals: List[Tuple[int, int]], L: int) -> List[Tuple[int, int]]:
    """Sort and merge intervals, clamp to [0, L]."""
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    for s, e in intervals:
        s = max(0, min(int(s), L))
        e = max(0, min(int(e), L))
        if e <= s:
            continue
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
    return merged


class GenerationOnly(BaseKVCompressor):
    """
    KV layout assumption:
      KV = [ history | prompt | generation ]

    Requirements:
      - latent_steps (steps) must equal the number of generation tokens appended to KV
        right before calling compress.

    Compression behavior:
      - Always keep history.
      - Keep prompt sink tokens only once (first compress call).
      - Always keep generation (the last `steps` tokens).
      - Physically slices KV (no masks).
    """

    def __init__(self, sink_size: int = 4, kv_budget: int = 32):
        super().__init__(sink_size=sink_size, kv_budget=kv_budget)
        self.reset()

    def reset(self):
        self._has_kept_sink = False

    @torch.no_grad()
    def compress(
        self,
        *,
        past_key_values: Any,
        latent_steps: int,
        prompt_mask: Optional[torch.Tensor] = None,
        all_steps_attentions: List[List[torch.Tensor]],
    ) -> tuple[Any, float, Any]:
        t0 = time.time()

        device = _get_cache_device(past_key_values)

        if past_key_values is None:
            return DynamicCache(), 0.0, None

        # steps must be exactly the number of generation tokens appended before compress
        L = _past_length(past_key_values)
        steps = int(latent_steps) if latent_steps is not None else 0
        steps = max(0, min(steps, L))

        if prompt_mask is None:
            raise ValueError("prompt_mask is required to compute boundaries for [history | prompt | generation].")

        if prompt_mask.dim() != 2:
            raise ValueError(f"prompt_mask must be 2D (B, prompt_len). Got shape {tuple(prompt_mask.shape)}")

        # Important: prompt length in KV is the padded length, not mask.sum()
        prompt_len_in_kv = int(prompt_mask.shape[1])

        # Boundaries in the current KV index space
        # KV = [0, history_end) | [history_end, history_end+prompt_len) | [history_end+prompt_len, L)
        history_end = L - prompt_len_in_kv - steps
        if history_end < 0:
            # If this happens, your assumption about KV layout or steps is violated.
            # Clamp for safety so slicing does not crash.
            history_end = 0

        prompt_start = history_end
        prompt_end = min(L, history_end + prompt_len_in_kv)

        gen_start = prompt_end
        gen_end = L

        intervals: List[Tuple[int, int]] = []

        # 1) Keep all history
        if history_end > 0:
            intervals.append((0, history_end))

        # 2) Keep sink tokens from the prompt only once
        if (not self._has_kept_sink) and self.sink_size > 0 and prompt_end > prompt_start:
            sink_end = min(prompt_start + int(self.sink_size), prompt_end)
            if sink_end > prompt_start:
                intervals.append((prompt_start, sink_end))
            self._has_kept_sink = True

        # 3) Keep generation tail (the last `steps` tokens are exactly [gen_start, gen_end))
        if gen_end > gen_start:
            intervals.append((gen_start, gen_end))

        merged = _merge_intervals(intervals, L)
        trimmed = _concat_slices(past_key_values, merged)

        return trimmed, time.time() - t0, None
