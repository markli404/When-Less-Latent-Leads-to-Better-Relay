from __future__ import annotations

from typing import Any, Optional, Tuple, List

import torch

class BaseKVCompressor:
    """
    Base class that only computes prompt boundaries from:
      - past_key_values (already trimmed, no padding inside KV)
      - prompt_mask (2D, marks which prompt tokens are real vs padding)

    It does NOT perform compression.
    """

    def __init__(self, sink_size: int = 4, kv_budget: int = 32):
        self.sink_size = int(sink_size)
        self.kv_budget = int(kv_budget)
        self._has_kept_sink = False

    def reset(self):
        self._has_kept_sink = False

    def compress(
        self,
        *,
        past_key_values: Any,
        latent_steps: int,
        all_steps_attentions: Any,
        prompt_mask: torch.Tensor,
        current_full_mask: Optional[Any] = None,
    ):
        """
        Placeholder to match your call-site signature:

            past_kv, past_mask, compress_time = self.compressor.compress(
                past_key_values=past_kv,
                latent_steps=self.latent_steps,
                all_steps_attentions=final_attention,
                prompt_mask=wrapped_mask,
                current_full_mask=current_full_mask,
            )

        Subclasses should implement the real compression logic.
        """
        raise NotImplementedError("compress() must be implemented in a subclass.")
