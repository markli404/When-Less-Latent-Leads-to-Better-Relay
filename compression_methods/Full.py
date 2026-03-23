import time
from typing import Any, Optional

import torch
from transformers import DynamicCache

from compression_methods.BaseKVCompressor import BaseKVCompressor


class Full(BaseKVCompressor):
    """
    Full KV compressor:
      - Keep everything
      - Do not slice, do not mask
      - Return KV as-is
    """

    def __init__(self, sink_size: int = 0, kv_budget: int = 0):
        super().__init__(sink_size=sink_size, kv_budget=kv_budget)


    @torch.no_grad()
    def compress(
        self,
        *,
        past_key_values: Any,
        latent_steps: int,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        t0 = time.time()

        # If no cache, return an empty DynamicCache to keep downstream code consistent
        if past_key_values is None:
            return DynamicCache(), time.time() - t0

        # Return KV unchanged
        return past_key_values, time.time() - t0, None
