import time
from typing import Any, Optional, Tuple

import torch
from transformers import DynamicCache

from compression_methods.BaseKVCompressor import BaseKVCompressor


class FullQuant(BaseKVCompressor):
    """
    KV-quantization baseline: full relay (keep every position) with the relayed
    KV fake-quantized to `quant_bits` at the relay boundary.

    Rationale: quantization compresses BITS PER
    TOKEN while eviction/OBF compress the NUMBER OF TOKENS — orthogonal axes.
    This baseline gives the direct comparison at matched memory (int4 full
    length = 25% of bf16 bytes, vs OBF's rho = 10-20% of positions), and
    LOBFQuant demonstrates the two axes stack.

    Quantization scheme (KIVI-informed, asymmetric min-max fake quant):
      - K: per-(head, channel) statistics over the TOKEN axis
      - V: per-(head, token) statistics over the CHANNEL axis
    Compute stays in the original dtype (quantize -> dequantize), which models
    the information loss of an n-bit relay without a custom kernel.

    NOTE on realism: the whole cache is re-quantized at EVERY relay hop, so
    quantization error compounds across agents — exactly what happens in a real
    relay pipeline where each hop transmits an n-bit payload.

    NOTE on logging: latent_mas.py detects the `quant_bits` attribute and logs
    avg_communication_MB as the TRUE n-bit payload (numel*bits/8 + per-group
    fp16 scale/zero metadata) — no analytic correction needed downstream.
    """

    def __init__(self, sink_size: int = 0, kv_budget: int = 0, quant_bits: int = 8):
        super().__init__(sink_size=sink_size, kv_budget=kv_budget)
        self.quant_bits = int(quant_bits)
        if self.quant_bits not in (2, 4, 8):
            raise ValueError(f"quant_bits must be one of 2/4/8, got {self.quant_bits}")

    @staticmethod
    def _as_legacy_tuple(past_key_values: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        if past_key_values is None:
            return tuple()
        if hasattr(past_key_values, "key_cache"):
            return tuple((k, v) for k, v in zip(past_key_values.key_cache, past_key_values.value_cache))
        return past_key_values

    @staticmethod
    def _fake_quant(x: torch.Tensor, dim: int, bits: int) -> torch.Tensor:
        """Asymmetric min-max fake quantization; statistics reduced over `dim`."""
        levels = float(2 ** bits - 1)
        xf = x.to(torch.float32)
        mn = xf.amin(dim=dim, keepdim=True)
        mx = xf.amax(dim=dim, keepdim=True)
        scale = ((mx - mn) / levels).clamp_min(1e-10)
        q = torch.clamp(torch.round((xf - mn) / scale), 0.0, levels)
        return (q * scale + mn).to(x.dtype)

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

        if past_key_values is None:
            return DynamicCache(), time.time() - t0, None

        layers = self._as_legacy_tuple(past_key_values)
        if len(layers) == 0:
            return past_key_values, time.time() - t0, None

        new_layers = []
        for k, v in layers:
            # K: stats over tokens (dim=-2) -> per-(head, channel) params.
            # V: stats over channels (dim=-1) -> per-(head, token) params.
            kq = self._fake_quant(k, dim=-2, bits=self.quant_bits)
            vq = self._fake_quant(v, dim=-1, bits=self.quant_bits)
            new_layers.append((kq, vq))

        new_cache = DynamicCache.from_legacy_cache(tuple(new_layers))
        return new_cache, time.time() - t0, None
