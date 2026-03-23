from typing import Dict, List, Optional, Tuple
import time

from . import default_agents
from models import ModelWrapper
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
from compression_methods import *

import torch
import argparse
import gc

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        compressor = None,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if compressor is None:
            compressor = Full()
        self.compressor = compressor

        if self.latent_only:
            self.sequential_info_only = True

        self.task = args.task

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        device = self.model.device

        # KV / mask state (assumed to be precomputed correctly by your pipeline)
        past_kv: Optional[Tuple] = None

        # External position cursor (one per sample)
        pos_cursor = torch.zeros(batch_size, device=self.model.device, dtype=torch.long)

        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = [""] * batch_size

        all_communication_overhead = 0
        all_compression_time = 0.0
        latent_inference_time = 0.0
        text_inference_time = 0.0
        prompt_len = [0] * batch_size
        token_usage = []
        peak_overhead = 0

        metadatas = []

        self.compressor.reset()
        # ----------------------------
        # Helpers
        # ----------------------------
        def _build_batch_messages(agent, items_: List[Dict]) -> List[List[Dict]]:
            if self.args.prompt == "sequential":
                return [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items_
                ]
            if self.args.prompt == "hierarchical":
                return [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items_
                ]
            raise ValueError(f"Unknown prompt style: {self.args.prompt}")

        def _maybe_add_think(prompts: List[str]) -> List[str]:
            if getattr(self.args, "think", False):
                return [f"{p}<think>" for p in prompts]
            return prompts

        def _encode_prompts(prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
            enc = self.model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            tokens_batch: List[List[str]] = []
            for ids_row, mask_row in zip(ids, mask):
                active_ids = ids_row[mask_row.bool()].tolist()
                tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
            return ids, mask, tokens_batch

        def _append_trace(
                idx: int,
                agent,
                prompt: str,
                input_ids: torch.Tensor,
                input_tokens: List[str],
                output: str,
                *,
                latent_steps: Optional[int] = None,
        ) -> None:
            entry = {
                "name": agent.name,
                "role": agent.role,
                "input": prompt,
                "input_ids": input_ids,
                "input_tokens": input_tokens,
                "output": output,
            }
            if latent_steps is not None:
                entry["latent_steps"] = latent_steps
            agent_traces[idx].append(entry)

        def _score_item(task: str, final_text: str, item: Dict) -> Tuple[bool, str, str, Optional[str]]:
            """
            returns: ok, pred, gold, error_msg
            """
            if task in ["mbppplus", "humanevalplus"]:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                if pred is None:
                    return False, "", gold, "python error: No python code block found"
                python_code_to_exe = pred + "\n" + gold
                ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                return ok, pred, gold, error_msg

            if task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    ok = (int(pred) == int(gold))
                    return ok, pred, gold, None
                except (ValueError, TypeError):
                    return False, pred, gold, f"Value/Type error in parsing answer. Pred: {pred}, Gold: {gold}"

            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item.get("gold", "")
            ok = (pred == gold) if (pred and gold) else False
            return ok, pred, gold, None

        # ----------------------------
        # Main loop over agents
        # ----------------------------
        for agent in self.agents:
            batch_messages = _build_batch_messages(agent, items)
            prompts, _, _, _ = self.model.prepare_chat_batch(batch_messages, add_generation_prompt=True)

            if agent.role != "judger":
                wrapped_prompts = _maybe_add_think(prompts)
                wrapped_ids, wrapped_mask, wrapped_tokens_batch = _encode_prompts(wrapped_prompts)

                # wrapped_mask: (B, L), dtype could be bool / int
                if wrapped_mask.dtype == torch.bool:
                    lengths = wrapped_mask.sum(dim=1).to(torch.long)  # (B,)
                else:
                    lengths = (wrapped_mask > 0).sum(dim=1).to(torch.long)  # (B,)
                lengths_list = lengths.detach().cpu().tolist()
                for i in range(len(prompt_len)):
                    prompt_len[i] += lengths_list[i]

                # ---- latent forward ----
                start = time.time()
                past_kv, final_attention, pos_cursor = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,  # 2D only
                    pos_cursor=pos_cursor,  # (B,)
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )

                latent_inference_time += time.time() - start

                # ---- compress ----
                past_kv, compress_time, metadata = self.compressor.compress(
                    past_key_values=past_kv,
                    latent_steps=self.latent_steps,
                    prompt_mask=wrapped_mask,
                    all_steps_attentions=final_attention,
                )
                all_compression_time += compress_time
                metadatas.append(metadata)
                kv_size = self.kv_size_mb(past_kv)
                all_communication_overhead += kv_size
                peak_overhead = max(peak_overhead, kv_size)

                # ---- tracing ----
                for i in range(batch_size):
                    m = wrapped_mask[i].bool()
                    trimmed_ids = wrapped_ids[i][m].to("cpu").tolist()
                    _append_trace(
                        idx=i,
                        agent=agent,
                        prompt=wrapped_prompts[i],
                        input_ids=trimmed_ids,
                        input_tokens=wrapped_tokens_batch[i],
                        output="",
                        latent_steps=self.latent_steps,
                    )

            else:
                # Judger decoding stage
                past_for_decoding = past_kv if self.latent_steps > 0 else None
                judger_prompts = _maybe_add_think(prompts)
                judger_ids, judger_mask, judger_tokens_batch = _encode_prompts(judger_prompts)

                start = time.time()
                generated_batch, past_kv, pos_cursor, token_usage = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    pos_cursor=pos_cursor,  # (B,)
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )

                text_inference_time += time.time() - start

                # If you need to continue after judger, uncomment the next line:
                # past_pos = decode_start_pos  # plus generated lengths if you track them
                for i in range(batch_size):
                    final_text = generated_batch[i].strip()
                    final_texts[i] = final_text

                    m = judger_mask[i].bool()
                    trimmed_ids = judger_ids[i][m].to("cpu").tolist()
                    _append_trace(
                        idx=i,
                        agent=agent,
                        prompt=judger_prompts[i],
                        input_ids=trimmed_ids,
                        input_tokens=judger_tokens_batch[i],
                        output=final_text,
                    )

                    # print(final_text)
                    # print("=" * 40)

        # ----------------------------
        # Package results
        # ----------------------------

        obf_metrics = {}
        if metadata is not None:
            for round_idx, meta in enumerate(metadatas):
                obf_metrics[f"r_perp_{round_idx}"] = meta["r_perp"][i] if meta is not None else None
                obf_metrics[f"recovery_R_{round_idx}"] = meta["recovery_R"][i] if meta is not None else None
                obf_metrics[f"recovery_cos_{round_idx}"] = meta["recovery_cos"][i] if meta is not None else None
                obf_metrics[f"pca_evr_{round_idx}"] = meta["pca_evr"][i] if meta is not None else None
                obf_metrics[f"ad_over_as_{round_idx}"] = meta["ad_over_as"][i] if meta is not None else None
                obf_metrics[f"inj_norm_{round_idx}"] = meta["inj_norm"][i] if meta is not None else None

        results: List[Dict] = []
        for i, item in enumerate(items):
            ok, pred, gold, error_msg = _score_item(self.task, final_texts[i], item)
            if error_msg:
                print(error_msg)

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_texts[i],
                    "agents": agent_traces[i],
                    "correct": ok,
                    "communication_overhead": all_communication_overhead / batch_size,
                    "compression_time": all_compression_time / batch_size,
                    "latent_inference_time": latent_inference_time / batch_size,
                    "text_inference_time": text_inference_time / batch_size,
                    "prompt_len": prompt_len[i],
                    "token_usage": token_usage[i],
                    "peak_overhead": peak_overhead,

                    # OBF metric
                    **obf_metrics,

                }
            )

        # Cleanup
        del past_kv
        gc.collect()
        torch.cuda.empty_cache()
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]

    @staticmethod
    def kv_size_mb(past_kv) -> float:
        if past_kv is None:
            return 0.0

        total_bytes = 0

        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()
            return total_bytes / (1024 ** 2)

        # legacy
        for layer in past_kv:
            for t in layer:
                if torch.is_tensor(t):
                    total_bytes += t.numel() * t.element_size()
        return total_bytes / (1024 ** 2)
