import torch
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

transformers.logging.set_verbosity_error()


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

def _past_length(past_key_values: Any) -> int:
    """
    Return trimmed KV length (shared across batch).
    Supports:
      - Cache objects with key_cache
      - Legacy tuple past_key_values
    """
    if past_key_values is None:
        return 0

    if hasattr(past_key_values, "key_cache"):
        key_cache = getattr(past_key_values, "key_cache")
        if key_cache is None or len(key_cache) == 0:
            return 0
        return int(key_cache[0].shape[-2])

    try:
        return int(past_key_values[0][0].shape[-2])
    except Exception:
        return 0

class ModelWrapper:
    """
    Minimal wrapper (transformers only) with:
      - generate_text_batch (Route B, external pos_cursor)
      - generate_latent_batch (Route B, external pos_cursor)
    Assumptions (as you requested):
      1) past_key_values and past_mask/past_attention_mask are already computed and aligned.
      2) You maintain `pos_cursor` OUTSIDE and pass it in.
      3) Padding is only for batch alignment and should NOT advance logical position timeline.
    """

    def __init__(self, model_name: str, device: torch.device, args=None):
        self.model_name = model_name
        self.device = device
        self.args = args
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.pre_aligned = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)

        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                attn_implementation="eager",
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(device).eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

    def _build_latent_realign_matrix(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")

        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        # Match the old behavior: even when realignment is disabled, keep norm alignment.
        if not self.latent_space_realign:
            realign_matrix = torch.eye(
                realign_matrix.shape[0],
                device=realign_matrix.device,
                dtype=realign_matrix.dtype,
            )

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        if isinstance(target_norm, torch.Tensor):
            target_norm = target_norm.to(device=target_device, dtype=matrix.dtype)
        else:
            target_norm = torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)

        self._latent_realign_matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    def _apply_latent_realignment(
        self,
        hidden: torch.Tensor,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.pre_aligned = aligned.detach().clone()
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _gather_row(mask4d: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
        # mask4d: (B, H, Q, K), row_idx: (B,)
        B, H, Q, K = mask4d.shape
        b_idx = torch.arange(B, device=mask4d.device)
        return mask4d[b_idx, :, row_idx, :].unsqueeze(2)  # (B,H,1,K)

    @staticmethod
    def _top_p_sample(probs: torch.Tensor, p: float) -> torch.Tensor:
        # probs: (B, V), returns (B,1)
        if p >= 1.0:
            return torch.multinomial(probs, 1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= p
        keep[:, 0] = True

        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_in_sorted = torch.multinomial(filtered, 1)
        next_token = sorted_idx.gather(-1, sampled_in_sorted)
        return next_token

    from typing import Any, Optional, Tuple, List
    import torch

    def _past_length(past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "key_cache"):
            if len(past_key_values.key_cache) == 0:
                return 0
            return int(past_key_values.key_cache[0].shape[-2])
        try:
            return int(past_key_values[0][0].shape[-2])
        except Exception:
            return 0

    @torch.no_grad()
    def generate_latent_batch(
            self,
            input_ids: torch.Tensor,  # (B, L)
            attention_mask: Optional[torch.Tensor] = None,  # (B, L) 0/1, right padded
            *,
            pos_cursor: torch.Tensor,  # (B,) absolute pos for first REAL token in this prompt window
            latent_steps: int,
            past_key_values: Optional[Any] = None,
    ) -> Tuple[Any, List[List[torch.Tensor]], torch.Tensor]:
        """
        - Uses full 2D attention_mask with length (past_len + cur_len [+ steps]).
        - position_ids are computed from pos_cursor (pads get 0).
        - Latent steps are generated in a for-loop.
        Returns:
          past_key_values, all_steps_attentions, new_pos_cursor
        """

        device = input_ids.device
        B, L = input_ids.shape

        if pos_cursor is None:
            pos_cursor = torch.zeros((B,), device=input_ids.device, dtype=torch.long)
        elif pos_cursor.dim() == 0:
            pos_cursor = pos_cursor.expand(B)
        elif pos_cursor.numel() == 1 and B > 1:
            pos_cursor = pos_cursor.expand(B)
        elif pos_cursor.size(0) != B:
            raise ValueError(f"pos_cursor size mismatch: {pos_cursor.size(0)} vs B={B}")

        if attention_mask is None:
            attention_mask = torch.ones((B, L), dtype=torch.long, device=device)
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.to(torch.long)
            attention_mask = (attention_mask > 0).to(torch.long).to(device)

        valid_len = attention_mask.sum(dim=-1).to(torch.long)  # (B,)
        last_valid_idx = (valid_len - 1).clamp(min=0)  # (B,)

        past_len = _past_length(past_key_values)
        if past_len > 0:
            past_mask = torch.ones((B, past_len), dtype=torch.long, device=device)
            full_mask = torch.cat([past_mask, attention_mask], dim=-1)  # (B, past_len + L)
        else:
            full_mask = attention_mask  # (B, L)

        pos_cursor = pos_cursor.to(device=device, dtype=torch.long)
        token_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        prompt_pos_ids = pos_cursor.unsqueeze(1) + token_pos  # (B, L)
        prompt_pos_ids = torch.where(attention_mask.bool(), prompt_pos_ids, torch.zeros_like(prompt_pos_ids))

        out = self.model(
            input_ids=input_ids,
            attention_mask=full_mask,  # IMPORTANT: full length
            position_ids=prompt_pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = out.past_key_values

        last_hidden = out.hidden_states[-1][torch.arange(B, device=device), last_valid_idx, :]  # (B, H)

        base_latent_pos = pos_cursor + valid_len  # (B,)
        all_steps_attentions: List[List[torch.Tensor]] = []

        for step in range(int(latent_steps)):
            # Extend full attention mask by 1 real token
            full_mask = torch.cat([full_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=-1)

            latent_vec = self._apply_latent_realignment(last_hidden, self.model)
            latent_embed = latent_vec.unsqueeze(1)
            if hasattr(self.model, "dtype"):
                latent_embed = latent_embed.to(dtype=self.model.dtype)

            latent_pos_ids = (base_latent_pos + step).unsqueeze(1)  # (B,1)

            out = self.model(
                inputs_embeds=latent_embed,
                attention_mask=full_mask,  # IMPORTANT: full length
                position_ids=latent_pos_ids,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            past = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1, :]

            step_attn = [a[:, :, -1, :].detach() for a in out.attentions]
            all_steps_attentions.append(step_attn)

        new_pos_cursor = base_latent_pos + int(latent_steps)
        return past, all_steps_attentions, new_pos_cursor

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,                          # (B, L)
        attention_mask: Optional[torch.Tensor] = None,    # (B, L)
        *,
        pos_cursor: torch.Tensor,                         # (B,)
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[List[str], Any, torch.Tensor, List]:
        """
        Text decoding with a for-loop.
        Uses full 2D attention_mask with length (past_len + cur_len [+ steps]).
        Returns:
          generations, past_key_values, new_pos_cursor
        """

        device = input_ids.device
        B, L = input_ids.shape

        if pos_cursor is None:
            pos_cursor = torch.zeros((B,), device=device, dtype=torch.long)
        elif pos_cursor.dim() == 0:
            pos_cursor = pos_cursor.expand(B)
        elif pos_cursor.numel() == 1 and B > 1:
            pos_cursor = pos_cursor.expand(B)
        elif pos_cursor.size(0) != B:
            raise ValueError(f"pos_cursor size mismatch: {pos_cursor.size(0)} vs B={B}")

        if attention_mask is None:
            attention_mask = torch.ones((B, L), dtype=torch.long, device=device)
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.to(torch.long)
            attention_mask = (attention_mask > 0).to(torch.long).to(device)

        valid_len = attention_mask.sum(dim=-1).to(torch.long)
        last_valid_idx = (valid_len - 1).clamp(min=0)

        past_len = _past_length(past_key_values)
        if past_len > 0:
            past_mask = torch.ones((B, past_len), dtype=torch.long, device=device)
            full_mask = torch.cat([past_mask, attention_mask], dim=-1)      # (B, past_len + L)
        else:
            full_mask = attention_mask                                      # (B, L)

        token_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        prompt_pos_ids = pos_cursor.unsqueeze(1) + token_pos
        prompt_pos_ids = torch.where(attention_mask.bool(), prompt_pos_ids, torch.zeros_like(prompt_pos_ids))

        out = self.model(
            input_ids=input_ids,
            attention_mask=full_mask,                # IMPORTANT: full length
            position_ids=prompt_pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values

        logits = out.logits[torch.arange(B, device=device), last_valid_idx, :]  # (B,V)
        decode_base_pos = pos_cursor + valid_len                                 # (B,)

        eos_id = self.tokenizer.eos_token_id
        finished = torch.zeros((B,), dtype=torch.bool, device=device)
        generated: List[torch.Tensor] = []

        steps_done = 0

        for step in range(int(max_new_tokens)):
            steps_done = step + 1

            if temperature is None or temperature <= 0:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)  # (B,1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)

                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    keep = cum <= top_p
                    keep[:, 0] = True
                    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
                    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    sampled = torch.multinomial(filtered, 1)
                    next_tok = sorted_idx.gather(-1, sampled)
                else:
                    next_tok = torch.multinomial(probs, 1)

            if eos_id is not None:
                next_tok = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_tok, int(eos_id)),
                    next_tok,
                )

            generated.append(next_tok)

            if eos_id is not None:
                finished |= (next_tok.squeeze(1) == int(eos_id))
                if bool(torch.all(finished)):
                    break

            # Extend full attention mask by 1 real token
            full_mask = torch.cat([full_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=-1)

            step_pos_ids = (decode_base_pos + step).unsqueeze(1)

            out = self.model(
                input_ids=next_tok,
                attention_mask=full_mask,            # IMPORTANT: full length
                position_ids=step_pos_ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]

        gen_ids = torch.cat(generated, dim=1) if len(generated) > 0 else torch.empty((B, 0), dtype=torch.long,
                                                                                     device=device)
        generations = [self.tokenizer.decode(gen_ids[b], skip_special_tokens=True).strip() for b in range(B)]

        T = gen_ids.size(1)
        eos_id = self.tokenizer.eos_token_id

        if eos_id is not None and T > 0:
            eos_mask = (gen_ids == int(eos_id))  # (B, T)
            idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
            # Use T as "not found" sentinel, then take min to find first EOS position
            first_eos = torch.where(eos_mask, idx, torch.full_like(idx, T)).min(dim=1).values  # (B,)
            has_eos = eos_mask.any(dim=1)
            # Include EOS token itself: length = first_eos + 1, otherwise T
            gen_lens = torch.where(has_eos, first_eos + 1, torch.full((B,), T, device=device, dtype=torch.long))
        else:
            gen_lens = torch.zeros((B,), device=device, dtype=torch.long)

        new_pos_cursor = decode_base_pos + steps_done
        return generations, past, new_pos_cursor, gen_lens.detach().cpu().tolist()

