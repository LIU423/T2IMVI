"""
Qwen3-0.6B model implementation for imageability evaluation.

This module implements the BaseImageabilityModel interface for the
Qwen3-0.6B model from Hugging Face.
"""

import math
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_model import BaseImageabilityModel, LogitResult


class Qwen3Model(BaseImageabilityModel):
    """
    Qwen3-0.6B implementation for imageability evaluation.
    
    This model extracts logits for "yes" and "no" tokens and computes
    normalized probabilities. Handles multi-token yes/no by computing
    the probability of generating the complete sequence.
    """
    
    MODEL_ID = "Qwen/Qwen3-0.6B"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize Qwen3 model wrapper.
        
        Args:
            model_id: HuggingFace model ID. Defaults to Qwen/Qwen3-0.6B.
            device: Device to load model on ('cuda' or 'cpu').
            torch_dtype: Torch data type for model weights.
        """
        self._model_id = model_id or self.MODEL_ID
        self._device = device
        self._torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None
        # Token sequences for "yes" and "no" (may be multi-token)
        self._yes_token_ids: Optional[list[int]] = None
        self._no_token_ids: Optional[list[int]] = None
        
    @property
    def model_name(self) -> str:
        return self._model_id
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def load(self) -> None:
        """Load model and tokenizer, cache yes/no token IDs."""
        if self.is_loaded:
            return
            
        print(f"Loading model: {self._model_id}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
            device_map=self._device,
            trust_remote_code=True,
        ).eval()
        
        # Cache token ID sequences for "yes" and "no" (may be multi-token)
        self._yes_token_ids = self._get_token_ids("yes")
        self._no_token_ids = self._get_token_ids("no")
        
        # Log tokenization details for debugging
        yes_tokens = [self._tokenizer.decode([tid]) for tid in self._yes_token_ids]
        no_tokens = [self._tokenizer.decode([tid]) for tid in self._no_token_ids]
        print(f"Model loaded.")
        print(f"  'yes' tokenizes to {len(self._yes_token_ids)} token(s): {self._yes_token_ids} -> {yes_tokens}")
        print(f"  'no' tokenizes to {len(self._no_token_ids)} token(s): {self._no_token_ids} -> {no_tokens}")
    
    def _get_token_ids(self, word: str) -> list[int]:
        """
        Get complete token ID sequence for a word.
        
        This method returns ALL token IDs needed to represent the word,
        handling cases where the word is split into multiple tokens.
        
        Args:
            word: The word to tokenize (e.g., "yes", "no").
            
        Returns:
            List of token IDs representing the complete word.
        """
        # Encode without special tokens to get raw token sequence
        ids = self._tokenizer.encode(word, add_special_tokens=False)
        return ids
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()
        print("Model unloaded.")
    
    def format_prompt(self, idiom: str, system_prompt: str) -> str:
        """
        Format prompt for Qwen3 model using chat template.
        
        The prompt asks the model to classify imageability and expects
        a yes/no response.
        
        Note: For imageability, we only need the idiom itself (no definition),
        as we judge based on the literal wording's ability to evoke mental images.
        """
        user_content = f"Idiom: {idiom}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        # Use tokenizer's chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return prompt
    
    @torch.no_grad()
    def get_yes_no_logits(self, prompt: str) -> LogitResult:
        """
        Compute probabilities for generating complete "yes" or "no" sequences.
        
        This method handles multi-token sequences correctly by computing:
            P(sequence) = ∏ P(token_i | prompt + tokens_0..i-1)
        
        The sequence probability is computed autoregressively:
        1. For each token in the target sequence
        2. Get the logit at the current position
        3. Convert to probability using softmax over full vocabulary
        4. Accumulate log probabilities
        
        Finally, the yes/no probabilities are normalized to sum to 1.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Tokenize prompt
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)
        
        # Compute log probability for "yes" sequence
        yes_log_prob, yes_logits = self._compute_sequence_log_prob(
            input_ids, attention_mask, self._yes_token_ids
        )
        
        # Compute log probability for "no" sequence
        no_log_prob, no_logits = self._compute_sequence_log_prob(
            input_ids, attention_mask, self._no_token_ids
        )
        
        # Normalize using softmax: P(yes) = exp(log_p_yes) / (exp(log_p_yes) + exp(log_p_no))
        # For numerical stability, subtract max before exp
        max_log_prob = max(yes_log_prob, no_log_prob)
        yes_exp = math.exp(yes_log_prob - max_log_prob)
        no_exp = math.exp(no_log_prob - max_log_prob)
        total = yes_exp + no_exp
        
        yes_prob = yes_exp / total
        no_prob = no_exp / total
        
        # Return first token logits for backward compatibility
        # (The actual probability uses the full sequence)
        return LogitResult(
            yes_logit=yes_logits[0] if yes_logits else 0.0,
            no_logit=no_logits[0] if no_logits else 0.0,
            yes_prob=yes_prob,
            no_prob=no_prob,
        )
    
    @torch.no_grad()
    def _compute_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        target_token_ids: list[int],
    ) -> tuple[float, list[float]]:
        """
        Compute log probability of generating a specific token sequence.
        
        For a sequence [t0, t1, t2, ...], compute:
            log P(sequence) = log P(t0|prompt) + log P(t1|prompt,t0) + ...
        
        Args:
            input_ids: Tokenized prompt (batch_size=1, seq_len).
            attention_mask: Attention mask for the prompt.
            target_token_ids: List of token IDs to compute probability for.
            
        Returns:
            Tuple of (total_log_prob, list_of_logits_for_each_token).
        """
        total_log_prob = 0.0
        logits_list = []
        
        # Start with the original prompt
        current_input_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        
        for i, target_tid in enumerate(target_token_ids):
            # Forward pass to get next token logits
            outputs = self._model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
            )
            
            # Get logits at the last position (next token prediction)
            last_logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)
            
            # Get the logit for the target token
            target_logit = last_logits[target_tid].item()
            logits_list.append(target_logit)
            
            # Convert to log probability using log_softmax over full vocabulary
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=0)
            token_log_prob = log_probs[target_tid].item()
            total_log_prob += token_log_prob
            
            # Append the target token for next iteration
            next_token = torch.tensor([[target_tid]], device=self._model.device)
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            
            if current_attention_mask is not None:
                next_mask = torch.ones((1, 1), device=self._model.device, dtype=current_attention_mask.dtype)
                current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)
        
        return total_log_prob, logits_list


class Qwen3_30B_A3B_Instruct_2507(Qwen3Model):
    """
    Qwen3-30B-A3B-Instruct-2507 model implementation.

    Uses the same logic as Qwen3Model, with a different MODEL_ID.
    """

    MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
