"""
Qwen3-VL-2B-Instruct model implementation for visual element verification.

This module implements the BaseVerifierModel interface for the
Qwen3-VL-2B-Instruct Vision-Language model from Hugging Face.

Model: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
"""

import math
from typing import Optional, Union, List
from pathlib import Path

import torch
from PIL import Image

from .base_model import BaseVerifierModel, LogitResult


class Qwen3VLModel(BaseVerifierModel):
    """
    Qwen3-VL-2B-Instruct implementation for visual element verification.
    
    This model extracts logits for "yes" and "no" tokens and computes
    normalized probabilities. Handles multi-token yes/no by computing
    the probability of generating the complete sequence.
    
    Uses VQAScore methodology: P("Yes" | image, question) as alignment score.
    """
    
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize Qwen3-VL model wrapper.
        
        Args:
            model_id: HuggingFace model ID. Defaults to Qwen/Qwen3-VL-2B-Instruct.
            device: Device to load model on ('cuda' or 'cpu').
            torch_dtype: Torch data type for model weights.
        """
        self._model_id = model_id or self.MODEL_ID
        self._device = device
        self._torch_dtype = torch_dtype
        self._model = None
        self._processor = None
        # Token sequences for "yes" and "no" (may be multi-token)
        self._yes_token_ids: Optional[List[int]] = None
        self._no_token_ids: Optional[List[int]] = None
        
    @property
    def model_name(self) -> str:
        return self._model_id
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def load(self) -> None:
        """Load model and processor, cache yes/no token IDs."""
        if self.is_loaded:
            return
            
        print(f"Loading model: {self._model_id}")
        
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
            AutoTokenizer,
            AutoImageProcessor,
        )
        
        # Manually load tokenizer, image_processor, and video_processor separately
        # to bypass video processor auto-detection issues in transformers 5.x
        # The AutoProcessor/Qwen3VLProcessor.from_pretrained() fails due to
        # a bug in video_processing_auto.py where extractors is None
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        
        print("  Loading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        
        print("  Loading video processor...")
        video_processor = Qwen3VLVideoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        
        # Manually assemble the processor
        print("  Assembling processor...")
        self._processor = Qwen3VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )
        
        # Copy chat_template from tokenizer to processor (not copied automatically)
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            self._processor.chat_template = tokenizer.chat_template
            print("  Chat template copied from tokenizer.")
        
        print("  Loading model weights...")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
            device_map=self._device,
            trust_remote_code=True,
        ).eval()
        
        # Cache token ID sequences for "yes" and "no" (may be multi-token)
        # We use lowercase as the prompt asks for "Yes" or "No" but models often output lowercase
        self._yes_token_ids = self._get_token_ids("Yes")
        self._no_token_ids = self._get_token_ids("No")
        
        # Also get lowercase variants
        self._yes_lower_ids = self._get_token_ids("yes")
        self._no_lower_ids = self._get_token_ids("no")
        
        # Log tokenization details for debugging
        print(f"Model loaded.")
        print(f"  'Yes' tokenizes to {len(self._yes_token_ids)} token(s): {self._yes_token_ids}")
        print(f"  'No' tokenizes to {len(self._no_token_ids)} token(s): {self._no_token_ids}")
        print(f"  'yes' tokenizes to {len(self._yes_lower_ids)} token(s): {self._yes_lower_ids}")
        print(f"  'no' tokenizes to {len(self._no_lower_ids)} token(s): {self._no_lower_ids}")
    
    def _get_token_ids(self, word: str) -> List[int]:
        """
        Get complete token ID sequence for a word.
        
        Args:
            word: The word to tokenize (e.g., "Yes", "No").
            
        Returns:
            List of token IDs representing the complete word.
        """
        ids = self._processor.tokenizer.encode(word, add_special_tokens=False)
        return ids
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
        print("Model unloaded.")
    
    def _load_image(self, image: Union[Image.Image, Path, str]) -> Image.Image:
        """
        Load image from various sources.
        
        Args:
            image: PIL Image, path, or URL
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, (Path, str)):
            path = Path(image)
            if path.exists():
                return Image.open(path).convert("RGB")
            else:
                # Assume it's a URL
                import requests
                from io import BytesIO
                response = requests.get(str(image))
                return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def format_figurative_prompt(
        self,
        content: str,
        rationale: str,
        system_prompt: str,
    ) -> str:
        """
        Format prompt for figurative element verification.
        
        The prompt follows the phase1_figurative_verifier_specialist template.
        """
        # Replace placeholders in system prompt
        formatted_prompt = system_prompt.replace("<content>", content)
        formatted_prompt = formatted_prompt.replace("<rationale>", rationale)
        
        return formatted_prompt
    
    def format_literal_prompt(
        self,
        content: str,
        system_prompt: str,
    ) -> str:
        """
        Format prompt for literal element verification.
        
        The prompt follows the phase1_literal_verifier_specialist template.
        """
        # Replace placeholder in system prompt
        formatted_prompt = system_prompt.replace("<content>", content)
        
        return formatted_prompt
    
    @torch.no_grad()
    def get_yes_no_probs(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> LogitResult:
        """
        Compute probabilities for generating "yes" or "no" given image and prompt.
        
        Uses the VQAScore methodology: extract P("Yes"|image, question) as the
        alignment/verification score.
        
        This method handles multi-token sequences correctly by computing:
            P(sequence) = ∏ P(token_i | prompt + tokens_0..i-1)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load image
        pil_image = self._load_image(image)
        
        # Prepare messages for Qwen3-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate with scores - we only need the first token
        # Using max_new_tokens=1 since we want single Yes/No token probability
        output = self._model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,  # Greedy for reproducibility
        )
        
        # Get logits for the first generated token
        # output.scores is a tuple of tensors, one per generated token
        # Each tensor has shape (batch_size, vocab_size)
        first_token_logits = output.scores[0][0]  # Shape: (vocab_size,)
        
        # Get logits for Yes/No tokens (use first token of each sequence)
        # We check both uppercase and lowercase variants
        yes_logit = max(
            first_token_logits[self._yes_token_ids[0]].item(),
            first_token_logits[self._yes_lower_ids[0]].item()
        )
        no_logit = max(
            first_token_logits[self._no_token_ids[0]].item(),
            first_token_logits[self._no_lower_ids[0]].item()
        )
        
        # Normalize using softmax over just yes/no
        # P(yes) = exp(yes_logit) / (exp(yes_logit) + exp(no_logit))
        max_logit = max(yes_logit, no_logit)
        yes_exp = math.exp(yes_logit - max_logit)
        no_exp = math.exp(no_logit - max_logit)
        total = yes_exp + no_exp
        
        yes_prob = yes_exp / total
        no_prob = no_exp / total
        
        return LogitResult(
            yes_logit=yes_logit,
            no_logit=no_logit,
            yes_prob=yes_prob,
            no_prob=no_prob,
        )
    
    @torch.no_grad()
    def get_yes_no_probs_full_sequence(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> LogitResult:
        """
        Alternative method: compute full sequence probabilities for multi-token yes/no.
        
        This is more accurate but slower. Use get_yes_no_probs() for single-token
        approximation which is usually sufficient.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load image
        pil_image = self._load_image(image)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Compute log probability for "Yes" sequence
        yes_log_prob, yes_logits = self._compute_sequence_log_prob(inputs, self._yes_token_ids)
        
        # Compute log probability for "No" sequence
        no_log_prob, no_logits = self._compute_sequence_log_prob(inputs, self._no_token_ids)
        
        # Normalize using softmax
        max_log_prob = max(yes_log_prob, no_log_prob)
        yes_exp = math.exp(yes_log_prob - max_log_prob)
        no_exp = math.exp(no_log_prob - max_log_prob)
        total = yes_exp + no_exp
        
        yes_prob = yes_exp / total
        no_prob = no_exp / total
        
        return LogitResult(
            yes_logit=yes_logits[0] if yes_logits else 0.0,
            no_logit=no_logits[0] if no_logits else 0.0,
            yes_prob=yes_prob,
            no_prob=no_prob,
        )
    
    @torch.no_grad()
    def _compute_sequence_log_prob(
        self,
        inputs: dict,
        target_token_ids: List[int],
    ) -> tuple:
        """
        Compute log probability of generating a specific token sequence.
        
        Args:
            inputs: Tokenized inputs dictionary
            target_token_ids: List of token IDs to compute probability for
            
        Returns:
            Tuple of (total_log_prob, list_of_logits_for_each_token)
        """
        import torch.nn.functional as F
        
        total_log_prob = 0.0
        logits_list = []
        
        # Clone inputs for iteration
        current_input_ids = inputs["input_ids"].clone()
        current_attention_mask = inputs.get("attention_mask")
        if current_attention_mask is not None:
            current_attention_mask = current_attention_mask.clone()
        
        # Handle pixel_values for vision input
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        
        for i, target_tid in enumerate(target_token_ids):
            # Forward pass
            model_inputs = {"input_ids": current_input_ids}
            if current_attention_mask is not None:
                model_inputs["attention_mask"] = current_attention_mask
            if pixel_values is not None and i == 0:  # Only pass pixel_values on first iteration
                model_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None and i == 0:
                model_inputs["image_grid_thw"] = image_grid_thw
            
            outputs = self._model(**model_inputs)
            
            # Get logits at the last position
            last_logits = outputs.logits[0, -1, :]
            
            # Get the logit for the target token
            target_logit = last_logits[target_tid].item()
            logits_list.append(target_logit)
            
            # Convert to log probability
            log_probs = F.log_softmax(last_logits, dim=0)
            token_log_prob = log_probs[target_tid].item()
            total_log_prob += token_log_prob
            
            # Append target token for next iteration
            next_token = torch.tensor([[target_tid]], device=self._model.device)
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            
            if current_attention_mask is not None:
                next_mask = torch.ones((1, 1), device=self._model.device, dtype=current_attention_mask.dtype)
                current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)
        
        return total_log_prob, logits_list
