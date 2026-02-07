"""
Qwen3-VL-2B-Instruct model implementation for IU calculation.

This module implements the BaseIUModel interface for the
Qwen3-VL-2B-Instruct Vision-Language model from Hugging Face.

Model: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

Uses VQAScore methodology: IU Score = P("yes")
"""

import math
from typing import Optional, Union, List
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .iu_base_model import BaseIUModel, YesNoLogitResult


class Qwen3VLIUModel(BaseIUModel):
    """
    Qwen3-VL-2B-Instruct implementation for IU calculation.
    
    This model extracts logits for "yes" and "no" tokens and computes
    normalized probabilities. The IU score is P("yes").
    
    Uses VQAScore methodology for binary yes/no scoring.
    """
    
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize Qwen3-VL model wrapper for IU.
        
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
        
        # Token IDs for yes/no outputs (cached after load)
        self._yes_token_id: Optional[int] = None
        self._no_token_id: Optional[int] = None
        
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
        
        # Cache token IDs for yes/no outputs
        # Try both with and without leading space, use the single-token version
        self._yes_token_id = self._get_best_token_id("yes")
        self._no_token_id = self._get_best_token_id("no")
        
        # Log tokenization details for debugging
        print(f"Model loaded.")
        print(f"  'yes' token ID: {self._yes_token_id}")
        print(f"  'no' token ID: {self._no_token_id}")
    
    def _get_best_token_id(self, word: str) -> int:
        """
        Get the best single token ID for a word.
        
        Tries both with and without leading space, prefers single-token encoding.
        
        Args:
            word: The word to tokenize (e.g., "yes", "no").
            
        Returns:
            Single token ID for the word.
        """
        # Try without space first
        ids_no_space = self._processor.tokenizer.encode(word, add_special_tokens=False)
        # Try with leading space (common in LLM tokenizers)
        ids_with_space = self._processor.tokenizer.encode(f" {word}", add_special_tokens=False)
        
        # Prefer single-token encoding
        if len(ids_no_space) == 1:
            return ids_no_space[0]
        elif len(ids_with_space) == 1:
            return ids_with_space[0]
        else:
            # Fall back to first token of the no-space version
            return ids_no_space[0]
    
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
    
    def format_relationships_prompt(
        self,
        core_abstract_concept: str,
        subject: str,
        action: str,
        obj: str,
        system_prompt: str,
    ) -> str:
        """
        Format prompt for relationship-based IU evaluation.
        
        Replaces template variables in the system prompt:
        - {{core_abstract_concept}} -> core_abstract_concept
        - {{subject}} -> subject
        - {{action}} -> action
        - {{object}} -> obj
        
        Args:
            core_abstract_concept: The core abstract concept
            subject: The subject entity content
            action: The action content
            obj: The object entity content
            system_prompt: The system prompt template
            
        Returns:
            Formatted prompt string
        """
        prompt = system_prompt.replace("{{core_abstract_concept}}", core_abstract_concept)
        prompt = prompt.replace("{{subject}}", subject)
        prompt = prompt.replace("{{action}}", action)
        prompt = prompt.replace("{{object}}", obj)
        return prompt
    
    def format_without_relationships_prompt(
        self,
        core_abstract_concept: str,
        entity: str,
        action: str,
        system_prompt: str,
    ) -> str:
        """
        Format prompt for entity-action based IU evaluation.
        
        Replaces template variables in the system prompt:
        - {{core_abstract_concept}} -> core_abstract_concept
        - {{entity}} -> entity
        - {{action}} -> action
        
        Args:
            core_abstract_concept: The core abstract concept
            entity: The highest-scoring entity content
            action: The highest-scoring action content
            system_prompt: The system prompt template
            
        Returns:
            Formatted prompt string
        """
        prompt = system_prompt.replace("{{core_abstract_concept}}", core_abstract_concept)
        prompt = prompt.replace("{{entity}}", entity)
        prompt = prompt.replace("{{action}}", action)
        return prompt
    
    @torch.no_grad()
    def get_yes_no_probs(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> YesNoLogitResult:
        """
        Compute probabilities for "yes" and "no" tokens.
        
        Uses VQAScore methodology: extract token probabilities from first
        generated token position.
        
        Args:
            image: Image to evaluate
            prompt: Formatted IU prompt
            
        Returns:
            YesNoLogitResult with probabilities for yes/no
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
        output = self._model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,  # Greedy for reproducibility
        )
        
        # Get logits for the first generated token
        first_token_logits = output.scores[0][0]  # Shape: (vocab_size,)
        
        # Get logits for yes/no tokens
        yes_logit = first_token_logits[self._yes_token_id].item()
        no_logit = first_token_logits[self._no_token_id].item()
        
        # Normalize using softmax over just yes/no tokens
        # P(yes) = exp(yes_logit) / (exp(yes_logit) + exp(no_logit))
        max_logit = max(yes_logit, no_logit)
        yes_exp = math.exp(yes_logit - max_logit)
        no_exp = math.exp(no_logit - max_logit)
        total = yes_exp + no_exp
        
        yes_prob = yes_exp / total
        no_prob = no_exp / total
        
        return YesNoLogitResult(
            yes_logit=yes_logit,
            no_logit=no_logit,
            yes_prob=yes_prob,
            no_prob=no_prob,
        )
