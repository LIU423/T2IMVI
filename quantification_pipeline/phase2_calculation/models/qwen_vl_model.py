"""
Qwen3-VL-2B-Instruct model implementation for AEA calculation.

This module implements the BaseAEAModel interface for the
Qwen3-VL-2B-Instruct Vision-Language model from Hugging Face.

Model: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

Adapted for 3-level output: "one" (clash), "two" (neutral), "three" (match)
AEA Score = 1 - P("one")
"""

import math
from typing import Optional, Union, List
from pathlib import Path

import torch
from PIL import Image

from .base_model import BaseAEAModel, LogitResult


class Qwen3VLModel(BaseAEAModel):
    """
    Qwen3-VL-2B-Instruct implementation for AEA calculation.
    
    This model extracts logits for "one", "two", "three" tokens and computes
    normalized probabilities. The AEA score is 1 - P("one").
    
    Uses VQAScore methodology adapted for 3-level output.
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
        
        # Token IDs for level outputs (cached after load)
        self._one_token_ids: Optional[List[int]] = None
        self._two_token_ids: Optional[List[int]] = None
        self._three_token_ids: Optional[List[int]] = None
        
    @property
    def model_name(self) -> str:
        return self._model_id
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def load(self) -> None:
        """Load model and processor, cache level token IDs."""
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
        
        # Cache token IDs for level outputs
        self._one_token_ids = self._get_token_ids("one")
        self._two_token_ids = self._get_token_ids("two")
        self._three_token_ids = self._get_token_ids("three")
        
        # Log tokenization details for debugging
        print(f"Model loaded.")
        print(f"  'one' tokenizes to {len(self._one_token_ids)} token(s): {self._one_token_ids}")
        print(f"  'two' tokenizes to {len(self._two_token_ids)} token(s): {self._two_token_ids}")
        print(f"  'three' tokenizes to {len(self._three_token_ids)} token(s): {self._three_token_ids}")
    
    def _get_token_ids(self, word: str) -> List[int]:
        """
        Get complete token ID sequence for a word.
        
        Args:
            word: The word to tokenize (e.g., "one", "two", "three").
            
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
    
    def format_aea_prompt(
        self,
        abstract_atmosphere: str,
        system_prompt: str,
    ) -> str:
        """
        Format prompt for AEA evaluation.
        
        The system_prompt contains the full evaluation instructions.
        We need to append the abstract_atmosphere as the input.
        
        Args:
            abstract_atmosphere: The abstract_atmosphere text from figurative.json
            system_prompt: The system prompt template from phase2_aea.txt
            
        Returns:
            Formatted prompt string
        """
        # The prompt template expects the abstract_atmosphere to be provided
        # as JSON context that the model evaluates against the image
        formatted_prompt = f"""{system_prompt}

# Your Input
JSON `abstract_atmosphere`: "{abstract_atmosphere}"
Image: [Attached]

Your output:"""
        
        return formatted_prompt
    
    @torch.no_grad()
    def get_level_probs(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> LogitResult:
        """
        Compute probabilities for "one", "two", "three" tokens.
        
        Uses VQAScore methodology: extract token probabilities from first
        generated token position.
        
        Args:
            image: Image to evaluate
            prompt: Formatted AEA prompt
            
        Returns:
            LogitResult with probabilities for each level
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
        
        # Get logits for level tokens (use first token of each sequence)
        one_logit = first_token_logits[self._one_token_ids[0]].item()
        two_logit = first_token_logits[self._two_token_ids[0]].item()
        three_logit = first_token_logits[self._three_token_ids[0]].item()
        
        # Normalize using softmax over just the three level tokens
        # P(level) = exp(logit) / sum(exp(all_level_logits))
        max_logit = max(one_logit, two_logit, three_logit)
        one_exp = math.exp(one_logit - max_logit)
        two_exp = math.exp(two_logit - max_logit)
        three_exp = math.exp(three_logit - max_logit)
        total = one_exp + two_exp + three_exp
        
        one_prob = one_exp / total
        two_prob = two_exp / total
        three_prob = three_exp / total
        
        return LogitResult(
            one_logit=one_logit,
            two_logit=two_logit,
            three_logit=three_logit,
            one_prob=one_prob,
            two_prob=two_prob,
            three_prob=three_prob,
        )
