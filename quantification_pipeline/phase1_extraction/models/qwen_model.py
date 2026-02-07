"""
Qwen Model Implementation

Supports Qwen series models from HuggingFace.
Default: Qwen/Qwen3-0.6B
"""

import json
import re
import logging
from typing import Type, Optional

from pydantic import BaseModel, ValidationError

from models.base_model import BaseExtractionModel, ModelConfig

logger = logging.getLogger(__name__)


class QwenModel(BaseExtractionModel):
    """
    Qwen model implementation using transformers library.
    
    Features:
    - Automatic device placement
    - Structured JSON output with Pydantic validation
    - Retry logic for parsing failures
    """
    
    DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name="Qwen3-0.6B",
                model_path=self.DEFAULT_MODEL,
            )
        super().__init__(config)
        
    def load_model(self) -> None:
        """Load Qwen model and tokenizer."""
        if self._is_loaded:
            logger.info("Model already loaded, skipping...")
            return
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            ) from e
        
        model_path = self.config.model_path or self.DEFAULT_MODEL
        logger.info(f"Loading model from: {model_path}")
        
        # Determine device and dtype
        if self.config.device == "auto":
            device_map = "auto"
        else:
            device_map = self.config.device
            
        if self.config.torch_dtype == "auto":
            torch_dtype = "auto"
        elif self.config.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=self.config.trust_remote_code,
            **self.config.extra_kwargs,
        )
        
        self._is_loaded = True
        logger.info(f"Model loaded successfully: {self.config.model_name}")
    
    def generate(self, prompt: str) -> str:
        """Generate raw text response."""
        if not self._is_loaded:
            self.load_model()
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking for Qwen3
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode - remove input tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Generate and validate structured JSON output.
        
        Uses retry logic to handle parsing failures.
        """
        if not self._is_loaded:
            self.load_model()
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Generate response
                response = self.generate(prompt)
                logger.debug(f"Raw response (attempt {attempt + 1}):\n{response}")
                
                # Extract JSON from response
                json_str = self._extract_json(response)
                
                # Parse and validate
                data = json.loads(json_str)
                result = response_schema.model_validate(data)
                
                logger.info(f"Successfully parsed response on attempt {attempt + 1}")
                return result
                
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(f"Parse attempt {attempt + 1} failed: {e}")
                
                # Add repair hint to prompt for next attempt
                if attempt < self.config.max_retries - 1:
                    prompt = self._add_repair_hint(prompt, str(e))
        
        raise ValueError(
            f"Failed to generate valid structured output after {self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text response."""
        # Try to find JSON block in markdown code fence
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json_match.group(1).strip()
        
        # Try to find raw JSON object
        # Look for outermost { ... } pair
        stack = []
        start_idx = None
        
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        return text[start_idx:i + 1]
        
        # If no valid JSON found, return the whole text and let parser handle error
        return text.strip()
    
    def _add_repair_hint(self, prompt: str, error_msg: str) -> str:
        """Add hint about previous parsing failure to prompt."""
        repair_hint = (
            f"\n\n[IMPORTANT: Your previous response had a JSON parsing error: {error_msg}. "
            "Please ensure you return a valid JSON object that strictly follows the schema.]"
        )
        return prompt + repair_hint
