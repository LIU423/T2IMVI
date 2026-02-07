"""
Gemini Model Implementation

Supports Google Gemini API for text generation.
Compatible with the BaseExtractionModel interface.
"""

import json
import re
import logging
from typing import Type, Optional

from pydantic import BaseModel, ValidationError

from models.base_model import BaseExtractionModel, ModelConfig

logger = logging.getLogger(__name__)


class GeminiModel(BaseExtractionModel):
    """
    Gemini API model implementation.
    
    Features:
    - Google Gemini API integration
    - Structured JSON output with Pydantic validation
    - Retry logic for parsing failures
    - Compatible with BaseExtractionModel interface
    """
    
    DEFAULT_MODEL = "gemini-2.0-flash"
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name=self.DEFAULT_MODEL,
            )
        super().__init__(config)
        self.client = None
        
    def load_model(self) -> None:
        """Initialize Gemini API client."""
        if self._is_loaded:
            logger.info("Gemini client already initialized, skipping...")
            return
            
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for Gemini API. "
                "Install with: pip install google-genai"
            ) from e
        
        # Get API key from config
        api_key = self.config.extra_kwargs.get("api_key")
        if not api_key:
            raise ValueError(
                "API key is required for Gemini API. "
                "Provide it via ModelConfig.extra_kwargs['api_key']"
            )
        
        logger.info(f"Initializing Gemini client with model: {self.config.model_name}")
        
        # Initialize client with API key
        self.client = genai.Client(api_key=api_key)
        
        self._is_loaded = True
        logger.info(f"Gemini client initialized successfully: {self.config.model_name}")
    
    def generate(self, prompt: str) -> str:
        """Generate raw text response using Gemini API."""
        if not self._is_loaded:
            self.load_model()
        
        model_name = self.config.model_path or self.config.model_name or self.DEFAULT_MODEL
        
        # Build generation config
        generation_config = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_new_tokens,
        }
        
        logger.debug(f"Generating with model: {model_name}")
        logger.debug(f"Generation config: {generation_config}")
        
        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config,
            )
            
            # Extract text from response
            if response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini API")
                return ""
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
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
        current_prompt = prompt
        
        for attempt in range(self.config.max_retries):
            try:
                # Generate response
                response = self.generate(current_prompt)
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
                    current_prompt = self._add_repair_hint(prompt, str(e))
        
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
    
    def unload(self) -> None:
        """Clean up Gemini client."""
        if self.client is not None:
            self.client = None
        self._is_loaded = False
        logger.info("Gemini client unloaded")
