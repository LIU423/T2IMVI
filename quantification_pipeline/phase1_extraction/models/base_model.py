"""
Base Model Abstraction Layer

Provides abstract interface for LLM models, enabling easy model swapping.
All model implementations should inherit from BaseExtractionModel.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, Optional, Any
from pydantic import BaseModel


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    
    model_name: str
    model_path: Optional[str] = None  # Local path or HuggingFace repo ID
    device: str = "auto"  # "auto", "cuda", "cpu", "cuda:0", etc.
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    max_new_tokens: int = 8192  # Increased from 2048 to prevent JSON truncation
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    trust_remote_code: bool = True
    
    # Generation constraints
    max_retries: int = 3  # Retry on parse failure
    
    # Additional model-specific kwargs
    extra_kwargs: dict = field(default_factory=dict)


class BaseExtractionModel(ABC):
    """
    Abstract base class for extraction models.
    
    Subclasses must implement:
    - load_model(): Initialize the model
    - generate(): Generate text response
    - generate_structured(): Generate and validate structured output
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer into memory."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate raw text response from prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Generate response and validate against Pydantic schema.
        
        Args:
            prompt: Input prompt string
            response_schema: Pydantic model class for validation
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            ValidationError: If response cannot be parsed to schema after retries
        """
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._is_loaded = False
        
        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
