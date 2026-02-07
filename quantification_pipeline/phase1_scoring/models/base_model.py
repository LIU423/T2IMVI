"""
Base model interface for visual element verification.

This module provides an abstract base class that all Vision-Language Model
implementations should inherit from. This design allows easy swapping of
different VLMs for element verification tasks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
from dataclasses import dataclass
from pathlib import Path
from PIL import Image


@dataclass
class LogitResult:
    """
    Container for yes/no probability extraction results.
    
    For multi-token sequences (e.g., if "yes" tokenizes to ["y", "es"]),
    the probabilities are computed over the complete sequence:
        P(yes) = P(t0|prompt) * P(t1|prompt,t0) * ...
    
    The logit fields contain the first token's logit for debugging purposes,
    while prob fields contain the normalized sequence probabilities.
    """
    yes_logit: float  # First token logit (for debugging)
    no_logit: float   # First token logit (for debugging)
    yes_prob: float   # Normalized probability for complete "yes" sequence
    no_prob: float    # Normalized probability for complete "no" sequence


class BaseVerifierModel(ABC):
    """
    Abstract base class for Vision-Language Models used in element verification.
    
    To add a new model:
    1. Create a new file (e.g., new_vlm_model.py)
    2. Inherit from BaseVerifierModel
    3. Implement all abstract methods
    4. Register in MODEL_REGISTRY in config.py
    
    Key methods:
    - load(): Load model and processor into memory
    - unload(): Free GPU memory
    - get_yes_no_probs(): Extract yes/no probabilities given image and prompt
    - format_figurative_prompt(): Format prompt for figurative verification
    - format_literal_prompt(): Format prompt for literal verification
    """
    
    @abstractmethod
    def load(self) -> None:
        """Load the model and processor into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        pass
    
    @abstractmethod
    def get_yes_no_probs(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> LogitResult:
        """
        Given an image and prompt, compute probabilities for generating "yes" and "no".
        
        Implementations must handle multi-token sequences correctly by
        computing the probability of the complete sequence:
            P(sequence) = ∏ P(token_i | prompt + tokens_0..i-1)
        
        Args:
            image: PIL Image, path to image file, or URL
            prompt: The complete prompt including verification question
            
        Returns:
            LogitResult containing logits and normalized probabilities.
        """
        pass
    
    @abstractmethod
    def format_figurative_prompt(
        self,
        content: str,
        rationale: str,
        system_prompt: str,
    ) -> str:
        """
        Format the prompt for figurative element verification.
        
        Args:
            content: The target concept/element to verify
            rationale: The metaphorical function/context
            system_prompt: The system prompt template from file
            
        Returns:
            Formatted prompt string ready for the model.
        """
        pass
    
    @abstractmethod
    def format_literal_prompt(
        self,
        content: str,
        system_prompt: str,
    ) -> str:
        """
        Format the prompt for literal element verification.
        
        Args:
            content: The target object to verify
            system_prompt: The system prompt template from file
            
        Returns:
            Formatted prompt string ready for the model.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier/name."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is currently loaded in memory."""
        pass
