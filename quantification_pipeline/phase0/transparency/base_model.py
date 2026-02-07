"""
Base model interface for transparency evaluation.

This module provides an abstract base class that all model implementations
should inherit from. This design allows easy swapping of different LLMs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
from dataclasses import dataclass


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


class BaseTransparencyModel(ABC):
    """
    Abstract base class for transparency evaluation models.
    
    To add a new model:
    1. Create a new file (e.g., new_model.py)
    2. Inherit from BaseTransparencyModel
    3. Implement all abstract methods
    4. Register in MODEL_REGISTRY in config.py
    """
    
    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        pass
    
    @abstractmethod
    def get_yes_no_logits(self, prompt: str) -> LogitResult:
        """
        Given a prompt, compute probabilities for generating "yes" and "no".
        
        Implementations must handle multi-token sequences correctly by
        computing the probability of the complete sequence:
            P(sequence) = ∏ P(token_i | prompt + tokens_0..i-1)
        
        Args:
            prompt: The complete prompt including idiom and meaning.
            
        Returns:
            LogitResult containing logits and normalized probabilities.
        """
        pass
    
    @abstractmethod
    def format_prompt(self, idiom: str, definition: str, system_prompt: str) -> str:
        """
        Format the idiom and definition into the model's expected prompt format.
        
        Args:
            idiom: The English idiom.
            definition: The figurative meaning of the idiom.
            system_prompt: The system prompt template.
            
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
