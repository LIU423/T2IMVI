"""
Abstract base class for AEA (Abstract Element Alignment) models.

This module defines the interface that all VLM implementations must follow
for the Phase 2 AEA calculation pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List

from PIL import Image


@dataclass
class LogitResult:
    """
    Result from logit extraction for multi-level scoring.
    
    For AEA, we use 3 levels: "one" (clash), "two" (neutral), "three" (match).
    The AEA score is computed as: 0.5 * P("two") + P("three")
    
    Attributes:
        one_logit: Raw logit for "one" token
        two_logit: Raw logit for "two" token
        three_logit: Raw logit for "three" token
        one_prob: Normalized probability for "one"
        two_prob: Normalized probability for "two"
        three_prob: Normalized probability for "three"
    """
    one_logit: float
    two_logit: float
    three_logit: float
    one_prob: float
    two_prob: float
    three_prob: float
    
    @property
    def aea_score(self) -> float:
        """
        Compute AEA score as 0.5 * P("two") + P("three").
        """
        return 0.5 * self.two_prob + self.three_prob


class BaseAEAModel(ABC):
    """
    Abstract base class for Vision-Language Models used in AEA calculation.
    
    All VLM implementations must inherit from this class and implement
    the abstract methods.
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/ID."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is loaded into memory."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    def format_aea_prompt(
        self,
        abstract_atmosphere: str,
        system_prompt: str,
    ) -> str:
        """
        Format prompt for AEA evaluation.
        
        Args:
            abstract_atmosphere: The abstract_atmosphere text from figurative.json
            system_prompt: The system prompt template from phase2_aea.txt
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def get_level_probs(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> LogitResult:
        """
        Get probabilities for "one", "two", "three" tokens.
        
        Args:
            image: Image to evaluate (PIL Image, path, or URL)
            prompt: Formatted AEA prompt
            
        Returns:
            LogitResult with probabilities for each level
        """
        pass
    
    def compute_aea_score(
        self,
        image: Union[Image.Image, Path, str],
        abstract_atmosphere: str,
        system_prompt: str,
    ) -> float:
        """
        Compute AEA score for an image-text pair.
        
        This is a convenience method that combines prompt formatting
        and probability extraction.
        
        Args:
            image: Image to evaluate
            abstract_atmosphere: Abstract atmosphere description
            system_prompt: System prompt template
            
        Returns:
            AEA score (0.5 * P("two") + P("three"))
        """
        prompt = self.format_aea_prompt(abstract_atmosphere, system_prompt)
        result = self.get_level_probs(image, prompt)
        return result.aea_score
