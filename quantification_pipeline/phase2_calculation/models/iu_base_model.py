"""
Abstract base class for IU (Image Understanding) models.

This module defines the interface that all VLM implementations must follow
for the Phase 2 IU calculation pipeline.

IU uses 3-level classification:
- "one": mismatch
- "two": partial/neutral match
- "three": strong match

IU score = 0.5 * P("two") + P("three")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List

from PIL import Image


@dataclass
class IULevelLogitResult:
    """
    Result from 3-level logit extraction for IU scoring.
    
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
    def iu_score(self) -> float:
        """
        Compute IU score as 0.5 * P("two") + P("three").
        """
        return 0.5 * self.two_prob + self.three_prob


class BaseIUModel(ABC):
    """
    Abstract base class for Vision-Language Models used in IU calculation.
    
    All VLM implementations must inherit from this class and implement
    the abstract methods. Uses VQAScore methodology for scoring.
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
        
        Args:
            core_abstract_concept: The core abstract concept from figurative.json
            subject: The subject entity content
            action: The action content
            obj: The object entity content
            system_prompt: The system prompt template from phase2_iu_relationships.txt
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def format_without_relationships_prompt(
        self,
        core_abstract_concept: str,
        entity: str,
        action: str,
        system_prompt: str,
    ) -> str:
        """
        Format prompt for entity-action based IU evaluation (no valid relationships).
        
        Args:
            core_abstract_concept: The core abstract concept from figurative.json
            entity: The highest-scoring entity content
            action: The highest-scoring action content
            system_prompt: The system prompt template from phase2_iu_without_relationships.txt
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def get_level_probs(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> IULevelLogitResult:
        """
        Get probabilities for "one", "two", and "three" tokens.
        
        Args:
            image: Image to evaluate (PIL Image, path, or URL)
            prompt: Formatted IU prompt
            
        Returns:
            IULevelLogitResult with probabilities for the three levels
        """
        pass
    
    def compute_iu_score(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> float:
        """
        Compute IU score for an image-prompt pair.
        
        This is a convenience method that extracts level probabilities
        and applies IU scoring.
        
        Args:
            image: Image to evaluate
            prompt: Formatted prompt
            
        Returns:
            IU score (0.5 * P("two") + P("three"))
        """
        result = self.get_level_probs(image, prompt)
        return result.iu_score
