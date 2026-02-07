"""
Abstract base class for IU (Image Understanding) models.

This module defines the interface that all VLM implementations must follow
for the Phase 2 IU calculation pipeline.

IU uses VQAScore methodology: P("yes") from yes/no questions about
whether the image embodies the core abstract concept.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List

from PIL import Image


@dataclass
class YesNoLogitResult:
    """
    Result from yes/no logit extraction for VQAScore-style scoring.
    
    The IU score is P("yes") - the probability that the model answers "yes"
    to whether the image embodies the concept.
    
    Attributes:
        yes_logit: Raw logit for "yes" token
        no_logit: Raw logit for "no" token
        yes_prob: Normalized probability for "yes"
        no_prob: Normalized probability for "no"
    """
    yes_logit: float
    no_logit: float
    yes_prob: float
    no_prob: float
    
    @property
    def iu_score(self) -> float:
        """
        Compute IU score as P("yes").
        
        Following VQAScore methodology:
        VQAScore(image, text) := P("Yes" | image, question(text))
        """
        return self.yes_prob


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
    def get_yes_no_probs(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> YesNoLogitResult:
        """
        Get probabilities for "yes" and "no" tokens.
        
        Args:
            image: Image to evaluate (PIL Image, path, or URL)
            prompt: Formatted IU prompt
            
        Returns:
            YesNoLogitResult with probabilities for yes/no
        """
        pass
    
    def compute_iu_score(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> float:
        """
        Compute IU score for an image-prompt pair.
        
        This is a convenience method that extracts P("yes").
        
        Args:
            image: Image to evaluate
            prompt: Formatted prompt
            
        Returns:
            IU score (P("yes"))
        """
        result = self.get_yes_no_probs(image, prompt)
        return result.iu_score
