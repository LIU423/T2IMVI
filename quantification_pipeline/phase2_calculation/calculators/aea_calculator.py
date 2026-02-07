"""
AEA (Abstract Element Alignment) Calculator.

This module implements the AEA scoring logic that evaluates whether
an image aligns with the abstract atmosphere description.

Score = 1 - P("one")

Where "one" indicates a clash between image and description.
"""

import logging
from pathlib import Path
from typing import Union, Optional

from PIL import Image

from ..models.base_model import BaseAEAModel, LogitResult
from ..utils.data_handler import ImageInfo, FigurativeData

logger = logging.getLogger(__name__)


class AEACalculator:
    """
    Calculator for Abstract Element Alignment scores.
    
    Uses a Vision-Language Model to evaluate alignment between
    an image and its abstract atmosphere description.
    
    Scoring:
    - Model outputs probability for "one", "two", "three"
    - "one" = clash (bad alignment)
    - "two" = neutral (acceptable)
    - "three" = strong match (good alignment)
    - AEA Score = 1 - P("one")
    
    Pre-filtering:
    - If all entity/action scores from Phase 1 are below threshold,
      skip VLM evaluation and return AEA = 0.0
    """
    
    # Default threshold for skipping AEA calculation
    DEFAULT_SCORE_THRESHOLD = 0.1
    
    def __init__(
        self,
        model: BaseAEAModel,
        prompt_template: str,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ):
        """
        Initialize AEA calculator.
        
        Args:
            model: Loaded VLM model implementing BaseAEAModel
            prompt_template: The AEA prompt template from phase2_aea.txt
            score_threshold: Minimum score threshold for entities/actions.
                If all scores are below this, AEA is set to 0.0 without VLM call.
        """
        self.model = model
        self.prompt_template = prompt_template
        self.score_threshold = score_threshold
    
    def calculate(
        self,
        image: Union[Image.Image, Path, str],
        abstract_atmosphere: str,
    ) -> float:
        """
        Calculate AEA score for an image-text pair.
        
        Args:
            image: Image to evaluate
            abstract_atmosphere: Abstract atmosphere description
            
        Returns:
            AEA score (1 - P("one")), range [0, 1]
            Higher is better (less clash)
        """
        # Format prompt with the abstract_atmosphere
        prompt = self.model.format_aea_prompt(
            abstract_atmosphere=abstract_atmosphere,
            system_prompt=self.prompt_template,
        )
        
        # Get level probabilities from model
        result = self.model.get_level_probs(image, prompt)
        
        # Log the probabilities for debugging
        logger.debug(
            f"Level probs: one={result.one_prob:.4f}, "
            f"two={result.two_prob:.4f}, three={result.three_prob:.4f}"
        )
        
        # AEA score = 1 - P("one")
        return result.aea_score
    
    def calculate_with_details(
        self,
        image: Union[Image.Image, Path, str],
        abstract_atmosphere: str,
    ) -> tuple:
        """
        Calculate AEA score with full probability details.
        
        Args:
            image: Image to evaluate
            abstract_atmosphere: Abstract atmosphere description
            
        Returns:
            Tuple of (aea_score, LogitResult)
        """
        prompt = self.model.format_aea_prompt(
            abstract_atmosphere=abstract_atmosphere,
            system_prompt=self.prompt_template,
        )
        
        result = self.model.get_level_probs(image, prompt)
        
        return result.aea_score, result
    
    def calculate_for_image_info(
        self,
        image_info: ImageInfo,
        figurative_data: FigurativeData,
        image: Image.Image,
    ) -> float:
        """
        Calculate AEA score for an ImageInfo object.
        
        Convenience method that uses pre-loaded data.
        
        Pre-filtering: If all entity/action scores are below threshold,
        returns 0.0 without calling the VLM.
        
        Args:
            image_info: ImageInfo with paths (for logging)
            figurative_data: Loaded figurative data with abstract_atmosphere
            image: Pre-loaded PIL Image
            
        Returns:
            AEA score (0.0 if filtered out, otherwise 1 - P("one"))
        """
        # Check if there are any significant scores
        if not figurative_data.has_significant_scores(self.score_threshold):
            max_score = figurative_data.get_max_score()
            logger.info(
                f"Skipping AEA for {image_info.key}: "
                f"all scores below threshold {self.score_threshold} "
                f"(max score: {max_score:.4f}). Setting AEA=0.0"
            )
            return 0.0
        
        if not figurative_data.abstract_atmosphere:
            logger.warning(
                f"Empty abstract_atmosphere for {image_info.key}, returning 0.0"
            )
            return 0.0
        
        return self.calculate(image, figurative_data.abstract_atmosphere)
