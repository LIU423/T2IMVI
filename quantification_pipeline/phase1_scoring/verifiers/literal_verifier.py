"""
Literal element verifier.

Verifies the presence of literal/concrete objects in images
using the forensic visual auditor prompt.
"""

import logging
from typing import Dict, List, Union
from pathlib import Path
from PIL import Image

from ..models.base_model import BaseVerifierModel, LogitResult
from ..utils.data_handler import ElementInfo

logger = logging.getLogger(__name__)


class LiteralVerifier:
    """
    Verifies literal elements in images.
    
    For each element (entity or action), uses the VLM to determine
    if the physical object is explicitly visible in the image,
    strictly ignoring metaphorical interpretations.
    
    Uses VQAScore methodology: P("Yes"|image, question) as verification score.
    """
    
    def __init__(
        self,
        model: BaseVerifierModel,
        prompt_template: str,
    ):
        """
        Initialize literal verifier.
        
        Args:
            model: Vision-Language model for verification
            prompt_template: System prompt template from phase1_literal_verifier_specialist.txt
        """
        self.model = model
        self.prompt_template = prompt_template
    
    def verify_element(
        self,
        image: Union[Image.Image, Path, str],
        element: ElementInfo,
    ) -> float:
        """
        Verify a single literal element in an image.
        
        Args:
            image: The image to verify against
            element: The literal element (entity or action)
            
        Returns:
            Verification score (P("Yes") probability)
        """
        # Format prompt with element content only (no rationale for literal)
        prompt = self.model.format_literal_prompt(
            content=element.content,
            system_prompt=self.prompt_template,
        )
        
        # Get yes/no probabilities
        result = self.model.get_yes_no_probs(image, prompt)
        
        logger.debug(
            f"Literal verify '{element.content}': "
            f"P(Yes)={result.yes_prob:.4f}, P(No)={result.no_prob:.4f}"
        )
        
        return result.yes_prob
    
    def verify_all_elements(
        self,
        image: Union[Image.Image, Path, str],
        entities: List[ElementInfo],
        actions: List[ElementInfo],
    ) -> Dict[str, float]:
        """
        Verify all literal elements for an image.
        
        Args:
            image: The image to verify against
            entities: List of literal entities
            actions: List of literal actions
            
        Returns:
            Dict mapping element IDs to verification scores
        """
        results = {}
        
        # Verify entities
        for entity in entities:
            score = self.verify_element(image, entity)
            results[entity.id] = score
            logger.info(f"  Entity '{entity.content}' ({entity.id}): score={score:.4f}")
        
        # Verify actions
        for action in actions:
            score = self.verify_element(image, action)
            results[action.id] = score
            logger.info(f"  Action '{action.content}' ({action.id}): score={score:.4f}")
        
        return results
