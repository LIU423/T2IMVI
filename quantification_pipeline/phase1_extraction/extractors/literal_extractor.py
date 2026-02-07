"""
Literal Extractor - Extract literal scene graph from idioms.

Based on phase1_literal_extraction_specialist.txt prompt.
"""

import json
import logging
from typing import Dict, Any, Type
from pathlib import Path

from pydantic import BaseModel

from extractors.base_extractor import BaseExtractor
from models.base_model import BaseExtractionModel
from schemas.literal_schema import LiteralExtractionResult

logger = logging.getLogger(__name__)


class LiteralExtractor(BaseExtractor):
    """
    Extractor for literal scene graphs.
    
    Generates morphological visualization based on explicit words
    and inferred implicit agents from idiom text.
    """
    
    def __init__(
        self,
        model: BaseExtractionModel,
        prompt_template: str,
        output_dir: Path,
    ):
        super().__init__(model, prompt_template, output_dir)
        logger.info(f"LiteralExtractor initialized. Output dir: {output_dir}")
    
    def get_prompt(self, idiom_data: Dict[str, Any]) -> str:
        """
        Build prompt for literal extraction.
        
        Expected idiom_data keys:
        - idiom: The idiom text
        """
        idiom = idiom_data.get("idiom", "")
        
        # Replace placeholder in template
        prompt = self.prompt_template.replace("<idiom>", idiom)
        
        return prompt
    
    def get_schema(self) -> Type[BaseModel]:
        """Return the literal extraction schema."""
        return LiteralExtractionResult
    
    def extract_and_save(self, idiom_data: Dict[str, Any]) -> LiteralExtractionResult:
        """
        Extract literal elements and save to file.
        
        Args:
            idiom_data: Dictionary with idiom_id, idiom, definition
            
        Returns:
            Extracted LiteralExtractionResult
        """
        idiom_id = idiom_data.get("idiom_id")
        idiom = idiom_data.get("idiom", "")
        
        logger.info(f"Extracting literal elements for idiom {idiom_id}: '{idiom}'")
        
        # Check if already processed (checkpoint)
        output_path = self.get_output_path(idiom_id, "literal")
        if output_path.exists():
            logger.info(f"Idiom {idiom_id} already processed, loading from cache...")
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return LiteralExtractionResult.model_validate(data)
        
        # Extract
        result = self.extract(idiom_data)
        
        # Save result
        output_data = {
            "idiom_id": idiom_id,
            "idiom": idiom,
            **result.model_dump()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved literal extraction for idiom {idiom_id} to {output_path}")
        
        return result
