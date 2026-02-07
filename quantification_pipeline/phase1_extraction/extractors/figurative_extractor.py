"""
Figurative Extractor - Extract figurative visual knowledge graph from idioms.

Based on phase1_figurative_extraction_specialist.txt prompt.
"""

import json
import logging
from typing import Dict, Any, Type
from pathlib import Path

from pydantic import BaseModel

from extractors.base_extractor import BaseExtractor
from models.base_model import BaseExtractionModel
from schemas.figurative_schema import FigurativeExtractionResult

logger = logging.getLogger(__name__)


class FigurativeExtractor(BaseExtractor):
    """
    Extractor for figurative visual knowledge graphs.
    
    Generates visual symbols and relationships based on
    the figurative/metaphorical meaning of idioms.
    """
    
    def __init__(
        self,
        model: BaseExtractionModel,
        prompt_template: str,
        output_dir: Path,
    ):
        super().__init__(model, prompt_template, output_dir)
        logger.info(f"FigurativeExtractor initialized. Output dir: {output_dir}")
    
    @property
    def track_type(self) -> str:
        """Return the track type for this extractor."""
        return "figurative"
    
    def get_prompt(self, idiom_data: Dict[str, Any]) -> str:
        """
        Build prompt for figurative extraction.
        
        Expected idiom_data keys:
        - idiom: The idiom text
        - definition: The idiom's definition/meaning
        """
        idiom = idiom_data.get("idiom", "")
        definition = idiom_data.get("definition", "")
        
        # Handle definition that may be a list string
        if isinstance(definition, str) and definition.startswith("["):
            # It's a string representation of a list
            pass  # Keep as is, prompt expects this format
        elif isinstance(definition, list):
            definition = str(definition)
        
        # Replace placeholders in template
        prompt = self.prompt_template.replace("<idiom>", idiom)
        prompt = prompt.replace("<definition>", definition)
        
        return prompt
    
    def get_schema(self) -> Type[BaseModel]:
        """Return the figurative extraction schema."""
        return FigurativeExtractionResult
    
    def extract_and_save(self, idiom_data: Dict[str, Any]) -> FigurativeExtractionResult:
        """
        Extract figurative elements and save to file.
        
        Args:
            idiom_data: Dictionary with idiom_id, idiom, definition
            
        Returns:
            Extracted FigurativeExtractionResult
        """
        idiom_id = idiom_data.get("idiom_id")
        idiom = idiom_data.get("idiom", "")
        
        logger.info(f"Extracting figurative elements for idiom {idiom_id}: '{idiom}'")
        
        # Check if already processed (checkpoint)
        output_path = self.get_output_path(idiom_id, "figurative")
        if output_path.exists():
            logger.info(f"Idiom {idiom_id} already processed, loading from cache...")
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return FigurativeExtractionResult.model_validate(data)
        
        # Extract
        result = self.extract(idiom_data)
        
        # Save result
        output_data = {
            "idiom_id": idiom_id,
            "idiom": idiom,
            "definition": idiom_data.get("definition", ""),
            **result.model_dump()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved figurative extraction for idiom {idiom_id} to {output_path}")
        
        return result
