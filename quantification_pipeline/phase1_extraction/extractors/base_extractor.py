"""
Base Extractor - Abstract interface for all extractors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from pathlib import Path
from pydantic import BaseModel

from models.base_model import BaseExtractionModel


class BaseExtractor(ABC):
    """
    Abstract base class for idiom element extractors.
    
    Subclasses must implement:
    - extract(): Perform extraction for a single idiom
    - get_prompt(): Build prompt for the idiom
    - get_schema(): Return the Pydantic schema for validation
    - track_type: Property returning the track type ("literal" or "figurative")
    """
    
    def __init__(
        self,
        model: BaseExtractionModel,
        prompt_template: str,
        output_dir: Path,
    ):
        """
        Initialize extractor.
        
        Args:
            model: LLM model instance for generation
            prompt_template: Template string with placeholders
            output_dir: Directory to save extraction results
        """
        self.model = model
        self.prompt_template = prompt_template
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def track_type(self) -> str:
        """Return the track type: 'literal' or 'figurative'."""
        pass
    
    @abstractmethod
    def get_prompt(self, idiom_data: Dict[str, Any]) -> str:
        """
        Build the prompt for extraction.
        
        Args:
            idiom_data: Dictionary containing idiom information
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Type[BaseModel]:
        """
        Get the Pydantic schema for output validation.
        
        Returns:
            Pydantic BaseModel class
        """
        pass
    
    def extract(self, idiom_data: Dict[str, Any]) -> BaseModel:
        """
        Extract elements from a single idiom.
        
        Args:
            idiom_data: Dictionary with idiom info (idiom_id, idiom, definition)
            
        Returns:
            Validated Pydantic model instance
        """
        idiom_id = idiom_data.get("idiom_id", 0)
        
        # Set extraction context for raw response logging (if model supports it)
        if hasattr(self.model, 'set_extraction_context'):
            self.model.set_extraction_context(idiom_id, self.track_type)
        
        prompt = self.get_prompt(idiom_data)
        schema = self.get_schema()
        
        result = self.model.generate_structured(prompt, schema)
        return result
    
    def get_output_path(self, idiom_id: int, track_type: str) -> Path:
        """
        Get output file path for an idiom.
        
        New structure: output_dir/idiom_<id>/<track_type>.json
        
        Args:
            idiom_id: The idiom ID
            track_type: Either 'figurative' or 'literal'
            
        Returns:
            Path to the output JSON file
        """
        idiom_dir = self.output_dir / f"idiom_{idiom_id}"
        idiom_dir.mkdir(parents=True, exist_ok=True)
        return idiom_dir / f"{track_type}.json"
    
    def result_exists(self, idiom_id: int, track_type: str) -> bool:
        """Check if result already exists (for checkpoint recovery)."""
        return self.get_output_path(idiom_id, track_type).exists()
