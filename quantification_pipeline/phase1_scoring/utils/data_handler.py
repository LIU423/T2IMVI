"""
Data handler for phase1_scoring verification pipeline.

Handles loading idiom extraction data, image files, prompt templates,
and saving verification results with checkpoint support.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ElementInfo:
    """Information about an entity or action element."""
    id: str
    content: str
    element_type: str  # "entity" or "action"
    rationale: Optional[str] = None  # Only for figurative elements
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class IdiomData:
    """Container for idiom extraction data."""
    idiom_id: int
    idiom: str
    definition: Optional[str] = None
    
    # Figurative track
    figurative_entities: List[ElementInfo] = field(default_factory=list)
    figurative_actions: List[ElementInfo] = field(default_factory=list)
    
    # Literal track
    literal_entities: List[ElementInfo] = field(default_factory=list)
    literal_actions: List[ElementInfo] = field(default_factory=list)
    
    # Raw data for output preservation
    raw_figurative: Optional[Dict] = None
    raw_literal: Optional[Dict] = None


@dataclass
class ImageInfo:
    """Information about an image file."""
    idiom_id: int
    image_num: int  # e.g., 1 for "1.jpeg"
    image_path: Path
    
    @property
    def image_id(self) -> str:
        """Get image identifier string."""
        return f"image_{self.image_num}"


class DataHandler:
    """
    Handles all data I/O for the verification pipeline.
    
    Responsibilities:
    - Load idiom extraction data (figurative.json, literal.json)
    - Enumerate image files for each idiom
    - Load prompt templates
    - Save verification results
    """
    
    def __init__(
        self,
        input_images_dir: Path,
        extraction_output_dir: Path,
        figurative_prompt_file: Path,
        literal_prompt_file: Path,
        output_dir: Path,
    ):
        """
        Initialize data handler.
        
        Args:
            input_images_dir: Directory containing matched_images/<id>/ folders
            extraction_output_dir: Directory containing idiom_<id>/ extraction outputs
            figurative_prompt_file: Path to figurative verifier prompt template
            literal_prompt_file: Path to literal verifier prompt template
            output_dir: Base output directory for verification results
        """
        self.input_images_dir = Path(input_images_dir)
        self.extraction_output_dir = Path(extraction_output_dir)
        self.figurative_prompt_file = Path(figurative_prompt_file)
        self.literal_prompt_file = Path(literal_prompt_file)
        self.output_dir = Path(output_dir)
        
        # Cached prompts
        self._figurative_prompt: Optional[str] = None
        self._literal_prompt: Optional[str] = None
    
    def load_prompts(self) -> Tuple[str, str]:
        """
        Load prompt templates from files.
        
        Returns:
            Tuple of (figurative_prompt, literal_prompt)
        """
        if self._figurative_prompt is None:
            with open(self.figurative_prompt_file, "r", encoding="utf-8") as f:
                self._figurative_prompt = f.read().strip()
            logger.info(f"Loaded figurative prompt from {self.figurative_prompt_file}")
        
        if self._literal_prompt is None:
            with open(self.literal_prompt_file, "r", encoding="utf-8") as f:
                self._literal_prompt = f.read().strip()
            logger.info(f"Loaded literal prompt from {self.literal_prompt_file}")
        
        return self._figurative_prompt, self._literal_prompt
    
    @property
    def figurative_prompt(self) -> str:
        """Get figurative verifier prompt template."""
        if self._figurative_prompt is None:
            self.load_prompts()
        return self._figurative_prompt
    
    @property
    def literal_prompt(self) -> str:
        """Get literal verifier prompt template."""
        if self._literal_prompt is None:
            self.load_prompts()
        return self._literal_prompt
    
    def get_available_idiom_ids(self) -> List[int]:
        """
        Get list of idiom IDs that have both extraction data and images.
        
        Returns:
            Sorted list of available idiom IDs
        """
        # Check extraction output directory for idiom folders
        extraction_idioms = set()
        for folder in self.extraction_output_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("idiom_"):
                try:
                    idiom_id = int(folder.name.split("_")[1])
                    # Verify both figurative.json and literal.json exist
                    if (folder / "figurative.json").exists() and (folder / "literal.json").exists():
                        extraction_idioms.add(idiom_id)
                except (ValueError, IndexError):
                    continue
        
        # Check image directory for matching folders
        available_idioms = []
        for idiom_id in extraction_idioms:
            image_folder = self.input_images_dir / str(idiom_id)
            if image_folder.exists():
                # Check if there are any jpeg files
                jpeg_files = list(image_folder.glob("*.jpeg")) + list(image_folder.glob("*.jpg"))
                if jpeg_files:
                    available_idioms.append(idiom_id)
        
        return sorted(available_idioms)
    
    def load_idiom_data(self, idiom_id: int) -> IdiomData:
        """
        Load extraction data for a specific idiom.
        
        Args:
            idiom_id: The idiom ID to load
            
        Returns:
            IdiomData containing parsed extraction results
        """
        idiom_folder = self.extraction_output_dir / f"idiom_{idiom_id}"
        
        # Load figurative.json
        figurative_path = idiom_folder / "figurative.json"
        with open(figurative_path, "r", encoding="utf-8") as f:
            figurative_data = json.load(f)
        
        # Load literal.json
        literal_path = idiom_folder / "literal.json"
        with open(literal_path, "r", encoding="utf-8") as f:
            literal_data = json.load(f)
        
        # Parse figurative elements
        figurative_entities = []
        figurative_actions = []
        
        fig_track = figurative_data.get("figurative_track", {})
        for entity in fig_track.get("entities", []):
            figurative_entities.append(ElementInfo(
                id=entity.get("id", ""),
                content=entity.get("content", ""),
                element_type="entity",
                rationale=entity.get("rationale", ""),
                extra_fields={k: v for k, v in entity.items() if k not in ["id", "content", "rationale"]}
            ))
        
        for action in fig_track.get("actions", []):
            figurative_actions.append(ElementInfo(
                id=action.get("id", ""),
                content=action.get("content", ""),
                element_type="action",
                rationale=action.get("rationale", ""),
                extra_fields={k: v for k, v in action.items() if k not in ["id", "content", "rationale"]}
            ))
        
        # Parse literal elements
        literal_entities = []
        literal_actions = []
        
        lit_track = literal_data.get("literal_track", {})
        for entity in lit_track.get("entities", []):
            literal_entities.append(ElementInfo(
                id=entity.get("id", ""),
                content=entity.get("content", ""),
                element_type="entity",
                extra_fields={k: v for k, v in entity.items() if k not in ["id", "content"]}
            ))
        
        for action in lit_track.get("actions", []):
            literal_actions.append(ElementInfo(
                id=action.get("id", ""),
                content=action.get("content", ""),
                element_type="action",
                extra_fields={k: v for k, v in action.items() if k not in ["id", "content"]}
            ))
        
        return IdiomData(
            idiom_id=idiom_id,
            idiom=figurative_data.get("idiom", literal_data.get("idiom", "")),
            definition=figurative_data.get("definition"),
            figurative_entities=figurative_entities,
            figurative_actions=figurative_actions,
            literal_entities=literal_entities,
            literal_actions=literal_actions,
            raw_figurative=figurative_data,
            raw_literal=literal_data,
        )
    
    def get_images_for_idiom(self, idiom_id: int) -> List[ImageInfo]:
        """
        Get list of image files for a specific idiom.
        
        Args:
            idiom_id: The idiom ID
            
        Returns:
            List of ImageInfo objects, sorted by image number
        """
        image_folder = self.input_images_dir / str(idiom_id)
        
        if not image_folder.exists():
            return []
        
        images = []
        for img_path in image_folder.iterdir():
            if img_path.suffix.lower() in [".jpeg", ".jpg", ".png"]:
                try:
                    # Extract image number from filename (e.g., "1.jpeg" -> 1)
                    img_num = int(img_path.stem)
                    images.append(ImageInfo(
                        idiom_id=idiom_id,
                        image_num=img_num,
                        image_path=img_path,
                    ))
                except ValueError:
                    # Skip files with non-numeric names
                    continue
        
        return sorted(images, key=lambda x: x.image_num)
    
    def load_image(self, image_info: ImageInfo) -> Image.Image:
        """
        Load image from file.
        
        Args:
            image_info: ImageInfo object
            
        Returns:
            PIL Image in RGB format
        """
        return Image.open(image_info.image_path).convert("RGB")
    
    def get_output_path(self, idiom_id: int, image_num: int) -> Path:
        """
        Get output directory path for a specific idiom-image pair.
        
        Args:
            idiom_id: The idiom ID
            image_num: The image number
            
        Returns:
            Path to output directory (e.g., .../idiom_1/image_1/)
        """
        return self.output_dir / f"idiom_{idiom_id}" / f"image_{image_num}"
    
    def save_verification_results(
        self,
        idiom_id: int,
        image_num: int,
        figurative_results: Dict[str, float],  # {element_id: score}
        literal_results: Dict[str, float],  # {element_id: score}
        idiom_data: IdiomData,
    ) -> None:
        """
        Save verification results for an idiom-image pair.
        
        Creates figurative.json and literal.json in the output directory,
        preserving original structure but adding 'score' field to each element.
        
        Args:
            idiom_id: The idiom ID
            image_num: The image number
            figurative_results: Dict mapping element IDs to scores
            literal_results: Dict mapping element IDs to scores
            idiom_data: Original idiom data for structure preservation
        """
        output_path = self.get_output_path(idiom_id, image_num)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build figurative output
        figurative_output = self._build_scored_figurative(
            idiom_data.raw_figurative,
            figurative_results,
        )
        
        # Build literal output
        literal_output = self._build_scored_literal(
            idiom_data.raw_literal,
            literal_results,
        )
        
        # Save files
        with open(output_path / "figurative.json", "w", encoding="utf-8") as f:
            json.dump(figurative_output, f, indent=2, ensure_ascii=False)
        
        with open(output_path / "literal.json", "w", encoding="utf-8") as f:
            json.dump(literal_output, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved results to {output_path}")
    
    def _build_scored_figurative(
        self,
        raw_data: Dict,
        scores: Dict[str, float],
    ) -> Dict:
        """Build figurative output with scores added."""
        import copy
        output = copy.deepcopy(raw_data)
        
        fig_track = output.get("figurative_track", {})
        
        # Add scores to entities
        for entity in fig_track.get("entities", []):
            entity_id = entity.get("id", "")
            if entity_id in scores:
                entity["score"] = scores[entity_id]
        
        # Add scores to actions
        for action in fig_track.get("actions", []):
            action_id = action.get("id", "")
            if action_id in scores:
                action["score"] = scores[action_id]
        
        return output
    
    def _build_scored_literal(
        self,
        raw_data: Dict,
        scores: Dict[str, float],
    ) -> Dict:
        """Build literal output with scores added."""
        import copy
        output = copy.deepcopy(raw_data)
        
        lit_track = output.get("literal_track", {})
        
        # Add scores to entities
        for entity in lit_track.get("entities", []):
            entity_id = entity.get("id", "")
            if entity_id in scores:
                entity["score"] = scores[entity_id]
        
        # Add scores to actions
        for action in lit_track.get("actions", []):
            action_id = action.get("id", "")
            if action_id in scores:
                action["score"] = scores[action_id]
        
        return output
    
    def result_exists(self, idiom_id: int, image_num: int) -> bool:
        """
        Check if verification results already exist for an idiom-image pair.
        
        Used for checkpoint/resume functionality.
        
        Args:
            idiom_id: The idiom ID
            image_num: The image number
            
        Returns:
            True if both figurative.json and literal.json exist
        """
        output_path = self.get_output_path(idiom_id, image_num)
        figurative_exists = (output_path / "figurative.json").exists()
        literal_exists = (output_path / "literal.json").exists()
        return figurative_exists and literal_exists
    
    def iter_pending_work(
        self,
        idiom_ids: Optional[List[int]] = None,
        max_images_per_idiom: Optional[int] = None,
    ) -> Generator[Tuple[IdiomData, ImageInfo], None, None]:
        """
        Iterate over pending (not yet processed) idiom-image pairs.
        
        Supports checkpoint/resume by skipping already completed work.
        
        Args:
            idiom_ids: Optional list of specific idiom IDs to process
            max_images_per_idiom: Optional limit on images per idiom (for testing)
            
        Yields:
            Tuples of (IdiomData, ImageInfo) for pending work
        """
        if idiom_ids is None:
            idiom_ids = self.get_available_idiom_ids()
        
        for idiom_id in idiom_ids:
            try:
                idiom_data = self.load_idiom_data(idiom_id)
            except Exception as e:
                logger.warning(f"Failed to load idiom {idiom_id}: {e}")
                continue
            
            images = self.get_images_for_idiom(idiom_id)
            
            if max_images_per_idiom is not None:
                images = images[:max_images_per_idiom]
            
            for image_info in images:
                # Check if already processed (checkpoint support)
                if self.result_exists(idiom_id, image_info.image_num):
                    logger.debug(f"Skipping completed: idiom_{idiom_id}/image_{image_info.image_num}")
                    continue
                
                yield idiom_data, image_info
