"""
Data handling utilities for Phase 2 AEA calculation pipeline.

This module manages:
- Loading figurative.json files from Phase 1 output
- Loading images from the dataset
- Loading the AEA prompt template
- Saving figurative_score.json with AEA scores
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """
    Information about a single image to process.
    
    Attributes:
        idiom_id: Idiom ID
        image_id: Image number
        image_path: Path to the image file
        figurative_json_path: Path to figurative.json (input)
        figurative_score_json_path: Path to figurative_score.json (output)
    """
    idiom_id: int
    image_id: int
    image_path: Path
    figurative_json_path: Path
    figurative_score_json_path: Path
    
    @property
    def key(self) -> str:
        """Unique key for this image."""
        return f"idiom_{self.idiom_id}/image_{self.image_id}"


@dataclass  
class FigurativeData:
    """
    Data loaded from a figurative.json file.
    
    Attributes:
        idiom_id: Idiom ID
        idiom: Idiom text
        abstract_atmosphere: The abstract atmosphere description for AEA
        raw_data: Full raw JSON data
    """
    idiom_id: int
    idiom: str
    abstract_atmosphere: str
    raw_data: Dict[str, Any]
    
    def get_all_scores(self) -> List[float]:
        """
        Extract all scores from entities and actions.
        
        Returns:
            List of all score values from entities and actions
        """
        scores = []
        figurative_track = self.raw_data.get("figurative_track", {})
        
        # Get entity scores
        entities = figurative_track.get("entities", [])
        for entity in entities:
            if "score" in entity:
                scores.append(float(entity["score"]))
        
        # Get action scores
        actions = figurative_track.get("actions", [])
        for action in actions:
            if "score" in action:
                scores.append(float(action["score"]))
        
        return scores
    
    def has_significant_scores(self, threshold: float = 0.1) -> bool:
        """
        Check if any entity or action has a score >= threshold.
        
        Args:
            threshold: Minimum score to be considered significant (default 0.1)
            
        Returns:
            True if at least one score is >= threshold
        """
        scores = self.get_all_scores()
        if not scores:
            # No scores found - consider as not significant
            return False
        return any(score >= threshold for score in scores)
    
    def get_max_score(self) -> float:
        """
        Get the maximum score from entities and actions.
        
        Returns:
            Maximum score, or 0.0 if no scores found
        """
        scores = self.get_all_scores()
        return max(scores) if scores else 0.0


class DataHandler:
    """
    Handles all data I/O for the AEA pipeline.
    
    Responsibilities:
    - Discover idiom directories and image subdirectories
    - Load figurative.json files
    - Load images
    - Load AEA prompt template
    - Save figurative_score.json with AEA scores
    """
    
    def __init__(
        self,
        phase1_output_dir: Path,
        images_root: Path,
        prompt_file: Path,
    ):
        """
        Initialize data handler.
        
        Args:
            phase1_output_dir: Directory containing Phase 1 output
                (idiom_<id>/image_<num>/figurative.json)
            images_root: Root directory for images
                (<id>/<num>.jpeg)
            prompt_file: Path to AEA prompt template file
        """
        self.phase1_output_dir = phase1_output_dir
        self.images_root = images_root
        self.prompt_file = prompt_file
        self._prompt_template: Optional[str] = None
    
    @property
    def prompt_template(self) -> str:
        """Load and cache the AEA prompt template."""
        if self._prompt_template is None:
            self._prompt_template = self.load_prompt()
        return self._prompt_template
    
    def load_prompt(self) -> str:
        """
        Load the AEA prompt template from file.
        
        Returns:
            Prompt template string
        """
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Loaded AEA prompt from {self.prompt_file}")
        return content
    
    def discover_idioms(
        self,
        idiom_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Discover available idiom directories.
        
        Args:
            idiom_ids: Optional list of specific idiom IDs to filter by
            
        Returns:
            List of idiom IDs with available data
        """
        idiom_dirs = list(self.phase1_output_dir.glob("idiom_*"))
        
        available_ids = []
        for idiom_dir in idiom_dirs:
            try:
                idiom_id = int(idiom_dir.name.split("_")[1])
                if idiom_ids is None or idiom_id in idiom_ids:
                    available_ids.append(idiom_id)
            except (ValueError, IndexError):
                logger.warning(f"Invalid idiom directory: {idiom_dir}")
        
        available_ids.sort()
        logger.info(f"Discovered {len(available_ids)} idioms")
        return available_ids
    
    def discover_images(
        self,
        idiom_id: int,
        max_images: Optional[int] = None,
    ) -> List[ImageInfo]:
        """
        Discover image directories for an idiom.
        
        Args:
            idiom_id: Idiom ID to discover images for
            max_images: Maximum number of images to return (for testing)
            
        Returns:
            List of ImageInfo for available images
        """
        idiom_dir = self.phase1_output_dir / f"idiom_{idiom_id}"
        
        if not idiom_dir.exists():
            logger.warning(f"Idiom directory not found: {idiom_dir}")
            return []
        
        image_dirs = list(idiom_dir.glob("image_*"))
        
        images = []
        for image_dir in image_dirs:
            try:
                image_id = int(image_dir.name.split("_")[1])
                
                # Check if figurative.json exists
                figurative_json = image_dir / "figurative.json"
                if not figurative_json.exists():
                    logger.debug(f"No figurative.json in {image_dir}")
                    continue
                
                # Check if image exists
                image_path = self.images_root / str(idiom_id) / f"{image_id}.jpeg"
                if not image_path.exists():
                    logger.debug(f"Image not found: {image_path}")
                    continue
                
                images.append(ImageInfo(
                    idiom_id=idiom_id,
                    image_id=image_id,
                    image_path=image_path,
                    figurative_json_path=figurative_json,
                    figurative_score_json_path=image_dir / "figurative_score.json",
                ))
                
            except (ValueError, IndexError):
                logger.warning(f"Invalid image directory: {image_dir}")
        
        # Sort by image_id
        images.sort(key=lambda x: x.image_id)
        
        # Apply max_images limit
        if max_images is not None:
            images = images[:max_images]
        
        logger.debug(f"Discovered {len(images)} images for idiom_{idiom_id}")
        return images
    
    def discover_all_images(
        self,
        idiom_ids: Optional[List[int]] = None,
        max_images_per_idiom: Optional[int] = None,
    ) -> List[ImageInfo]:
        """
        Discover all images across all (or specified) idioms.
        
        Args:
            idiom_ids: Optional list of idiom IDs to filter by
            max_images_per_idiom: Max images per idiom (for testing)
            
        Returns:
            List of all ImageInfo objects
        """
        discovered_idioms = self.discover_idioms(idiom_ids)
        
        all_images = []
        for idiom_id in discovered_idioms:
            images = self.discover_images(idiom_id, max_images_per_idiom)
            all_images.extend(images)
        
        logger.info(f"Total images discovered: {len(all_images)}")
        return all_images
    
    def load_figurative_data(self, image_info: ImageInfo) -> FigurativeData:
        """
        Load figurative.json data for an image.
        
        Args:
            image_info: ImageInfo with paths
            
        Returns:
            FigurativeData with abstract_atmosphere extracted
        """
        with open(image_info.figurative_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract abstract_atmosphere from figurative_track
        figurative_track = data.get("figurative_track", {})
        abstract_atmosphere = figurative_track.get("abstract_atmosphere", "")
        
        if not abstract_atmosphere:
            logger.warning(
                f"No abstract_atmosphere in {image_info.figurative_json_path}"
            )
        
        return FigurativeData(
            idiom_id=data.get("idiom_id", image_info.idiom_id),
            idiom=data.get("idiom", ""),
            abstract_atmosphere=abstract_atmosphere,
            raw_data=data,
        )
    
    def load_image(self, image_info: ImageInfo) -> Image.Image:
        """
        Load image file.
        
        Args:
            image_info: ImageInfo with image path
            
        Returns:
            PIL Image in RGB format
        """
        return Image.open(image_info.image_path).convert("RGB")
    
    def save_aea_score(
        self,
        image_info: ImageInfo,
        aea_score: float,
    ) -> None:
        """
        Save AEA score to figurative_score.json.
        
        If the file already exists, adds/updates the 'aea' key.
        If not, creates a new file with just the 'aea' key.
        
        Args:
            image_info: ImageInfo with output path
            aea_score: Computed AEA score
        """
        output_path = image_info.figurative_score_json_path
        
        # Load existing data if file exists
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Add/update AEA score
        data["aea"] = aea_score
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved AEA score {aea_score:.4f} to {output_path}")
    
    def load_existing_aea_score(self, image_info: ImageInfo) -> Optional[float]:
        """
        Load existing AEA score if available.
        
        Args:
            image_info: ImageInfo with output path
            
        Returns:
            AEA score if exists, None otherwise
        """
        output_path = image_info.figurative_score_json_path
        
        if not output_path.exists():
            return None
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("aea")
        except (json.JSONDecodeError, KeyError):
            return None
