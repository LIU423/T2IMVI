"""
Checkpoint management for Phase 2 AEA calculation pipeline.

Provides functionality to save and restore pipeline state,
enabling resumption from interruptions.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Set, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """
    Checkpoint data structure.
    
    Attributes:
        completed_images: Set of "idiom_<id>/image_<num>" strings that are done
        model_name: Name of the model used
        started_at: ISO timestamp of when processing started
        last_updated: ISO timestamp of last checkpoint save
        total_processed: Count of processed images
    """
    completed_images: Set[str] = field(default_factory=set)
    model_name: str = ""
    started_at: str = ""
    last_updated: str = ""
    total_processed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "completed_images": list(self.completed_images),
            "model_name": self.model_name,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_processed": self.total_processed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Create from dictionary."""
        return cls(
            completed_images=set(data.get("completed_images", [])),
            model_name=data.get("model_name", ""),
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
            total_processed=data.get("total_processed", 0),
        )


class CheckpointManager:
    """
    Manages checkpoint saving and loading for the AEA pipeline.
    
    Checkpoints track which images have been processed, enabling
    resume functionality after interruptions.
    """
    
    def __init__(self, checkpoint_path: Path, model_name: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            model_name: Name of the model being used
        """
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self._data: Optional[CheckpointData] = None
    
    @property
    def data(self) -> CheckpointData:
        """Get current checkpoint data, loading if necessary."""
        if self._data is None:
            self._data = self.load()
        return self._data
    
    def load(self) -> CheckpointData:
        """
        Load checkpoint from file.
        
        Returns:
            CheckpointData with previous state or fresh state if no checkpoint
        """
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                data = CheckpointData.from_dict(raw_data)
                
                # Validate model name matches
                if data.model_name and data.model_name != self.model_name:
                    logger.warning(
                        f"Checkpoint model '{data.model_name}' differs from "
                        f"current model '{self.model_name}'. Starting fresh."
                    )
                    return self._create_fresh()
                
                logger.info(
                    f"Loaded checkpoint: {data.total_processed} images completed"
                )
                return data
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                return self._create_fresh()
        else:
            logger.info("No checkpoint found, starting fresh")
            return self._create_fresh()
    
    def _create_fresh(self) -> CheckpointData:
        """Create fresh checkpoint data."""
        return CheckpointData(
            model_name=self.model_name,
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )
    
    def save(self) -> None:
        """Save current checkpoint to file."""
        if self._data is None:
            return
        
        self._data.last_updated = datetime.now().isoformat()
        
        # Ensure parent directory exists
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(self._data.to_dict(), f, indent=2)
        
        logger.debug(f"Checkpoint saved: {self._data.total_processed} images")
    
    def mark_completed(self, idiom_id: int, image_id: int) -> None:
        """
        Mark an image as completed.
        
        Args:
            idiom_id: Idiom ID
            image_id: Image ID
        """
        key = f"idiom_{idiom_id}/image_{image_id}"
        self.data.completed_images.add(key)
        self.data.total_processed = len(self.data.completed_images)
    
    def is_completed(self, idiom_id: int, image_id: int) -> bool:
        """
        Check if an image has been completed.
        
        Args:
            idiom_id: Idiom ID
            image_id: Image ID
            
        Returns:
            True if image has been processed
        """
        key = f"idiom_{idiom_id}/image_{image_id}"
        return key in self.data.completed_images
    
    def get_completed_count(self) -> int:
        """Get count of completed images."""
        return self.data.total_processed
    
    def reset(self) -> None:
        """Reset checkpoint to fresh state."""
        self._data = self._create_fresh()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        logger.info("Checkpoint reset")
