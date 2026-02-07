"""
Checkpoint manager for phase1_scoring verification pipeline.

Handles saving and loading progress to enable resumption after interruption.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """Data structure for checkpoint information."""
    
    # Completed work tracking
    completed_images: Dict[str, Set[str]] = field(default_factory=dict)
    # Format: {"idiom_1": {"image_1", "image_2", ...}, ...}
    
    # Metadata
    model_name: str = ""
    started_at: str = ""
    last_updated: str = ""
    total_processed: int = 0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "completed_images": {k: list(v) for k, v in self.completed_images.items()},
            "model_name": self.model_name,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_processed": self.total_processed,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointData":
        """Create from dict."""
        return cls(
            completed_images={k: set(v) for k, v in data.get("completed_images", {}).items()},
            model_name=data.get("model_name", ""),
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
            total_processed=data.get("total_processed", 0),
        )


class CheckpointManager:
    """
    Manages checkpointing for the verification pipeline.
    
    Tracks which idiom-image pairs have been processed to enable
    resumption after interruption.
    """
    
    def __init__(self, checkpoint_file: Path, model_name: str = ""):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_file: Path to checkpoint JSON file
            model_name: Name of the model being used
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.data = CheckpointData(
            model_name=model_name,
            started_at=datetime.now().isoformat(),
        )
    
    def load(self) -> bool:
        """
        Load checkpoint from file.
        
        Returns:
            True if checkpoint was loaded, False if starting fresh
        """
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint found, starting fresh")
            return False
        
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            self.data = CheckpointData.from_dict(raw_data)
            logger.info(
                f"Loaded checkpoint: {self.data.total_processed} images processed, "
                f"last updated {self.data.last_updated}"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            return False
    
    def save(self) -> None:
        """Save checkpoint to file."""
        self.data.last_updated = datetime.now().isoformat()
        
        # Ensure parent directory exists
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.data.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Checkpoint saved: {self.data.total_processed} images processed")
    
    def mark_completed(self, idiom_id: str, image_id: str) -> None:
        """
        Mark an idiom-image pair as completed.
        
        Args:
            idiom_id: Idiom identifier (e.g., "idiom_3")
            image_id: Image identifier (e.g., "image_1")
        """
        if idiom_id not in self.data.completed_images:
            self.data.completed_images[idiom_id] = set()
        
        self.data.completed_images[idiom_id].add(image_id)
        self.data.total_processed += 1
    
    def is_completed(self, idiom_id: str, image_id: str) -> bool:
        """
        Check if an idiom-image pair has been completed.
        
        Args:
            idiom_id: Idiom identifier
            image_id: Image identifier
            
        Returns:
            True if already processed
        """
        if idiom_id not in self.data.completed_images:
            return False
        return image_id in self.data.completed_images[idiom_id]
    
    def get_completed_count(self) -> int:
        """Get total number of completed images."""
        return self.data.total_processed
    
    def get_completed_idioms(self) -> Set[str]:
        """Get set of idiom IDs that have at least one completed image."""
        return set(self.data.completed_images.keys())
