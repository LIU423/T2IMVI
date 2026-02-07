"""
Checkpoint Manager - Handles extraction progress persistence.

Supports:
- Tracking processed idiom IDs
- Resume from interruption
- Error logging for failed extractions
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Set, Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """State persisted in checkpoint file."""
    
    # Track type: "literal" or "figurative"
    track_type: str
    
    # Processed idiom IDs
    processed_ids: Set[int] = field(default_factory=set)
    
    # Failed idiom IDs with error messages
    failed_ids: Dict[int, str] = field(default_factory=dict)
    
    # Metadata
    started_at: str = ""
    last_updated: str = ""
    total_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        # Ensure failed_ids keys are strings for JSON compatibility
        failed_ids_str = {str(k): v for k, v in self.failed_ids.items()}
        return {
            "track_type": self.track_type,
            "processed_ids": list(self.processed_ids),
            "failed_ids": failed_ids_str,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_count": self.total_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dict."""
        # Convert string keys in failed_ids back to int
        raw_failed = data.get("failed_ids", {})
        failed_ids = {int(k): v for k, v in raw_failed.items()}
        
        return cls(
            track_type=data.get("track_type", ""),
            processed_ids=set(data.get("processed_ids", [])),
            failed_ids=failed_ids,
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
            total_count=data.get("total_count", 0),
        )


class CheckpointManager:
    """
    Manages extraction checkpoints for resume capability.
    
    Features:
    - Persist progress to JSON file
    - Load previous state on startup
    - Track failed extractions separately
    - Provide statistics
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        track_type: str,
        total_count: int = 0,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            track_type: "literal" or "figurative"
            total_count: Total number of idioms to process
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / f"{track_type}_checkpoint.json"
        self.track_type = track_type
        
        # Load or create state
        self.state = self._load_or_create_state(total_count)
        
        logger.info(
            f"CheckpointManager initialized for {track_type}. "
            f"Progress: {len(self.state.processed_ids)}/{self.state.total_count}"
        )
    
    def _load_or_create_state(self, total_count: int) -> CheckpointState:
        """Load existing checkpoint or create new one."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                state = CheckpointState.from_dict(data)
                
                # Update total count if provided
                if total_count > 0:
                    state.total_count = total_count
                
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                return state
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load checkpoint: {e}. Creating new one.")
        
        # Create new state
        now = datetime.now().isoformat()
        return CheckpointState(
            track_type=self.track_type,
            started_at=now,
            last_updated=now,
            total_count=total_count,
        )
    
    def save(self) -> None:
        """Persist current state to file."""
        self.state.last_updated = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def mark_processed(self, idiom_id: int) -> None:
        """Mark an idiom as successfully processed."""
        self.state.processed_ids.add(idiom_id)
        
        # Remove from failed if previously failed (check both int and str keys)
        str_id = str(idiom_id)
        if idiom_id in self.state.failed_ids:
            del self.state.failed_ids[idiom_id]
        elif str_id in self.state.failed_ids:
            del self.state.failed_ids[str_id]
        
        self.save()
    
    def mark_failed(self, idiom_id: int, error_msg: str) -> None:
        """Mark an idiom as failed with error message."""
        self.state.failed_ids[idiom_id] = error_msg
        self.save()
    
    def is_processed(self, idiom_id: int) -> bool:
        """Check if idiom has been processed."""
        return idiom_id in self.state.processed_ids
    
    def get_pending_ids(self, all_ids: List[int]) -> List[int]:
        """Get list of idiom IDs that haven't been processed yet."""
        return [
            id_ for id_ in all_ids 
            if id_ not in self.state.processed_ids
        ]
    
    def get_failed_ids(self) -> Dict[int, str]:
        """Get dict of failed idiom IDs with error messages."""
        return self.state.failed_ids.copy()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        processed = len(self.state.processed_ids)
        failed = len(self.state.failed_ids)
        total = self.state.total_count
        
        return {
            "processed": processed,
            "failed": failed,
            "pending": total - processed,
            "total": total,
            "progress_pct": (processed / total * 100) if total > 0 else 0,
            "started_at": self.state.started_at,
            "last_updated": self.state.last_updated,
        }
    
    def reset(self) -> None:
        """Reset checkpoint (clear all progress)."""
        now = datetime.now().isoformat()
        self.state = CheckpointState(
            track_type=self.track_type,
            started_at=now,
            last_updated=now,
            total_count=self.state.total_count,
        )
        self.save()
        logger.info(f"Checkpoint reset for {self.track_type}")
