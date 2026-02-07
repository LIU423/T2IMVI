"""
Data handling module with checkpoint support for resumable processing.

This module handles:
- Loading idioms from JSON
- Loading prompt templates
- Saving results with checkpoint support
- Resuming from last checkpoint
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class IdiomEntry:
    """Represents a single idiom entry from input data."""
    idiom_id: int
    idiom: str
    definition: str


@dataclass
class ImageabilityResult:
    """Result of imageability evaluation for a single idiom."""
    idiom_id: int
    idiom: str
    definition: str
    imageability: float  # P(yes) - the probability of "yes"
    yes_logit: Optional[float] = None
    no_logit: Optional[float] = None
    
    def to_output_dict(self) -> Dict[str, Any]:
        """Convert to output format matching example."""
        return {
            "idiom_id": self.idiom_id,
            "idiom": self.idiom,
            "definition": self.definition,
            "imageability": self.imageability,
        }
    
    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Convert to checkpoint format with full details."""
        return asdict(self)


class DataHandler:
    """
    Handles data loading, saving, and checkpoint management.
    
    Supports resumable processing by tracking which idiom IDs
    have been completed.
    """
    
    def __init__(
        self,
        idioms_file: Path,
        prompt_file: Path,
        output_file: Path,
        checkpoint_file: Path,
    ):
        self.idioms_file = idioms_file
        self.prompt_file = prompt_file
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        
        self._idioms: List[IdiomEntry] = []
        self._prompt: str = ""
        self._results: Dict[int, ImageabilityResult] = {}  # idiom_id -> result
        self._completed_ids: Set[int] = set()
    
    def load_idioms(self) -> List[IdiomEntry]:
        """Load idioms from JSON file."""
        with open(self.idioms_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self._idioms = []
        for item in data:
            entry = IdiomEntry(
                idiom_id=item["idiom_id"],
                idiom=item["idiom"],
                definition=item["definition"],
            )
            self._idioms.append(entry)
        
        print(f"Loaded {len(self._idioms)} idioms from {self.idioms_file}")
        return self._idioms
    
    def load_prompt(self) -> str:
        """Load prompt template from file."""
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            self._prompt = f.read().strip()
        print(f"Loaded prompt template ({len(self._prompt)} chars)")
        return self._prompt
    
    def load_checkpoint(self) -> bool:
        """
        Load checkpoint if exists. Returns True if checkpoint was loaded.
        """
        if not self.checkpoint_file.exists():
            print("No checkpoint found, starting fresh.")
            return False
        
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            # Restore completed results
            for item in checkpoint_data.get("results", []):
                result = ImageabilityResult(
                    idiom_id=item["idiom_id"],
                    idiom=item["idiom"],
                    definition=item["definition"],
                    imageability=item["imageability"],
                    yes_logit=item.get("yes_logit"),
                    no_logit=item.get("no_logit"),
                )
                self._results[result.idiom_id] = result
                self._completed_ids.add(result.idiom_id)
            
            print(f"Loaded checkpoint: {len(self._completed_ids)} items completed")
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            return False
    
    def get_pending_idioms(self) -> List[IdiomEntry]:
        """Get list of idioms that haven't been processed yet."""
        pending = [
            idiom for idiom in self._idioms
            if idiom.idiom_id not in self._completed_ids
        ]
        print(f"Pending idioms: {len(pending)} / {len(self._idioms)}")
        return pending
    
    def add_result(self, result: ImageabilityResult) -> None:
        """Add a completed result."""
        self._results[result.idiom_id] = result
        self._completed_ids.add(result.idiom_id)
    
    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        # Ensure output directory exists
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "last_updated": datetime.now().isoformat(),
            "total_completed": len(self._completed_ids),
            "results": [
                result.to_checkpoint_dict()
                for result in self._results.values()
            ],
        }
        
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        print(f"Checkpoint saved: {len(self._completed_ids)} items")
    
    def save_final_results(self) -> None:
        """Save final results in output format."""
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort by idiom_id for consistent output
        sorted_results = sorted(
            self._results.values(),
            key=lambda r: r.idiom_id
        )
        
        output_data = [result.to_output_dict() for result in sorted_results]
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Final results saved: {len(output_data)} items -> {self.output_file}")
    
    @property
    def prompt(self) -> str:
        return self._prompt
    
    @property
    def total_idioms(self) -> int:
        return len(self._idioms)
    
    @property
    def completed_count(self) -> int:
        return len(self._completed_ids)
