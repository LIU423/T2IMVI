"""
Phase 1 Extraction Pipeline

Main entry point for idiom element extraction.
Supports both literal and figurative extraction tracks.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

# Add project root to path for centralized config import
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import centralized path configuration
from project_config import (
    UNIQUE_IDIOMS_FILE,
    PHASE1_LITERAL_EXTRACTION_PROMPT,
    PHASE1_FIGURATIVE_EXTRACTION_PROMPT,
    OUTPUT_PHASE1_EXTRACTION_OUTPUT_DIR,
    OUTPUT_PHASE1_EXTRACTION_CHECKPOINTS_DIR,
)

# Add parent directory to path for local imports
sys.path.insert(0, str(_THIS_DIR))

from models import QwenModel, GeminiModel, ModelConfig
from extractors import LiteralExtractor, FigurativeExtractor
from utils import CheckpointManager
from utils.data_loader import load_idioms, load_prompt_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("phase1_extraction.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# Default paths (derived from centralized project_config)
DEFAULT_PATHS = {
    "idioms": UNIQUE_IDIOMS_FILE,
    "literal_prompt": PHASE1_LITERAL_EXTRACTION_PROMPT,
    "figurative_prompt": PHASE1_FIGURATIVE_EXTRACTION_PROMPT,
    "output_dir": OUTPUT_PHASE1_EXTRACTION_OUTPUT_DIR,
    "checkpoint_dir": OUTPUT_PHASE1_EXTRACTION_CHECKPOINTS_DIR,
}


class ExtractionPipeline:
    """
    Main pipeline for idiom element extraction.
    
    Supports:
    - Literal and figurative extraction modes
    - Checkpoint-based resume capability
    - Configurable model backend (local or API)
    - Progress tracking with tqdm
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        idioms_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        model_type: str = "local",
        api_key: Optional[str] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            model_path: HuggingFace model path or API model name
            device: Device for model ("auto", "cuda", "cpu") - only for local models
            idioms_path: Path to idioms JSON file
            output_dir: Base output directory
            checkpoint_dir: Directory for checkpoint files
            model_type: "local" for HuggingFace models, "gemini" for Gemini API
            api_key: API key for Gemini (required when model_type="gemini")
        """
        self.idioms_path = idioms_path or DEFAULT_PATHS["idioms"]
        self.output_dir = output_dir or DEFAULT_PATHS["output_dir"]
        self.checkpoint_dir = checkpoint_dir or DEFAULT_PATHS["checkpoint_dir"]
        
        # Initialize model based on type
        logger.info(f"Initializing model: {model_path} (type: {model_type})")
        
        if model_type == "gemini":
            if not api_key:
                raise ValueError("API key is required for Gemini model")
            
            config = ModelConfig(
                model_name=model_path,
                model_path=model_path,
                max_new_tokens=2048,
                temperature=0.7,
                max_retries=3,
                extra_kwargs={"api_key": api_key},
            )
            self.model = GeminiModel(config)
        else:
            # Default: local model (QwenModel)
            config = ModelConfig(
                model_name=model_path.split("/")[-1] if "/" in model_path else model_path,
                model_path=model_path,
                device=device,
                max_new_tokens=2048,
                temperature=0.7,
                max_retries=3,
            )
            self.model = QwenModel(config)
        
        # Load idioms
        self.idioms = load_idioms(self.idioms_path)
        logger.info(f"Loaded {len(self.idioms)} idioms")
        
        # Initialize extractors (lazy - created on demand)
        self._literal_extractor = None
        self._figurative_extractor = None
    
    @property
    def literal_extractor(self) -> LiteralExtractor:
        """Get or create literal extractor."""
        if self._literal_extractor is None:
            prompt_template = load_prompt_template(DEFAULT_PATHS["literal_prompt"])
            self._literal_extractor = LiteralExtractor(
                model=self.model,
                prompt_template=prompt_template,
                output_dir=self.output_dir,
            )
        return self._literal_extractor
    
    @property
    def figurative_extractor(self) -> FigurativeExtractor:
        """Get or create figurative extractor."""
        if self._figurative_extractor is None:
            prompt_template = load_prompt_template(DEFAULT_PATHS["figurative_prompt"])
            self._figurative_extractor = FigurativeExtractor(
                model=self.model,
                prompt_template=prompt_template,
                output_dir=self.output_dir,
            )
        return self._figurative_extractor
    
    def run_literal_extraction(
        self,
        limit: Optional[int] = None,
        idiom_ids: Optional[List[int]] = None,
        reset_checkpoint: bool = False,
    ) -> None:
        """
        Run literal extraction on idioms.
        
        Args:
            limit: Max number of idioms to process (None = all)
            idiom_ids: Specific idiom IDs to process (None = all)
            reset_checkpoint: If True, reset checkpoint and start fresh
        """
        logger.info("Starting literal extraction...")
        
        # Setup checkpoint
        checkpoint = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            track_type="literal",
            total_count=len(self.idioms),
        )
        
        if reset_checkpoint:
            checkpoint.reset()
        
        # Filter idioms
        idioms_to_process = self._filter_idioms(
            idioms=self.idioms,
            checkpoint=checkpoint,
            limit=limit,
            idiom_ids=idiom_ids,
        )
        
        logger.info(f"Processing {len(idioms_to_process)} idioms")
        
        # Process with progress bar
        for idiom_data in tqdm(idioms_to_process, desc="Literal Extraction"):
            idiom_id = idiom_data["idiom_id"]
            
            try:
                self.literal_extractor.extract_and_save(idiom_data)
                checkpoint.mark_processed(idiom_id)
                
            except Exception as e:
                logger.error(f"Failed to process idiom {idiom_id}: {e}")
                checkpoint.mark_failed(idiom_id, str(e))
        
        # Report progress
        progress = checkpoint.get_progress()
        logger.info(
            f"Literal extraction complete. "
            f"Processed: {progress['processed']}/{progress['total']} "
            f"({progress['progress_pct']:.1f}%), "
            f"Failed: {progress['failed']}"
        )
    
    def run_figurative_extraction(
        self,
        limit: Optional[int] = None,
        idiom_ids: Optional[List[int]] = None,
        reset_checkpoint: bool = False,
    ) -> None:
        """
        Run figurative extraction on idioms.
        
        Args:
            limit: Max number of idioms to process (None = all)
            idiom_ids: Specific idiom IDs to process (None = all)
            reset_checkpoint: If True, reset checkpoint and start fresh
        """
        logger.info("Starting figurative extraction...")
        
        # Setup checkpoint
        checkpoint = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            track_type="figurative",
            total_count=len(self.idioms),
        )
        
        if reset_checkpoint:
            checkpoint.reset()
        
        # Filter idioms
        idioms_to_process = self._filter_idioms(
            idioms=self.idioms,
            checkpoint=checkpoint,
            limit=limit,
            idiom_ids=idiom_ids,
        )
        
        logger.info(f"Processing {len(idioms_to_process)} idioms")
        
        # Process with progress bar
        for idiom_data in tqdm(idioms_to_process, desc="Figurative Extraction"):
            idiom_id = idiom_data["idiom_id"]
            
            try:
                self.figurative_extractor.extract_and_save(idiom_data)
                checkpoint.mark_processed(idiom_id)
                
            except Exception as e:
                logger.error(f"Failed to process idiom {idiom_id}: {e}")
                checkpoint.mark_failed(idiom_id, str(e))
        
        # Report progress
        progress = checkpoint.get_progress()
        logger.info(
            f"Figurative extraction complete. "
            f"Processed: {progress['processed']}/{progress['total']} "
            f"({progress['progress_pct']:.1f}%), "
            f"Failed: {progress['failed']}"
        )
    
    def run_both(
        self,
        limit: Optional[int] = None,
        idiom_ids: Optional[List[int]] = None,
        reset_checkpoint: bool = False,
    ) -> None:
        """Run both literal and figurative extraction."""
        self.run_literal_extraction(limit, idiom_ids, reset_checkpoint)
        self.run_figurative_extraction(limit, idiom_ids, reset_checkpoint)
    
    def _filter_idioms(
        self,
        idioms: List[dict],
        checkpoint: CheckpointManager,
        limit: Optional[int] = None,
        idiom_ids: Optional[List[int]] = None,
    ) -> List[dict]:
        """Filter idioms based on checkpoint and parameters."""
        result = []
        
        for idiom in idioms:
            idiom_id = idiom["idiom_id"]
            
            # Filter by specific IDs if provided
            if idiom_ids is not None and idiom_id not in idiom_ids:
                continue
            
            # Skip already processed
            if checkpoint.is_processed(idiom_id):
                continue
            
            result.append(idiom)
            
            # Apply limit
            if limit is not None and len(result) >= limit:
                break
        
        return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Idiom Element Extraction Pipeline"
    )
    
    parser.add_argument(
        "--mode",
        choices=["literal", "figurative", "both"],
        default="both",
        help="Extraction mode (default: both)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model path (HuggingFace) or model name (API). Default: Qwen/Qwen3-0.6B"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["local", "gemini"],
        default="local",
        help="Model type: 'local' for HuggingFace, 'gemini' for Gemini API (default: local)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for Gemini (required when --model-type=gemini)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for local model: auto, cuda, cpu (default: auto)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of idioms to process (default: all)"
    )
    
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific idiom IDs to process"
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start fresh"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 1 idiom"
    )
    
    args = parser.parse_args()
    
    # Test mode override
    if args.test:
        args.limit = 1
        logger.info("Test mode enabled: processing 1 idiom")
    
    # Initialize pipeline
    pipeline = ExtractionPipeline(
        model_path=args.model,
        device=args.device,
        model_type=args.model_type,
        api_key=args.api_key,
    )
    
    # Run extraction
    if args.mode == "literal":
        pipeline.run_literal_extraction(
            limit=args.limit,
            idiom_ids=args.ids,
            reset_checkpoint=args.reset,
        )
    elif args.mode == "figurative":
        pipeline.run_figurative_extraction(
            limit=args.limit,
            idiom_ids=args.ids,
            reset_checkpoint=args.reset,
        )
    else:
        pipeline.run_both(
            limit=args.limit,
            idiom_ids=args.ids,
            reset_checkpoint=args.reset,
        )


if __name__ == "__main__":
    main()
