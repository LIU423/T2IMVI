"""
CLI entry point for Phase 2 AEA Calculation Pipeline.

Usage:
    # Full run on all idioms
    python -m quantification_pipeline.phase2_calculation.main
    
    # Specific idioms
    python -m quantification_pipeline.phase2_calculation.main --idiom-ids 1 2 3
    
    # Test mode (limited images)
    python -m quantification_pipeline.phase2_calculation.main --idiom-ids 1 --test --test-n-images 2
    
    # Different model
    python -m quantification_pipeline.phase2_calculation.main --model qwen3-vl-2b
    
    # No checkpoint resume
    python -m quantification_pipeline.phase2_calculation.main --no-resume
"""

import argparse
import logging
import os
import sys
import traceback
from typing import List, Optional

from .config import (
    AEAConfig,
    MODEL_REGISTRY,
    DATASET_NAME,
    DEFAULT_MODEL_KEY,
    setup_logging,
)
from .evaluator import AEAEvaluator


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2 AEA Calculation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all idioms
  python -m quantification_pipeline.phase2_calculation.main
  
  # Run on specific idioms
  python -m quantification_pipeline.phase2_calculation.main --idiom-ids 1 2 3
  
  # Test mode with 1 image per idiom
  python -m quantification_pipeline.phase2_calculation.main --idiom-ids 1 --test --test-n-images 1
  
  # Use different model
  python -m quantification_pipeline.phase2_calculation.main --model qwen3-vl-2b
        """,
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_KEY,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"VLM model to use (default: {DEFAULT_MODEL_KEY})",
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
        help=f"Dataset name (default: {DATASET_NAME})",
    )
    
    # Idiom selection
    parser.add_argument(
        "--idiom-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific idiom IDs to process (default: all)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to run model on (default: auto)",
    )
    
    # Checkpoint
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring existing checkpoint",
    )
    
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N images (default: 10)",
    )
    
    # Test mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode (limit processing)",
    )
    
    parser.add_argument(
        "--test-n-images",
        type=int,
        default=1,
        help="Max images per idiom in test mode (default: 1)",
    )
    
    # Batch size
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)",
    )
    
    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Enable HF download progress/logging and apply conservative timeouts/cache.
    # Only set defaults if user hasn't configured them already.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_TIMEOUT", "60")
    os.environ.setdefault("HF_HOME", "/tmp/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf")

    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    # Log configuration
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Resume from checkpoint: {not args.no_resume}")
    
    if args.idiom_ids:
        logger.info(f"Processing idiom IDs: {args.idiom_ids}")
    else:
        logger.info("Processing all idioms")
    
    if args.test:
        logger.info(f"TEST MODE: {len(args.idiom_ids) if args.idiom_ids else 'all'} idiom(s), "
                   f"{args.test_n_images} images each")
    
    # Create configuration
    config = AEAConfig(
        model_key=args.model,
        dataset=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        idiom_ids=args.idiom_ids,
        test_mode=args.test,
        test_n_images=args.test_n_images,
    )
    
    # Create and run evaluator
    evaluator = AEAEvaluator(config)
    
    try:
        evaluator.run()
        return 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("Saving checkpoint before exit...")
        if evaluator.checkpoint_manager is not None:
            evaluator.checkpoint_manager.save()
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        print(f"\nError: {e}")
        print("Saving checkpoint before exit...")
        if evaluator.checkpoint_manager is not None:
            evaluator.checkpoint_manager.save()
        return 1


if __name__ == "__main__":
    sys.exit(main())
