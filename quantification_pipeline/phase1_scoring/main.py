"""
Main entry point for Phase 1 Scoring: Visual Element Verification Pipeline.

This script runs the figurative and literal element verification process
using Vision-Language Models to score element presence in images.

Usage:
    # Run on all idioms with default settings
    python -m quantification_pipeline.phase1_scoring.main
    
    # Run on specific idiom(s)
    python -m quantification_pipeline.phase1_scoring.main --idiom-ids 3
    python -m quantification_pipeline.phase1_scoring.main --idiom-ids 1 2 3 4 5
    
    # Test mode (process limited data)
    python -m quantification_pipeline.phase1_scoring.main --test
    
    # Use different model
    python -m quantification_pipeline.phase1_scoring.main --model qwen3-vl-2b
    
    # Resume from checkpoint (default behavior)
    python -m quantification_pipeline.phase1_scoring.main --resume
    
    # Start fresh (ignore checkpoint)
    python -m quantification_pipeline.phase1_scoring.main --no-resume
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import ScoringConfig, get_available_models
from .evaluator import ScoringEvaluator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Reduce noise from external libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Scoring: Visual Element Verification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with idiom 3 only
  python -m quantification_pipeline.phase1_scoring.main --idiom-ids 3 --test
  
  # Run all idioms with checkpoint resume
  python -m quantification_pipeline.phase1_scoring.main
  
  # Process specific idioms
  python -m quantification_pipeline.phase1_scoring.main --idiom-ids 1 2 3
        """,
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3-vl-2b",
        choices=get_available_models(),
        help="VLM model to use for verification (default: qwen3-vl-2b)",
    )
    
    # Idiom selection
    parser.add_argument(
        "--idiom-ids", "-i",
        type=int,
        nargs="+",
        default=None,
        help="Specific idiom IDs to process (default: all available)",
    )
    
    # Device settings
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device to run model on (default: cuda)",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model data type (default: float16)",
    )
    
    # Checkpoint/resume
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)",
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )
    
    # Test mode
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test mode: process limited data",
    )
    
    parser.add_argument(
        "--test-n-idioms",
        type=int,
        default=1,
        help="Number of idioms to process in test mode (default: 11)",
    )
    
    parser.add_argument(
        "--test-n-images",
        type=int,
        default=20,
        help="Number of images per idiom in test mode (default: 20)",
    )
    
    # Save interval
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N images (default: 10)",
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    # Build configuration
    config = ScoringConfig(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        save_interval=args.save_interval,
        resume_from_checkpoint=not args.no_resume,
        test_mode=args.test,
        test_n_idioms=args.test_n_idioms,
        test_n_images=args.test_n_images,
        idiom_ids=args.idiom_ids,
    )
    
    # Log configuration
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Resume from checkpoint: {config.resume_from_checkpoint}")
    if config.idiom_ids:
        logger.info(f"Processing idiom IDs: {config.idiom_ids}")
    if config.test_mode:
        logger.info(f"TEST MODE: {config.test_n_idioms} idiom(s), {config.test_n_images} images each")
    
    # Run evaluator
    try:
        evaluator = ScoringEvaluator(config)
        evaluator.run()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
