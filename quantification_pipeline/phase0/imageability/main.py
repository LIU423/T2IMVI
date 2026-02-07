"""
Main entry point for imageability evaluation.

Usage:
    # Test mode (1 idiom only)
    python -m quantification_pipeline.phase0.imageability.main --test
    
    # Full run
    python -m quantification_pipeline.phase0.imageability.main
    
    # Resume from checkpoint (default behavior)
    python -m quantification_pipeline.phase0.imageability.main
    
    # Start fresh (ignore checkpoint)
    python -m quantification_pipeline.phase0.imageability.main --no-resume
    
    # Use different model
    python -m quantification_pipeline.phase0.imageability.main --model qwen3-0.6b
"""

import argparse
import sys
from pathlib import Path

from .config import EvalConfig, get_default_config
from .evaluator import ImageabilityEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate imageability of idioms using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 1 idiom for verification.",
    )
    
    parser.add_argument(
        "--test-n",
        type=int,
        default=1,
        help="Number of items to process in test mode (default: 1).",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-0.6b",
        help="Model to use (default: qwen3-0.6b).",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda).",
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint.",
    )
    
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N items (default: 10).",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Build config from args
    config = EvalConfig(
        model_name=args.model,
        device=args.device,
        resume_from_checkpoint=not args.no_resume,
        save_interval=args.save_interval,
        test_mode=args.test,
        test_n_items=args.test_n,
    )
    
    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Resume: {config.resume_from_checkpoint}")
    print(f"  Test mode: {config.test_mode}")
    if config.test_mode:
        print(f"  Test items: {config.test_n_items}")
    
    # Run evaluation
    try:
        evaluator = ImageabilityEvaluator(config)
        evaluator.run()
        return 0
    except KeyboardInterrupt:
        print("\nAborted by user.")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
