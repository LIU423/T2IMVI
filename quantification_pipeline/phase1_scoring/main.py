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
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from .config import ScoringConfig, get_available_models
from .evaluator import ScoringEvaluator
from .utils.data_handler import DataHandler


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
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to run model on (default: auto). Use 'auto' for multi-GPU offload.",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["float16", "float32", "bfloat16","auto"],
        help="Model data type (default: auto)",
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
        help="Number of idioms to process in test mode (default: 1)",
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

    # Throughput / resource utilization
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help=(
            "Number of GPU worker processes. "
            "Default: auto-enable for qwen3-vl-2b when multi-GPU is available."
        ),
    )

    parser.add_argument(
        "--worker-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--worker-id",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    
    return parser.parse_args()


def _split_evenly(items: List[int], n_parts: int) -> List[List[int]]:
    """Split items into n_parts contiguous chunks with balanced sizes."""
    if n_parts <= 1:
        return [items]
    n_parts = min(n_parts, len(items))
    base, extra = divmod(len(items), n_parts)
    chunks: List[List[int]] = []
    start = 0
    for i in range(n_parts):
        size = base + (1 if i < extra else 0)
        end = start + size
        chunks.append(items[start:end])
        start = end
    return [c for c in chunks if c]


def _get_target_idiom_ids(config: ScoringConfig) -> List[int]:
    """Resolve idiom IDs that should be processed for this run."""
    if config.idiom_ids:
        idiom_ids = list(config.idiom_ids)
    else:
        data_handler = DataHandler(
            input_images_dir=config.input_images_dir,
            extraction_output_dir=config.extraction_output_dir,
            figurative_prompt_file=config.figurative_prompt_file,
            literal_prompt_file=config.literal_prompt_file,
            output_dir=config.get_output_dir(),
        )
        idiom_ids = data_handler.get_available_idiom_ids()

    if config.test_mode:
        idiom_ids = idiom_ids[: config.test_n_idioms]
    return idiom_ids


def _detect_gpu_count(logger: logging.Logger) -> int:
    """Best-effort CUDA device count detection."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception as exc:
        logger.warning(f"Failed to detect CUDA devices: {exc}")
        return 0


def _should_enable_parallel(args: argparse.Namespace, config: ScoringConfig, gpu_count: int) -> bool:
    """Decide whether to launch multiple worker processes."""
    if args.worker_mode:
        return False
    if args.device == "cpu":
        return False
    if gpu_count < 2:
        return False
    if args.parallel_workers is not None:
        return args.parallel_workers > 1
    # Auto mode: only enable by default for the lightweight model.
    return config.model_name == "qwen3-vl-2b"


def _run_parallel_workers(
    args: argparse.Namespace,
    config: ScoringConfig,
    logger: logging.Logger,
    idiom_ids: List[int],
    gpu_count: int,
) -> int:
    """Launch worker processes pinned to individual GPUs."""
    requested_workers = args.parallel_workers if args.parallel_workers is not None else gpu_count
    n_workers = max(1, min(requested_workers, gpu_count, len(idiom_ids)))
    if n_workers <= 1:
        return -1

    chunks = _split_evenly(idiom_ids, n_workers)
    if len(chunks) <= 1:
        return -1

    logger.info(
        "Parallel mode enabled: %s workers across %s GPUs (idioms=%s)",
        len(chunks),
        gpu_count,
        len(idiom_ids),
    )

    processes = []
    for worker_idx, worker_idioms in enumerate(chunks):
        gpu_id = worker_idx % gpu_count
        cmd = [
            sys.executable,
            "-m",
            "quantification_pipeline.phase1_scoring.main",
            "--worker-mode",
            "--worker-id",
            str(worker_idx),
            "--model",
            args.model,
            "--device",
            "cuda",
            "--dtype",
            args.dtype,
            "--save-interval",
            str(args.save_interval),
            "--idiom-ids",
            *[str(x) for x in worker_idioms],
        ]

        if args.no_resume:
            cmd.append("--no-resume")
        if args.test:
            cmd.extend(["--test", "--test-n-images", str(args.test_n_images)])
        if args.verbose:
            cmd.append("--verbose")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PHASE1_CHECKPOINT_SUFFIX"] = f"worker_{worker_idx}"

        logger.info(
            "Launching worker %s on GPU %s with %s idiom(s)",
            worker_idx,
            gpu_id,
            len(worker_idioms),
        )
        processes.append((worker_idx, subprocess.Popen(cmd, env=env)))

    failed = False
    for worker_idx, proc in processes:
        rc = proc.wait()
        if rc != 0:
            failed = True
            logger.error(f"Worker {worker_idx} failed with exit code {rc}")

    return 1 if failed else 0


def main() -> int:
    """Main entry point."""
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

    gpu_count = _detect_gpu_count(logger)
    idiom_ids = _get_target_idiom_ids(config)
    if idiom_ids:
        logger.info(f"Resolved idiom count: {len(idiom_ids)}")

    if _should_enable_parallel(args, config, gpu_count) and idiom_ids:
        parallel_rc = _run_parallel_workers(args, config, logger, idiom_ids, gpu_count)
        if parallel_rc >= 0:
            return parallel_rc

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
