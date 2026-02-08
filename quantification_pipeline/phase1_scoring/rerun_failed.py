"""
Rerun only failed phase1_scoring idiom-image items.

Reads failed-items JSONL logs and invokes phase1_scoring.main in target-only mode.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from .config import ScoringConfig, get_available_models
from .utils.failed_items import extract_target_image_map, load_failed_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun failed phase1_scoring items from failed-items logs",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen3-vl-2b",
        choices=get_available_models(),
        help="Model name used for failed logs and rerun",
    )
    parser.add_argument(
        "--failed-files",
        type=str,
        nargs="*",
        default=None,
        help="Optional failed_items*.jsonl files. Default: auto-discover under data/phase1_scoring/failed_items/<model>/",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device for rerun (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["float16", "float32", "bfloat16", "auto"],
        help="Model dtype for rerun",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Checkpoint save interval",
    )
    parser.add_argument(
        "--oom-max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per image for OOM errors",
    )
    parser.add_argument(
        "--oom-retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Backoff seconds between OOM retries",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and start fresh",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop when one rerun target fails",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def _discover_failed_files(model_name: str) -> List[Path]:
    cfg = ScoringConfig(model_name=model_name)
    model_dir = cfg.failed_items_dir / model_name.replace("-", "_")
    return sorted(model_dir.glob("failed_items*.jsonl"))


def _write_targets_jsonl(target_map: dict, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for idiom_id in sorted(target_map.keys()):
            for image_num in sorted(target_map[idiom_id]):
                payload = {"idiom_id": idiom_id, "image_num": image_num}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_file


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    if args.failed_files:
        failed_files = [Path(x) for x in args.failed_files]
    else:
        failed_files = _discover_failed_files(args.model)

    if not failed_files:
        logger.error("No failed-items files found for model=%s", args.model)
        return 1

    records = load_failed_items(failed_files)
    if not records:
        logger.error("No failed records found in: %s", [str(x) for x in failed_files])
        return 1

    target_map = extract_target_image_map(records)
    n_targets = sum(len(v) for v in target_map.values())
    if n_targets == 0:
        logger.error("No valid rerun targets parsed from failed records.")
        return 1

    cfg = ScoringConfig(model_name=args.model)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_file = cfg.failed_items_dir / args.model.replace("-", "_") / f"rerun_targets_{stamp}.jsonl"
    _write_targets_jsonl(target_map, target_file)

    logger.info("Failed files: %s", [str(x) for x in failed_files])
    logger.info("Rerun targets: %s idiom(s), %s image(s)", len(target_map), n_targets)
    logger.info("Target file: %s", target_file)

    cmd = [
        sys.executable,
        "-m",
        "quantification_pipeline.phase1_scoring.main",
        "--model",
        args.model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--save-interval",
        str(args.save_interval),
        "--oom-max-attempts",
        str(args.oom_max_attempts),
        "--oom-retry-backoff-seconds",
        str(args.oom_retry_backoff_seconds),
        "--target-items-file",
        str(target_file),
    ]
    if args.no_resume:
        cmd.append("--no-resume")
    if args.fail_fast:
        cmd.append("--fail-fast")
    if args.verbose:
        cmd.append("--verbose")

    logger.info("Executing: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
