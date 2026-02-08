"""
Compare average per-image processing time between pipeline and direct baseline.

Pipeline total average time is defined as:
  phase1_scoring avg + phase2_aea avg + phase2_iu avg + phase2_vc_ca_cr avg
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from project_config import (
    OUTPUT_IRFL_DIR,
    RELIABILITY_ANALYSIS_COMPARISON_DIR,
)


MODEL_TO_PIPELINE_DIR = {
    "qwen3-vl-2b": "qwen3_vl_2b_T2IMVI",
    "qwen3-vl-30b-a3b-instruct": "qwen3_vl_30b_a3b_instruct_T2IMVI",
}

MODEL_TO_COMPARISON_DIR = {
    "qwen3-vl-2b": "qwen3_vl_2b",
    "qwen3-vl-30b-a3b-instruct": "qwen3_vl_30b_a3b_instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-image average time: pipeline vs direct baseline",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-30b-a3b-instruct",
        choices=sorted(MODEL_TO_PIPELINE_DIR.keys()),
        help="Model key",
    )
    parser.add_argument(
        "--idiom-id",
        type=int,
        required=True,
        help="One idiom id for direct-baseline timing run",
    )
    parser.add_argument(
        "--run-direct-baseline",
        action="store_true",
        help="Run direct baseline for this idiom before comparison",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for direct baseline run",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="dtype for direct baseline run",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_direct_baseline(model: str, idiom_id: int, device: str, dtype: str) -> None:
    cmd = [
        sys.executable,
        "reliability_analysis/comparison/direct_vlm_scoring_baseline.py",
        "--model",
        model,
        "--idiom-ids",
        str(idiom_id),
        "--device",
        device,
        "--dtype",
        dtype,
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)


def _get_pipeline_avg_seconds_per_image(pipeline_dir: Path) -> Dict[str, float]:
    phase1 = _read_json(pipeline_dir / "timing_phase1_scoring.json") or {}
    p2_aea = _read_json(pipeline_dir / "timing_phase2_aea.json") or {}
    p2_iu = _read_json(pipeline_dir / "timing_phase2_iu.json") or {}
    p2_vc = _read_json(pipeline_dir / "timing_phase2_vc_ca_cr.json") or {}

    phase1_avg = float(phase1.get("avg_seconds_per_success_image", 0.0))
    aea_avg = float(p2_aea.get("avg_seconds_per_success_image", 0.0))
    iu_avg = float(p2_iu.get("avg_seconds_per_success_image", 0.0))
    vc_avg = float(p2_vc.get("avg_seconds_per_processed_image", 0.0))

    return {
        "phase1_scoring_avg_seconds_per_image": phase1_avg,
        "phase2_aea_avg_seconds_per_image": aea_avg,
        "phase2_iu_avg_seconds_per_image": iu_avg,
        "phase2_vc_ca_cr_avg_seconds_per_image": vc_avg,
        "pipeline_total_avg_seconds_per_image": phase1_avg + aea_avg + iu_avg + vc_avg,
    }


def main() -> int:
    args = parse_args()

    if args.run_direct_baseline:
        _run_direct_baseline(
            model=args.model,
            idiom_id=args.idiom_id,
            device=args.device,
            dtype=args.dtype,
        )

    pipeline_dir = OUTPUT_IRFL_DIR / MODEL_TO_PIPELINE_DIR[args.model]
    direct_dir = (
        RELIABILITY_ANALYSIS_COMPARISON_DIR
        / "direct_vlm_scoring_baseline"
        / MODEL_TO_COMPARISON_DIR[args.model]
    )

    pipeline_time = _get_pipeline_avg_seconds_per_image(pipeline_dir)
    direct_summary = _read_json(direct_dir / "timing_direct_vlm_baseline.json") or {}
    per_idiom = direct_summary.get("per_idiom", {})
    idiom_info = per_idiom.get(str(args.idiom_id), {})

    direct_avg_one_idiom = float(idiom_info.get("avg_seconds_per_image", 0.0))
    direct_avg_overall = float(direct_summary.get("avg_seconds_per_image", 0.0))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "idiom_id": args.idiom_id,
        "pipeline_timing": pipeline_time,
        "direct_baseline_timing": {
            "avg_seconds_per_image_for_target_idiom": direct_avg_one_idiom,
            "avg_seconds_per_image_overall_last_run": direct_avg_overall,
            "processed_images_overall_last_run": int(direct_summary.get("processed_images", 0)),
        },
        "comparison": {
            "pipeline_total_minus_direct_target_idiom_avg_seconds": (
                pipeline_time["pipeline_total_avg_seconds_per_image"] - direct_avg_one_idiom
            ),
            "direct_over_pipeline_ratio_target_idiom": (
                (direct_avg_one_idiom / pipeline_time["pipeline_total_avg_seconds_per_image"])
                if pipeline_time["pipeline_total_avg_seconds_per_image"] > 0
                else 0.0
            ),
        },
        "notes": [
            "Pipeline total average is the sum of per-phase average image times.",
            "Missing timing files are treated as 0.0.",
        ],
    }

    out_dir = (
        RELIABILITY_ANALYSIS_COMPARISON_DIR
        / "timing_experiment"
        / MODEL_TO_COMPARISON_DIR[args.model]
        / f"idiom_{args.idiom_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "time_per_image_comparison_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
