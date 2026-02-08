"""CLI entry for quadrant-based reliability analysis."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure reliability_analysis package-local imports resolve when run as script.
_THIS_DIR = Path(__file__).resolve().parent
_RA_ROOT = _THIS_DIR.parent
if str(_RA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RA_ROOT))

from config import DEFAULT_EXPERIMENT_CONFIG, list_available_models
from quadrant_analysis.analyzer import print_quadrant_summary, run_quadrant_analysis
from quadrant_analysis.config import DEFAULT_SCORE_FIELDS, QuadrantConfig


def _parse_idiom_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quadrant analysis by idiom-level transparency/imageability. "
            "Computes RBO (standard/tie-aware), ICC, Pearson r, MAE for "
            "S_pot, S_fid, Entity+Action Avg, and (Fig+Lit)/2."
        )
    )
    parser.add_argument("--model", type=str, default="qwen3_vl_2b_T2IMVI", help="Model key")
    parser.add_argument(
        "--idiom-ids",
        type=str,
        default=None,
        help="Comma-separated idiom IDs, e.g. 1,2,3. Omit to process all idioms.",
    )
    parser.add_argument("--transparency-threshold", type=float, default=0.5)
    parser.add_argument("--imageability-threshold", type=float, default=0.5)
    parser.add_argument("--rbo-p", type=float, default=DEFAULT_EXPERIMENT_CONFIG.rbo_p)
    parser.add_argument(
        "--score-fields",
        type=str,
        default=",".join(DEFAULT_SCORE_FIELDS),
        help="Comma-separated score fields. Default: s_pot,s_fid,entity_action_avg,fig_lit_avg",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON file")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.model not in list_available_models():
        parser.error(f"Unknown model: {args.model}. Available: {list_available_models()}")

    idiom_ids = _parse_idiom_ids(args.idiom_ids)
    score_fields = [s.strip() for s in args.score_fields.split(",") if s.strip()]

    cfg = QuadrantConfig(
        transparency_threshold=args.transparency_threshold,
        imageability_threshold=args.imageability_threshold,
        rbo_p=args.rbo_p,
        score_fields=score_fields,
    )

    results = run_quadrant_analysis(
        model_key=args.model,
        idiom_ids=idiom_ids,
        config=cfg,
        save_results=not args.no_save,
    )

    print_quadrant_summary(results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
