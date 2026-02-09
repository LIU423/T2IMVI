"""Quadrant-style comparison between strategy outputs and T2IMVI outputs.

This script reuses the numerical ranking metrics used by quadrant analysis,
but reports side-by-side aggregates for:
1) data/reliability_analysis/comparison/<strategy>/<model_dir>
2) data/output/IRFL/<model_dir>_T2IMVI
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure reliability_analysis package-local imports resolve when run as script.
_THIS_DIR = Path(__file__).resolve().parent
_RA_ROOT = _THIS_DIR.parent
if str(_RA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RA_ROOT))

from config import DEFAULT_EXPERIMENT_CONFIG, DEFAULT_SCORING_CONFIG, MODEL_CONFIGS, ModelConfig
from data_loader import discover_idioms, load_combined_data_for_idiom
from experiment1_ranking import compare_rankings_numerical
from project_config import OUTPUT_IRFL_DIR, RELIABILITY_ANALYSIS_COMPARISON_DIR, RELIABILITY_ANALYSIS_RESULTS_DIR
from quadrant_analysis.config import DEFAULT_SCORE_FIELDS, SCORE_FIELD_LABELS


logger = logging.getLogger(__name__)

THRESHOLD_PRESETS: Dict[str, Tuple[float, float]] = {
    "default": (0.5, 0.5),
    "strict": (0.6, 0.6),
    "lenient": (0.4, 0.4),
}
RBO_P_PRESETS: Dict[str, float] = {
    "default": DEFAULT_EXPERIMENT_CONFIG.rbo_p,
    "top_heavy": 0.95,
    "balanced": 0.9,
    "flatter": 0.8,
}


def _mean_or_zero(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _assign_quadrant(
    transparency: float,
    imageability: float,
    transparency_threshold: float,
    imageability_threshold: float,
) -> str:
    trans_high = transparency >= transparency_threshold
    img_high = imageability >= imageability_threshold
    if trans_high and img_high:
        return "Q1_highT_highI"
    if trans_high and not img_high:
        return "Q2_highT_lowI"
    if not trans_high and img_high:
        return "Q3_lowT_highI"
    return "Q4_lowT_lowI"


def _parse_idiom_ids(values: Optional[List[str]]) -> Optional[List[int]]:
    if not values:
        return None
    parsed: List[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(int(token))
    return sorted(set(parsed)) if parsed else None


def _resolve_thresholds(
    threshold_preset: str,
    transparency_threshold: Optional[float],
    imageability_threshold: Optional[float],
) -> Tuple[float, float]:
    if threshold_preset not in THRESHOLD_PRESETS:
        raise ValueError(
            f"Unknown threshold preset: {threshold_preset}. "
            f"Available: {sorted(THRESHOLD_PRESETS.keys())}"
        )
    preset_t, preset_i = THRESHOLD_PRESETS[threshold_preset]
    final_t = preset_t if transparency_threshold is None else transparency_threshold
    final_i = preset_i if imageability_threshold is None else imageability_threshold
    return final_t, final_i


def _resolve_rbo_p(rbo_preset: str, rbo_p: Optional[float]) -> float:
    if rbo_preset not in RBO_P_PRESETS:
        raise ValueError(
            f"Unknown rbo preset: {rbo_preset}. "
            f"Available: {sorted(RBO_P_PRESETS.keys())}"
        )
    final_p = RBO_P_PRESETS[rbo_preset] if rbo_p is None else rbo_p
    if not (0.0 < final_p < 1.0):
        raise ValueError(f"Invalid rbo p: {final_p}. Must be in (0, 1).")
    return final_p


def _discover_model_idioms(model_path: Path) -> List[int]:
    idioms: List[int] = []
    if not model_path.exists():
        return idioms
    for path in model_path.iterdir():
        if not path.is_dir() or not path.name.startswith("idiom_"):
            continue
        tail = path.name.replace("idiom_", "", 1)
        try:
            idioms.append(int(tail))
        except ValueError:
            continue
    return sorted(set(idioms))


def _register_temp_model(model_key: str, model_name: str, model_path: Path) -> None:
    MODEL_CONFIGS[model_key] = ModelConfig(
        name=model_name,
        strategy_id=model_key,
        description=f"Temporary model config for {model_path}",
        score_field="figurative_score",
        output_path_override=model_path,
    )


def _compute_idiom_ti_means(model_key: str, idiom_id: int) -> Optional[Tuple[float, float, int]]:
    image_data = load_combined_data_for_idiom(idiom_id, [model_key])
    valid = [img for img in image_data if model_key in img.model_outputs]
    if not valid:
        return None
    transparencies = [img.model_outputs[model_key].transparency for img in valid]
    imageabilities = [img.model_outputs[model_key].imageability for img in valid]
    return _mean_or_zero(transparencies), _mean_or_zero(imageabilities), len(valid)


@dataclass
class ScoreAggregate:
    score_field: str
    score_label: str
    n_idioms: int
    mean_rbo_standard: float
    mean_ta_rbo: float
    mean_tta_rbo: float


@dataclass
class QuadrantModelSummary:
    quadrant: str
    model_role: str
    model_key: str
    n_idioms: int
    idiom_ids: List[int]
    score_field_results: List[ScoreAggregate]


@dataclass
class QuadrantDelta:
    quadrant: str
    score_field: str
    score_label: str
    n_overlap_idioms: int
    delta_rbo_standard: float
    delta_ta_rbo: float
    delta_tta_rbo: float


@dataclass
class ScoreSideBySide:
    quadrant: str
    comparison_score_field: str
    comparison_score_label: str
    t2imvi_score_field: str
    t2imvi_score_label: str
    comparison_n_idioms: int
    t2imvi_n_idioms: int
    comparison_rbo_standard: float
    t2imvi_rbo_standard: float
    comparison_ta_rbo: float
    t2imvi_ta_rbo: float
    comparison_tta_rbo: float
    t2imvi_tta_rbo: float


@dataclass
class ComparisonResults:
    timestamp: str
    strategy: str
    model_dir: str
    comparison_model_path: str
    t2imvi_model_path: str
    quadrant_anchor: str
    thresholds: Dict[str, float]
    low_score_zero_threshold: Optional[int]
    rbo_p: float
    comparison_score_field: str
    t2imvi_score_fields: List[str]
    idiom_quadrants: List[Dict[str, object]]
    model_summaries: List[QuadrantModelSummary]
    side_by_side_scores: List[ScoreSideBySide]
    deltas_comparison_minus_t2imvi: List[QuadrantDelta]


def _aggregate_group(
    model_key: str,
    idiom_ids: List[int],
    score_field: str,
    rbo_p: float,
    low_score_zero_threshold: Optional[int],
) -> ScoreAggregate:
    rbo_std_vals: List[float] = []
    ta_rbo_vals: List[float] = []
    tta_rbo_vals: List[float] = []

    for idiom_id in idiom_ids:
        image_data = load_combined_data_for_idiom(idiom_id, [model_key])
        if not image_data:
            continue
        result = compare_rankings_numerical(
            idiom_id=idiom_id,
            image_data_list=image_data,
            model_key=model_key,
            score_field=score_field,
            rbo_p=rbo_p,
            scoring_config=DEFAULT_SCORING_CONFIG,
            low_score_zero_threshold=low_score_zero_threshold,
        )
        if result is None:
            continue
        rbo_std_vals.append(result.rbo_standard)
        ta_rbo_vals.append(result.ta_rbo)
        tta_rbo_vals.append(result.tta_rbo)

    return ScoreAggregate(
        score_field=score_field,
        score_label=SCORE_FIELD_LABELS.get(score_field, score_field),
        n_idioms=len(rbo_std_vals),
        mean_rbo_standard=_mean_or_zero(rbo_std_vals),
        mean_ta_rbo=_mean_or_zero(ta_rbo_vals),
        mean_tta_rbo=_mean_or_zero(tta_rbo_vals),
    )


def run_comparison(
    strategy: str,
    model_dir: str,
    t2imvi_model_dir: Optional[str],
    idiom_ids: Optional[List[int]],
    transparency_threshold: float,
    imageability_threshold: float,
    low_score_zero_threshold: Optional[int],
    rbo_p: float,
    comparison_score_field: str,
    t2imvi_score_fields: List[str],
    quadrant_anchor: str,
    save_results: bool,
    results_dir: Optional[Path],
) -> ComparisonResults:
    comparison_model_path = RELIABILITY_ANALYSIS_COMPARISON_DIR / strategy / model_dir
    t2imvi_dir = t2imvi_model_dir or f"{model_dir}_T2IMVI"
    t2imvi_model_path = OUTPUT_IRFL_DIR / t2imvi_dir

    if not comparison_model_path.exists():
        raise FileNotFoundError(f"Comparison path not found: {comparison_model_path}")
    if not t2imvi_model_path.exists():
        raise FileNotFoundError(f"T2IMVI path not found: {t2imvi_model_path}")

    comparison_key = f"temp_comparison_{strategy}_{model_dir}".replace("-", "_")
    t2imvi_key = f"temp_t2imvi_{t2imvi_dir}".replace("-", "_")
    _register_temp_model(comparison_key, f"comparison/{strategy}/{model_dir}", comparison_model_path)
    _register_temp_model(t2imvi_key, f"IRFL/{t2imvi_dir}", t2imvi_model_path)

    base_idioms = idiom_ids or discover_idioms()
    comparison_idioms = set(_discover_model_idioms(comparison_model_path))
    t2imvi_idioms = set(_discover_model_idioms(t2imvi_model_path))
    shared_idioms = sorted(set(base_idioms) & comparison_idioms & t2imvi_idioms)

    if not shared_idioms:
        raise ValueError("No shared idioms found between annotation data and both model directories.")

    anchor_key = t2imvi_key if quadrant_anchor == "t2imvi" else comparison_key
    groups: Dict[str, List[int]] = {
        "Q1_highT_highI": [],
        "Q2_highT_lowI": [],
        "Q3_lowT_highI": [],
        "Q4_lowT_lowI": [],
    }
    idiom_quadrants: List[Dict[str, object]] = []

    for idiom_id in shared_idioms:
        ti = _compute_idiom_ti_means(anchor_key, idiom_id)
        if ti is None:
            continue
        transparency, imageability, n_images = ti
        quadrant = _assign_quadrant(
            transparency=transparency,
            imageability=imageability,
            transparency_threshold=transparency_threshold,
            imageability_threshold=imageability_threshold,
        )
        groups[quadrant].append(idiom_id)
        idiom_quadrants.append(
            {
                "idiom_id": idiom_id,
                "quadrant": quadrant,
                "transparency": transparency,
                "imageability": imageability,
                "n_images": n_images,
            }
        )

    model_summaries: List[QuadrantModelSummary] = []
    delta_rows: List[QuadrantDelta] = []
    side_by_side_rows: List[ScoreSideBySide] = []
    quadrants = ["Q1_highT_highI", "Q2_highT_lowI", "Q3_lowT_highI", "Q4_lowT_lowI"]

    for quadrant in quadrants:
        members = groups[quadrant]
        comparison_agg = _aggregate_group(
            comparison_key,
            members,
            comparison_score_field,
            rbo_p,
            low_score_zero_threshold,
        )
        t2imvi_fields = [
            _aggregate_group(
                t2imvi_key,
                members,
                score_field,
                rbo_p,
                low_score_zero_threshold,
            )
            for score_field in t2imvi_score_fields
        ]

        model_summaries.append(
            QuadrantModelSummary(
                quadrant=quadrant,
                model_role="comparison",
                model_key=comparison_key,
                n_idioms=len(members),
                idiom_ids=members,
                score_field_results=[comparison_agg],
            )
        )
        model_summaries.append(
            QuadrantModelSummary(
                quadrant=quadrant,
                model_role="t2imvi",
                model_key=t2imvi_key,
                n_idioms=len(members),
                idiom_ids=members,
                score_field_results=t2imvi_fields,
            )
        )

        by_score_t2imvi = {x.score_field: x for x in t2imvi_fields}
        for t2_row in t2imvi_fields:
            n_overlap = min(comparison_agg.n_idioms, t2_row.n_idioms)
            side_by_side_rows.append(
                ScoreSideBySide(
                    quadrant=quadrant,
                    comparison_score_field=comparison_agg.score_field,
                    comparison_score_label=comparison_agg.score_label,
                    t2imvi_score_field=t2_row.score_field,
                    t2imvi_score_label=t2_row.score_label,
                    comparison_n_idioms=comparison_agg.n_idioms,
                    t2imvi_n_idioms=t2_row.n_idioms,
                    comparison_rbo_standard=comparison_agg.mean_rbo_standard,
                    t2imvi_rbo_standard=t2_row.mean_rbo_standard,
                    comparison_ta_rbo=comparison_agg.mean_ta_rbo,
                    t2imvi_ta_rbo=t2_row.mean_ta_rbo,
                    comparison_tta_rbo=comparison_agg.mean_tta_rbo,
                    t2imvi_tta_rbo=t2_row.mean_tta_rbo,
                )
            )
            delta_rows.append(
                QuadrantDelta(
                    quadrant=quadrant,
                    score_field=t2_row.score_field,
                    score_label=t2_row.score_label,
                    n_overlap_idioms=n_overlap,
                    delta_rbo_standard=comparison_agg.mean_rbo_standard - t2_row.mean_rbo_standard,
                    delta_ta_rbo=comparison_agg.mean_ta_rbo - t2_row.mean_ta_rbo,
                    delta_tta_rbo=comparison_agg.mean_tta_rbo - t2_row.mean_tta_rbo,
                )
            )

    results = ComparisonResults(
        timestamp=datetime.now().isoformat(),
        strategy=strategy,
        model_dir=model_dir,
        comparison_model_path=str(comparison_model_path),
        t2imvi_model_path=str(t2imvi_model_path),
        quadrant_anchor=quadrant_anchor,
        thresholds={
            "transparency": transparency_threshold,
            "imageability": imageability_threshold,
        },
        low_score_zero_threshold=low_score_zero_threshold,
        rbo_p=rbo_p,
        comparison_score_field=comparison_score_field,
        t2imvi_score_fields=t2imvi_score_fields,
        idiom_quadrants=idiom_quadrants,
        model_summaries=model_summaries,
        side_by_side_scores=side_by_side_rows,
        deltas_comparison_minus_t2imvi=delta_rows,
    )

    if save_results:
        out_dir = results_dir or (RELIABILITY_ANALYSIS_RESULTS_DIR / "comparison_quadrant_analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / (
            f"comparison_quadrant_{strategy}_{model_dir}_"
            f"T{transparency_threshold:.2f}_I{imageability_threshold:.2f}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(results), f, indent=2, ensure_ascii=False)
        logger.info("Saved comparison result: %s", out_file)

    return results


def _print_summary(results: ComparisonResults, show_delta: bool = False) -> None:
    print("\n" + "=" * 98)
    print("Quadrant Comparison: comparison strategy vs T2IMVI")
    print("=" * 98)
    print(f"Strategy/model: {results.strategy}/{results.model_dir}")
    print(f"Comparison path: {results.comparison_model_path}")
    print(f"T2IMVI path: {results.t2imvi_model_path}")
    print(
        f"Thresholds: transparency={results.thresholds['transparency']}, "
        f"imageability={results.thresholds['imageability']} | anchor={results.quadrant_anchor}"
    )
    print(f"Comparison score field: {results.comparison_score_field}")
    print(f"T2IMVI score fields: {','.join(results.t2imvi_score_fields)}")
    print(f"Low-score collapse threshold (for tie-aware RBO variant): {results.low_score_zero_threshold}")
    print(f"RBO p: {results.rbo_p}")

    quadrants = ["Q1_highT_highI", "Q2_highT_lowI", "Q3_lowT_highI", "Q4_lowT_lowI"]
    for quadrant in quadrants:
        rows = [x for x in results.side_by_side_scores if x.quadrant == quadrant]
        if not rows:
            continue
        row = rows[0]
        has_data = row.comparison_n_idioms > 0 or row.t2imvi_n_idioms > 0
        if not has_data:
            continue
        print("\n" + "-" * 98)
        print(f"{quadrant}")
        print(
            f"{'T2IMVI score':<18} {'n(c/t)':>8} {'RBO(c/t)':>20} {'TA-RBO(c/t)':>20} "
            f"{'TTA-RBO(c/t)':>20}"
        )
        print("-" * 98)
        for row in rows:
            print(
                f"{row.t2imvi_score_label:<18} "
                f"{row.comparison_n_idioms:>3d}/{row.t2imvi_n_idioms:<3d} "
                f"{row.comparison_rbo_standard:>9.4f}/{row.t2imvi_rbo_standard:<9.4f} "
                f"{row.comparison_ta_rbo:>9.4f}/{row.t2imvi_ta_rbo:<9.4f} "
                f"{row.comparison_tta_rbo:>9.4f}/{row.t2imvi_tta_rbo:<9.4f}"
            )

    if show_delta:
        print("\n" + "=" * 98)
        print("Delta (comparison - T2IMVI)")
        print("=" * 98)
        for delta in results.deltas_comparison_minus_t2imvi:
            print(
                f"{delta.quadrant:>14} | {delta.score_label:<18} | n={delta.n_overlap_idioms:>3d} "
                f"| dRBO={delta.delta_rbo_standard:+.4f} dTA-RBO={delta.delta_ta_rbo:+.4f} "
                f"dTTA-RBO={delta.delta_tta_rbo:+.4f}"
            )


def _write_csv(results: ComparisonResults, csv_output: Path) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "strategy",
                "model_dir",
                "comparison_score_field",
                "t2imvi_score_field",
                "t2imvi_score_label",
                "quadrant",
                "comparison_n_idioms",
                "t2imvi_n_idioms",
                "comparison_rbo_standard",
                "t2imvi_rbo_standard",
                "comparison_ta_rbo",
                "t2imvi_ta_rbo",
                "comparison_tta_rbo",
                "t2imvi_tta_rbo",
                "delta_rbo_standard",
                "delta_ta_rbo",
                "delta_tta_rbo",
            ]
        )
        delta_map = {(d.quadrant, d.score_field): d for d in results.deltas_comparison_minus_t2imvi}
        for row in results.side_by_side_scores:
            delta = delta_map.get((row.quadrant, row.t2imvi_score_field))
            writer.writerow(
                [
                    results.strategy,
                    results.model_dir,
                    row.comparison_score_field,
                    row.t2imvi_score_field,
                    row.t2imvi_score_label,
                    row.quadrant,
                    row.comparison_n_idioms,
                    row.t2imvi_n_idioms,
                    row.comparison_rbo_standard,
                    row.t2imvi_rbo_standard,
                    row.comparison_ta_rbo,
                    row.t2imvi_ta_rbo,
                    row.comparison_tta_rbo,
                    row.t2imvi_tta_rbo,
                    delta.delta_rbo_standard if delta else "",
                    delta.delta_ta_rbo if delta else "",
                    delta.delta_tta_rbo if delta else "",
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quadrant-style comparison between comparison strategy output and <model>_T2IMVI output.",
    )
    parser.add_argument("--strategy", type=str, required=True, help="Comparison strategy directory name")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory name under strategy")
    parser.add_argument(
        "--t2imvi-model-dir",
        type=str,
        default=None,
        help="Model directory under data/output/IRFL. Default: <model-dir>_T2IMVI",
    )
    parser.add_argument(
        "--idiom-ids",
        type=str,
        nargs="+",
        default=None,
        help="Idiom IDs, supports '--idiom-ids 1 2' and '--idiom-ids 1,2'. Default: all shared idioms.",
    )
    parser.add_argument(
        "--threshold-preset",
        type=str,
        default="default",
        choices=sorted(THRESHOLD_PRESETS.keys()),
        help="Threshold preset. You can still override either threshold explicitly.",
    )
    parser.add_argument(
        "--transparency-threshold",
        type=float,
        default=None,
        help="Transparency threshold override. Default comes from --threshold-preset.",
    )
    parser.add_argument(
        "--imageability-threshold",
        type=float,
        default=None,
        help="Imageability threshold override. Default comes from --threshold-preset.",
    )
    parser.add_argument(
        "--low-score-zero-threshold",
        type=int,
        default=10,
        help="Human total scores <= threshold are collapsed to 0 for an extra tie-aware RBO metric.",
    )
    parser.add_argument(
        "--rbo-preset",
        type=str,
        default="default",
        choices=sorted(RBO_P_PRESETS.keys()),
        help="RBO p preset. You can still override with --rbo-p.",
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=None,
        help="RBO p override. Must be in (0,1). Default comes from --rbo-preset.",
    )
    parser.add_argument(
        "--comparison-score-field",
        type=str,
        default="figurative_score",
        help="Score field used on comparison side (direct baseline). Default: figurative_score",
    )
    parser.add_argument(
        "--t2imvi-score-fields",
        type=str,
        default=",".join(DEFAULT_SCORE_FIELDS),
        help="Comma-separated T2IMVI score fields. Default: s_pot,s_fid,entity_action_avg,fig_lit_avg",
    )
    parser.add_argument(
        "--quadrant-anchor",
        type=str,
        choices=["t2imvi", "comparison"],
        default="t2imvi",
        help="Which side provides idiom-level transparency/imageability for quadrant assignment.",
    )
    parser.add_argument("--show-delta", action="store_true", help="Also print delta (comparison - T2IMVI).")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV export path for side-by-side comparison rows.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON result.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    idiom_ids = _parse_idiom_ids(args.idiom_ids)
    transparency_threshold, imageability_threshold = _resolve_thresholds(
        threshold_preset=args.threshold_preset,
        transparency_threshold=args.transparency_threshold,
        imageability_threshold=args.imageability_threshold,
    )
    rbo_p = _resolve_rbo_p(
        rbo_preset=args.rbo_preset,
        rbo_p=args.rbo_p,
    )
    comparison_score_field = args.comparison_score_field.strip()
    t2imvi_score_fields = [x.strip() for x in args.t2imvi_score_fields.split(",") if x.strip()]

    results = run_comparison(
        strategy=args.strategy,
        model_dir=args.model_dir,
        t2imvi_model_dir=args.t2imvi_model_dir,
        idiom_ids=idiom_ids,
        transparency_threshold=transparency_threshold,
        imageability_threshold=imageability_threshold,
        low_score_zero_threshold=args.low_score_zero_threshold,
        rbo_p=rbo_p,
        comparison_score_field=comparison_score_field,
        t2imvi_score_fields=t2imvi_score_fields,
        quadrant_anchor=args.quadrant_anchor,
        save_results=not args.no_save,
        results_dir=None,
    )
    _print_summary(results, show_delta=args.show_delta)
    if args.csv_output is not None:
        _write_csv(results, args.csv_output)
        print(f"\nCSV saved: {args.csv_output}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
