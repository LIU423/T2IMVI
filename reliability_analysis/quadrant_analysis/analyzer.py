"""Quadrant-based reliability analysis.

For each idiom, assign a quadrant by idiom-level transparency and imageability,
then evaluate ranking metrics for multiple score fields inside each quadrant.
"""

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DEFAULT_EXPERIMENT_CONFIG, DEFAULT_SCORING_CONFIG, RESULTS_PATH
from data_loader import discover_idioms, load_combined_data_for_idiom
from experiment1_ranking import compare_rankings_numerical

from .config import QuadrantConfig, SCORE_FIELD_LABELS

logger = logging.getLogger(__name__)


@dataclass
class IdiomQuadrantInfo:
    """Idiom-level quadrant assignment metadata."""

    idiom_id: int
    transparency: float
    imageability: float
    quadrant: str
    n_images: int


@dataclass
class ScoreFieldAggregate:
    """Aggregated metrics for one score field in one quadrant."""

    score_field: str
    score_label: str
    n_idioms: int
    mean_rbo_standard: float
    mean_rbo_tie_aware: float
    mean_icc: float
    mean_pearson_r: float
    mean_mae: float


@dataclass
class QuadrantSummary:
    """Metrics summary for one quadrant."""

    quadrant: str
    n_idioms: int
    idiom_ids: List[int]
    score_field_results: List[ScoreFieldAggregate]


@dataclass
class QuadrantAnalysisResults:
    """Top-level output of quadrant analysis."""

    model_key: str
    timestamp: str
    thresholds: Dict[str, float]
    rbo_p: float
    score_fields: List[str]
    idiom_quadrants: List[IdiomQuadrantInfo]
    quadrant_summaries: List[QuadrantSummary]


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


def _compute_idiom_ti_means(model_key: str, idiom_id: int) -> Optional[Tuple[float, float, int]]:
    """Compute idiom-level transparency/imageability by averaging image-level outputs."""
    image_data = load_combined_data_for_idiom(idiom_id, [model_key])
    valid = [img for img in image_data if model_key in img.model_outputs]
    if not valid:
        return None

    transparencies = [img.model_outputs[model_key].transparency for img in valid]
    imageabilities = [img.model_outputs[model_key].imageability for img in valid]

    return _mean_or_zero(transparencies), _mean_or_zero(imageabilities), len(valid)


def _build_quadrant_membership(
    model_key: str,
    idiom_ids: List[int],
    config: QuadrantConfig,
) -> Tuple[List[IdiomQuadrantInfo], Dict[str, List[int]]]:
    idiom_infos: List[IdiomQuadrantInfo] = []
    groups: Dict[str, List[int]] = {
        "Q1_highT_highI": [],
        "Q2_highT_lowI": [],
        "Q3_lowT_highI": [],
        "Q4_lowT_lowI": [],
    }

    for idiom_id in idiom_ids:
        ti = _compute_idiom_ti_means(model_key, idiom_id)
        if ti is None:
            continue

        transparency, imageability, n_images = ti
        quadrant = _assign_quadrant(
            transparency=transparency,
            imageability=imageability,
            transparency_threshold=config.transparency_threshold,
            imageability_threshold=config.imageability_threshold,
        )
        groups[quadrant].append(idiom_id)
        idiom_infos.append(
            IdiomQuadrantInfo(
                idiom_id=idiom_id,
                transparency=transparency,
                imageability=imageability,
                quadrant=quadrant,
                n_images=n_images,
            )
        )

    return idiom_infos, groups


def _aggregate_one_group_one_score(
    model_key: str,
    idiom_ids: List[int],
    score_field: str,
    rbo_p: float,
) -> ScoreFieldAggregate:
    rbo_std_vals: List[float] = []
    rbo_tie_vals: List[float] = []
    icc_vals: List[float] = []
    pearson_vals: List[float] = []
    mae_vals: List[float] = []

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
        )
        if result is None:
            continue

        rbo_std_vals.append(result.rbo_standard)
        rbo_tie_vals.append(result.rbo_with_ties)
        icc_vals.append(result.icc)
        pearson_vals.append(result.pearson_r)
        mae_vals.append(result.mae)

    return ScoreFieldAggregate(
        score_field=score_field,
        score_label=SCORE_FIELD_LABELS.get(score_field, score_field),
        n_idioms=len(rbo_std_vals),
        mean_rbo_standard=_mean_or_zero(rbo_std_vals),
        mean_rbo_tie_aware=_mean_or_zero(rbo_tie_vals),
        mean_icc=_mean_or_zero(icc_vals),
        mean_pearson_r=_mean_or_zero(pearson_vals),
        mean_mae=_mean_or_zero(mae_vals),
    )


def run_quadrant_analysis(
    model_key: str,
    idiom_ids: Optional[List[int]] = None,
    config: QuadrantConfig = QuadrantConfig(),
    save_results: bool = True,
    results_dir: Optional[Path] = None,
) -> QuadrantAnalysisResults:
    """Run quadrant-based analysis.

    Args:
        model_key: model key in reliability_analysis/config.py
        idiom_ids: optional idiom subset; if None, process all discovered idioms
        config: thresholds and score fields configuration
        save_results: whether to persist JSON output
        results_dir: optional custom output directory
    """
    if idiom_ids is None:
        idiom_ids = discover_idioms()

    logger.info("Starting quadrant analysis for model=%s idioms=%d", model_key, len(idiom_ids))
    idiom_infos, groups = _build_quadrant_membership(model_key=model_key, idiom_ids=idiom_ids, config=config)

    quadrant_summaries: List[QuadrantSummary] = []
    for quadrant in ["Q1_highT_highI", "Q2_highT_lowI", "Q3_lowT_highI", "Q4_lowT_lowI"]:
        members = groups.get(quadrant, [])
        field_results = [
            _aggregate_one_group_one_score(
                model_key=model_key,
                idiom_ids=members,
                score_field=score_field,
                rbo_p=config.rbo_p,
            )
            for score_field in config.score_fields
        ]
        quadrant_summaries.append(
            QuadrantSummary(
                quadrant=quadrant,
                n_idioms=len(members),
                idiom_ids=members,
                score_field_results=field_results,
            )
        )

    results = QuadrantAnalysisResults(
        model_key=model_key,
        timestamp=datetime.now().isoformat(),
        thresholds={
            "transparency": config.transparency_threshold,
            "imageability": config.imageability_threshold,
        },
        rbo_p=config.rbo_p,
        score_fields=config.score_fields,
        idiom_quadrants=idiom_infos,
        quadrant_summaries=quadrant_summaries,
    )

    if save_results:
        out_dir = results_dir or (RESULTS_PATH / "quadrant_analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / (
            f"quadrant_analysis_{model_key}_"
            f"T{config.transparency_threshold:.2f}_I{config.imageability_threshold:.2f}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(results), f, indent=2, ensure_ascii=False)
        logger.info("Saved quadrant results: %s", out_file)

    return results


def print_quadrant_summary(results: QuadrantAnalysisResults) -> None:
    """Print concise terminal summary."""
    print("\n" + "=" * 86)
    print("Quadrant-Based Reliability Analysis")
    print("=" * 86)
    print(f"Model: {results.model_key}")
    print(
        f"Thresholds: transparency={results.thresholds['transparency']}, "
        f"imageability={results.thresholds['imageability']}"
    )
    print(f"RBO p: {results.rbo_p}")

    for quad in results.quadrant_summaries:
        print("\n" + "-" * 86)
        print(f"{quad.quadrant} | idioms={quad.n_idioms}")
        header = (
            f"{'Score':<20} {'n':>4} {'RBO':>10} {'RBO_Tie':>10} "
            f"{'ICC':>10} {'Pearson':>10} {'MAE':>10}"
        )
        print(header)
        print("-" * len(header))
        for row in quad.score_field_results:
            print(
                f"{row.score_label:<20} {row.n_idioms:>4d} "
                f"{row.mean_rbo_standard:>10.4f} {row.mean_rbo_tie_aware:>10.4f} "
                f"{row.mean_icc:>10.4f} {row.mean_pearson_r:>10.4f} {row.mean_mae:>10.4f}"
            )


if __name__ == "__main__":
    # Optional quick smoke run when executed directly.
    default_model = "qwen3_vl_2b_T2IMVI"
    result = run_quadrant_analysis(
        model_key=default_model,
        idiom_ids=None,
        config=QuadrantConfig(
            transparency_threshold=0.5,
            imageability_threshold=0.5,
            rbo_p=DEFAULT_EXPERIMENT_CONFIG.rbo_p,
        ),
        save_results=False,
    )
    print_quadrant_summary(result)
