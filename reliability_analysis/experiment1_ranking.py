"""
Experiment I: Ranking Alignment

This experiment assesses the pipeline's alignment with human judgment using
Rank-Biased Overlap (RBO) to compare model rankings against human ground truth.

Human Ground Truth Ranking: I_fig > I_lit > I_rand

Methodology:
1. For each idiom, classify all images into {I_fig, I_lit, I_rand} based on annotations
2. Generate L_model ranked list based on model's figurative_score (higher = better)
3. Calculate RBO with p=0.9 (top-weighted)

Output: RBO score per idiom, aggregate statistics
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import statistics

from config import (
    DEFAULT_EXPERIMENT_CONFIG,
    ExperimentConfig,
    RESULTS_PATH,
    get_model_config,
    list_available_models,
    export_config,
    DEFAULT_SCORING_CONFIG,
    ScoringConfig,
)
from data_loader import (
    discover_idioms,
    load_combined_data_for_idiom,
    load_all_annotations_for_idiom,
    ImageData,
)
from classification import (
    ImageClass,
    classify_idiom_images,
    create_human_ranked_list,
    create_model_ranked_list,
    IdiomClassificationSummary,
    # New numerical scoring imports
    create_human_ranked_list_by_score,
    create_model_ranked_list_with_scores,
    RankedListWithTies,
    score_all_images,
)
from metrics import (
    rbo,
    calculate_rbo_detailed,
    RBOResult,
    extract_class_sequence,
    rbo_with_ties,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SCORE TYPES FOR MULTI-SCORE COMPARISON
# =============================================================================

# Available score types for comparison
# Key: score_field value, Value: human-readable description
SCORE_TYPES: Dict[str, str] = {
    "s_pot": "S_pot (Potential)",
    "s_fid": "S_fid (Fidelity)",
    "fig_lit_avg": "(Fig + Lit) / 2",
    "entity_action_avg": "Entity+Action Avg",
}


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class IdiomRankingResult:
    """Result for a single idiom's ranking alignment.
    
    Attributes:
        idiom_id: The idiom ID
        rbo_score: RBO score comparing model vs human ranking
        num_images: Total number of images
        num_fig: Number of I_fig images
        num_lit: Number of I_lit images
        num_rand: Number of I_rand images
        human_ranking: Class sequence in human ranking order
        model_ranking: Class sequence in model ranking order
        model_key: Which model was used
        score_field: Which score field was used for ranking
    """
    idiom_id: int
    rbo_score: float
    num_images: int
    num_fig: int
    num_lit: int
    num_rand: int
    human_ranking: List[str]  # Class sequence
    model_ranking: List[str]  # Class sequence
    model_key: str
    score_field: str


@dataclass
class IdiomNumericalRankingResult:
    """Result for a single idiom's ranking alignment using numerical scoring.
    
    This uses the new scoring system where each annotation label has a weight:
    - Figurative+Literal = 20
    - Figurative = 15
    - Literal = 10
    - Partial Literal = 5
    - None = 0
    
    Human score = sum of 5 annotation weights (range 0-100).
    
    Attributes:
        idiom_id: The idiom ID
        num_images: Total number of images
        model_key: Which model was used
        score_field: Which score field was used for ranking
        
        # RBO metrics
        rbo_standard: Standard RBO (ignoring ties)
        ta_rbo: Tie-aware RBO accounting for tied human scores
        
        # Tie information
        num_tie_groups: Number of score groups with multiple images
        
        # Rankings for debugging
        human_ranking_ids: Image IDs in human score order
        model_ranking_ids: Image IDs in model score order
    """
    idiom_id: int
    num_images: int
    model_key: str
    score_field: str
    
    # RBO metrics
    rbo_standard: float
    ta_rbo: float
    tta_rbo: float
    
    # Tie information
    num_tie_groups: int
    
    # Rankings
    human_ranking_ids: List[int]
    model_ranking_ids: List[int]


@dataclass
class ExperimentIResults:
    """Complete results for Experiment I.
    
    Attributes:
        model_key: Which model was evaluated
        score_field: Which score field was used
        rbo_p: RBO persistence parameter
        timestamp: When the experiment was run
        idiom_results: Results for each idiom
        aggregate_stats: Aggregate statistics
    """
    model_key: str
    score_field: str
    rbo_p: float
    timestamp: str
    idiom_results: List[IdiomRankingResult]
    aggregate_stats: Dict[str, float]


@dataclass
class ExperimentINumericalResults:
    """Complete results for Experiment I using numerical scoring.
    
    Attributes:
        model_key: Which model was evaluated
        score_field: Which score field was used
        rbo_p: RBO persistence parameter
        timestamp: When the experiment was run
        scoring_config: The scoring weights used
        idiom_results: Results for each idiom
        aggregate_stats: Aggregate statistics across all metrics
    """
    model_key: str
    score_field: str
    rbo_p: float
    timestamp: str
    scoring_weights: Dict[str, int]  # The label weights used
    krippendorff_alpha: Optional[float]
    idiom_results: List[IdiomNumericalRankingResult]
    aggregate_stats: Dict[str, float]


@dataclass
class MultiScoreResults:
    """Results from running Experiment I with multiple score types.
    
    Enables side-by-side comparison of metrics across different scoring methods.
    
    Attributes:
        model_key: Which model was evaluated
        rbo_p: RBO persistence parameter
        timestamp: When the experiment was run
        scoring_weights: The annotation weights used for human scoring
        results_by_score_type: Dictionary mapping score_type to results
    """
    model_key: str
    rbo_p: float
    timestamp: str
    scoring_weights: Dict[str, int]
    krippendorff_alpha: Optional[float]
    results_by_score_type: Dict[str, ExperimentINumericalResults]


# =============================================================================
# RANKING COMPARISON
# =============================================================================

def compare_rankings_for_idiom(
    idiom_id: int,
    image_data_list: List[ImageData],
    model_key: str,
    score_field: str = "figurative_score",
    rbo_p: float = 0.9,
) -> Optional[IdiomRankingResult]:
    """Compare model ranking against human ranking for a single idiom.
    
    Args:
        idiom_id: The idiom ID
        image_data_list: Combined annotation and model output data
        model_key: Which model to use
        score_field: Which score field for ranking
        rbo_p: RBO persistence parameter
        
    Returns:
        IdiomRankingResult if successful, None if insufficient data
    """
    # Filter to images that have model output
    valid_images = [
        img for img in image_data_list 
        if model_key in img.model_outputs
    ]
    
    if len(valid_images) < 2:
        logger.warning(
            f"Idiom {idiom_id}: Not enough images with model output "
            f"({len(valid_images)} < 2)"
        )
        return None
    
    # Classify images
    summary = classify_idiom_images(valid_images)
    
    # Create human ground truth ranking (I_fig > I_lit > I_rand)
    human_ranked = create_human_ranked_list(summary)
    
    # Create model ranking (by score, descending)
    model_ranked = create_model_ranked_list(
        valid_images, model_key, summary, score_field
    )
    
    if not human_ranked or not model_ranked:
        logger.warning(f"Idiom {idiom_id}: Empty ranking lists")
        return None
    
    # Extract class sequences for RBO comparison
    human_classes = extract_class_sequence(human_ranked)
    model_classes = extract_class_sequence(model_ranked)
    
    # Convert ImageClass enum to string for serialization
    human_classes_str = [c.value for c in human_classes]
    model_classes_str = [c.value for c in model_classes]
    
    # Calculate RBO
    rbo_score = rbo(human_classes_str, model_classes_str, p=rbo_p)
    
    return IdiomRankingResult(
        idiom_id=idiom_id,
        rbo_score=rbo_score,
        num_images=len(valid_images),
        num_fig=summary.fig_count,
        num_lit=summary.lit_count,
        num_rand=summary.rand_count,
        human_ranking=human_classes_str,
        model_ranking=model_classes_str,
        model_key=model_key,
        score_field=score_field,
    )


def compare_rankings_numerical(
    idiom_id: int,
    image_data_list: List[ImageData],
    model_key: str,
    score_field: str = "figurative_score",
    rbo_p: float = 0.9,
    scoring_config: ScoringConfig = DEFAULT_SCORING_CONFIG,
    low_score_zero_threshold: Optional[int] = None,
) -> Optional[IdiomNumericalRankingResult]:
    """Compare model ranking against human ranking using numerical scoring.
    
    This uses the new scoring system where each annotation label has a weight,
    and human score = sum of 5 annotation weights.
    
    Handles ties: When human scores are tied, any ordering of tied items
    in the model ranking is considered equally valid.
    
    Args:
        idiom_id: The idiom ID
        image_data_list: Combined annotation and model output data
        model_key: Which model to use
        score_field: Which score field for ranking
        rbo_p: RBO persistence parameter
        scoring_config: Configuration for annotation weights
        low_score_zero_threshold:
            If set, human scores <= threshold are collapsed to 0 before
            tie-aware RBO is computed for the additional metric.
        
    Returns:
        IdiomNumericalRankingResult if successful, None if insufficient data
    """
    # Filter to images that have model output
    valid_images = [
        img for img in image_data_list 
        if model_key in img.model_outputs
    ]
    
    if len(valid_images) < 2:
        logger.warning(
            f"Idiom {idiom_id}: Not enough images with model output "
            f"({len(valid_images)} < 2)"
        )
        return None
    
    # Create human ranked list with tie information
    human_ranked = create_human_ranked_list_by_score(valid_images, scoring_config)
    
    # Create model ranked list
    model_ranking_ids, model_score_map = create_model_ranked_list_with_scores(
        valid_images, model_key, score_field
    )
    
    if not human_ranked.ranked_image_ids or not model_ranking_ids:
        logger.warning(f"Idiom {idiom_id}: Empty ranking lists")
        return None
    
    # Ensure both rankings overlap on enough items.
    aligned_image_ids = [
        img_id for img_id in human_ranked.ranked_image_ids 
        if img_id in model_score_map
    ]
    
    if len(aligned_image_ids) < 2:
        logger.warning(f"Idiom {idiom_id}: Not enough aligned images")
        return None
    
    # Prepare tie groups for RBO calculation.
    tie_groups = [
        (tg.score, tg.image_ids) for tg in human_ranked.tie_groups
    ]
    
    # Calculate RBO metrics only.
    rbo_standard = rbo(human_ranked.ranked_image_ids, model_ranking_ids, rbo_p)
    ta_rbo = rbo_with_ties(
        human_ranked.ranked_image_ids, 
        model_ranking_ids, 
        tie_groups, 
        rbo_p
    )
    tta_rbo = ta_rbo
    if low_score_zero_threshold is not None:
        adjusted_score_to_images: Dict[int, List[int]] = {}
        for image_id in human_ranked.ranked_image_ids:
            raw_score = int(human_ranked.image_scores.get(image_id, 0))
            adjusted_score = 0 if raw_score <= low_score_zero_threshold else raw_score
            if adjusted_score not in adjusted_score_to_images:
                adjusted_score_to_images[adjusted_score] = []
            adjusted_score_to_images[adjusted_score].append(image_id)

        adjusted_ranked_ids: List[int] = []
        adjusted_tie_groups: List[Tuple[int, List[int]]] = []
        for score in sorted(adjusted_score_to_images.keys(), reverse=True):
            image_ids = sorted(adjusted_score_to_images[score])
            adjusted_ranked_ids.extend(image_ids)
            adjusted_tie_groups.append((score, image_ids))

        tta_rbo = rbo_with_ties(
            adjusted_ranked_ids,
            model_ranking_ids,
            adjusted_tie_groups,
            rbo_p,
        )
    
    # Count tie groups (groups with >1 image)
    num_tie_groups = sum(1 for tg in human_ranked.tie_groups if len(tg.image_ids) > 1)
    
    return IdiomNumericalRankingResult(
        idiom_id=idiom_id,
        num_images=len(valid_images),
        model_key=model_key,
        score_field=score_field,
        rbo_standard=rbo_standard,
        ta_rbo=ta_rbo,
        tta_rbo=tta_rbo,
        num_tie_groups=num_tie_groups,
        human_ranking_ids=human_ranked.ranked_image_ids,
        model_ranking_ids=model_ranking_ids,
    )


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def calculate_krippendorff_alpha_for_irfl(
    idiom_ids: List[int],
    scoring_config: ScoringConfig,
    expected_raters: int = 5,
) -> Optional[float]:
    """Compute Krippendorff's Alpha (ordinal) from IRFL annotations.

    Uses ordinal indices derived from scoring weights to preserve label order.
    """
    try:
        import numpy as np
        import krippendorff
    except ImportError as exc:
        logger.warning("Skip Krippendorff alpha: missing dependency (%s)", exc)
        return None

    # Convert weight-based labels into ordinal ranks: 0..N-1
    ordered_labels = sorted(
        scoring_config.weights.items(),
        key=lambda item: item[1],
    )
    ordinal_map = {label: idx for idx, (label, _weight) in enumerate(ordered_labels)}
    value_domain = list(range(len(ordinal_map)))

    rater_data: List[List[float]] = [[] for _ in range(expected_raters)]
    total_images = 0
    skipped_bad_rater_count = 0
    skipped_unknown_label = 0

    for idiom_id in idiom_ids:
        for annotation in load_all_annotations_for_idiom(idiom_id):
            if len(annotation.annotations) != expected_raters:
                skipped_bad_rater_count += 1
                continue
            total_images += 1

            for i, label in enumerate(annotation.annotations):
                clean_label = label.strip()
                if clean_label in ordinal_map:
                    rater_data[i].append(float(ordinal_map[clean_label]))
                else:
                    skipped_unknown_label += 1
                    rater_data[i].append(np.nan)

    if total_images == 0:
        logger.warning("Skip Krippendorff alpha: no valid annotations found")
        return None

    try:
        alpha = krippendorff.alpha(
            reliability_data=np.array(rater_data, dtype=float),
            level_of_measurement="ordinal",
            value_domain=value_domain,
        )
    except Exception as exc:
        logger.warning("Krippendorff alpha calculation failed: %s", exc)
        return None

    logger.info(
        "Krippendorff alpha (ordinal) = %.4f | images=%d | skipped_bad_raters=%d | unknown_labels=%d",
        alpha,
        total_images,
        skipped_bad_rater_count,
        skipped_unknown_label,
    )
    return float(alpha)

def run_experiment_i(
    model_key: str,
    score_field: Optional[str] = None,
    idiom_ids: Optional[List[int]] = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    save_results: bool = True,
    results_dir: Optional[Path] = None,
) -> ExperimentIResults:
    """Run Experiment I: Ranking Alignment.
    
    Args:
        model_key: Which model to evaluate
        score_field: Which score field to use (default: from model config)
        idiom_ids: Specific idioms to process (default: all)
        config: Experiment configuration
        save_results: Whether to save results to file
        results_dir: Directory for results (default: RESULTS_PATH)
        
    Returns:
        ExperimentIResults with all results and statistics
    """
    logger.info(f"Starting Experiment I for model: {model_key}")
    
    # Get model configuration
    model_config = get_model_config(model_key)
    if score_field is None:
        score_field = model_config.score_field
    
    # Discover idioms if not specified
    if idiom_ids is None:
        idiom_ids = discover_idioms()
    
    logger.info(f"Processing {len(idiom_ids)} idioms...")
    
    # Process each idiom
    idiom_results: List[IdiomRankingResult] = []
    
    for i, idiom_id in enumerate(idiom_ids):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(idiom_ids)}")
        
        # Load data for this idiom
        image_data = load_combined_data_for_idiom(idiom_id, [model_key])
        
        if not image_data:
            logger.debug(f"  Idiom {idiom_id}: No data found")
            continue
        
        # Compare rankings
        result = compare_rankings_for_idiom(
            idiom_id=idiom_id,
            image_data_list=image_data,
            model_key=model_key,
            score_field=score_field,
            rbo_p=config.rbo_p,
        )
        
        if result:
            idiom_results.append(result)
    
    logger.info(f"Processed {len(idiom_results)} idioms successfully")
    
    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_stats(idiom_results)
    
    # Create results object
    results = ExperimentIResults(
        model_key=model_key,
        score_field=score_field,
        rbo_p=config.rbo_p,
        timestamp=datetime.now().isoformat(),
        idiom_results=idiom_results,
        aggregate_stats=aggregate_stats,
    )
    
    # Save results
    if save_results:
        save_experiment_results(results, results_dir)
    
    return results


def calculate_aggregate_stats(
    idiom_results: List[IdiomRankingResult]
) -> Dict[str, float]:
    """Calculate aggregate statistics from idiom results.
    
    Args:
        idiom_results: List of IdiomRankingResult
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not idiom_results:
        return {
            "count": 0,
            "mean_rbo": 0.0,
            "std_rbo": 0.0,
            "min_rbo": 0.0,
            "max_rbo": 0.0,
            "median_rbo": 0.0,
        }
    
    rbo_scores = [r.rbo_score for r in idiom_results]
    total_images = sum(r.num_images for r in idiom_results)
    total_fig = sum(r.num_fig for r in idiom_results)
    total_lit = sum(r.num_lit for r in idiom_results)
    total_rand = sum(r.num_rand for r in idiom_results)
    
    stats = {
        "count": len(idiom_results),
        "total_images": total_images,
        "total_fig": total_fig,
        "total_lit": total_lit,
        "total_rand": total_rand,
        "mean_rbo": statistics.mean(rbo_scores),
        "std_rbo": statistics.stdev(rbo_scores) if len(rbo_scores) > 1 else 0.0,
        "min_rbo": min(rbo_scores),
        "max_rbo": max(rbo_scores),
        "median_rbo": statistics.median(rbo_scores),
    }
    
    # Calculate percentiles
    sorted_scores = sorted(rbo_scores)
    n = len(sorted_scores)
    stats["p25_rbo"] = sorted_scores[int(n * 0.25)] if n >= 4 else stats["min_rbo"]
    stats["p75_rbo"] = sorted_scores[int(n * 0.75)] if n >= 4 else stats["max_rbo"]
    
    return stats


def run_experiment_i_numerical(
    model_key: str,
    score_field: Optional[str] = None,
    idiom_ids: Optional[List[int]] = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    scoring_config: ScoringConfig = DEFAULT_SCORING_CONFIG,
    precomputed_krippendorff_alpha: Optional[float] = None,
    save_results: bool = True,
    results_dir: Optional[Path] = None,
) -> ExperimentINumericalResults:
    """Run Experiment I with numerical scoring: Ranking Alignment.
    
    Uses the new scoring system where:
    - Figurative+Literal = 20, Figurative = 15, Literal = 10, 
      Partial Literal = 5, None = 0
    - Human score = sum of 5 annotation weights (range 0-100)
    
    Calculates RBO metrics:
    - RBO (standard and tie-aware)
    
    Args:
        model_key: Which model to evaluate
        score_field: Which score field to use (default: from model config)
        idiom_ids: Specific idioms to process (default: all)
        config: Experiment configuration
        scoring_config: Configuration for annotation weights
        precomputed_krippendorff_alpha: Optional precomputed alpha value
        save_results: Whether to save results to file
        results_dir: Directory for results (default: RESULTS_PATH)
        
    Returns:
        ExperimentINumericalResults with all results and statistics
    """
    logger.info(f"Starting Experiment I (Numerical) for model: {model_key}")
    logger.info(f"Scoring weights: {scoring_config.weights}")
    
    # Get model configuration
    model_config = get_model_config(model_key)
    if score_field is None:
        score_field = model_config.score_field
    
    # Discover idioms if not specified
    if idiom_ids is None:
        idiom_ids = discover_idioms()
    
    krippendorff_alpha = precomputed_krippendorff_alpha
    if krippendorff_alpha is None:
        krippendorff_alpha = calculate_krippendorff_alpha_for_irfl(
            idiom_ids=idiom_ids,
            scoring_config=scoring_config,
        )

    logger.info(f"Processing {len(idiom_ids)} idioms...")
    
    # Process each idiom
    idiom_results: List[IdiomNumericalRankingResult] = []
    
    for i, idiom_id in enumerate(idiom_ids):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(idiom_ids)}")
        
        # Load data for this idiom
        image_data = load_combined_data_for_idiom(idiom_id, [model_key])
        
        if not image_data:
            logger.debug(f"  Idiom {idiom_id}: No data found")
            continue
        
        # Compare rankings using numerical scoring
        result = compare_rankings_numerical(
            idiom_id=idiom_id,
            image_data_list=image_data,
            model_key=model_key,
            score_field=score_field,
            rbo_p=config.rbo_p,
            scoring_config=scoring_config,
        )
        
        if result:
            idiom_results.append(result)
    
    logger.info(f"Processed {len(idiom_results)} idioms successfully")
    
    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_stats_numerical(idiom_results)
    
    # Create results object
    results = ExperimentINumericalResults(
        model_key=model_key,
        score_field=score_field,
        rbo_p=config.rbo_p,
        timestamp=datetime.now().isoformat(),
        scoring_weights=scoring_config.weights,
        krippendorff_alpha=krippendorff_alpha,
        idiom_results=idiom_results,
        aggregate_stats=aggregate_stats,
    )
    
    # Save results
    if save_results:
        save_numerical_experiment_results(results, results_dir)
    
    return results


def calculate_aggregate_stats_numerical(
    idiom_results: List[IdiomNumericalRankingResult]
) -> Dict[str, float]:
    """Calculate aggregate statistics from numerical idiom results.
    
    Args:
        idiom_results: List of IdiomNumericalRankingResult
        
    Returns:
        Dictionary with aggregate statistics for all metrics
    """
    if not idiom_results:
        return {
            "count": 0,
            "total_images": 0,
            "mean_rbo_standard": 0.0,
            "mean_ta_rbo": 0.0,
            "mean_tta_rbo": 0.0,
        }
    
    # Extract metrics lists
    rbo_standard = [r.rbo_standard for r in idiom_results]
    ta_rbo_vals = [r.ta_rbo for r in idiom_results]
    tta_rbo_vals = [r.tta_rbo for r in idiom_results]
    
    total_images = sum(r.num_images for r in idiom_results)
    total_tie_groups = sum(r.num_tie_groups for r in idiom_results)
    
    stats = {
        "count": len(idiom_results),
        "total_images": total_images,
        "total_tie_groups": total_tie_groups,
        
        # RBO Standard
        "mean_rbo_standard": statistics.mean(rbo_standard),
        "std_rbo_standard": statistics.stdev(rbo_standard) if len(rbo_standard) > 1 else 0.0,
        "min_rbo_standard": min(rbo_standard),
        "max_rbo_standard": max(rbo_standard),
        "median_rbo_standard": statistics.median(rbo_standard),
        
        # RBO with Ties
        "mean_ta_rbo": statistics.mean(ta_rbo_vals),
        "std_ta_rbo": statistics.stdev(ta_rbo_vals) if len(ta_rbo_vals) > 1 else 0.0,
        "min_ta_rbo": min(ta_rbo_vals),
        "max_ta_rbo": max(ta_rbo_vals),
        "median_ta_rbo": statistics.median(ta_rbo_vals),
        "mean_tta_rbo": statistics.mean(tta_rbo_vals),
        "std_tta_rbo": (
            statistics.stdev(tta_rbo_vals) if len(tta_rbo_vals) > 1 else 0.0
        ),
        "min_tta_rbo": min(tta_rbo_vals),
        "max_tta_rbo": max(tta_rbo_vals),
        "median_tta_rbo": statistics.median(tta_rbo_vals),
    }
    
    return stats


def run_experiment_i_multi_score(
    model_key: str,
    idiom_ids: Optional[List[int]] = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    scoring_config: ScoringConfig = DEFAULT_SCORING_CONFIG,
    save_results: bool = True,
    results_dir: Optional[Path] = None,
) -> MultiScoreResults:
    """Run Experiment I with all score types for comparison.
    
    Runs the numerical scoring experiment with:
    - s_pot: Potential score
    - s_fid: Fidelity score  
    - fig_lit_avg: (figurative_score + literal_score) / 2
    - entity_action_avg: mean of entity/action scores from figurative/literal tracks
    
    This allows direct comparison of how well each scoring method
    aligns with human judgment.
    
    Args:
        model_key: Which model to evaluate
        idiom_ids: Specific idioms to process (default: all)
        config: Experiment configuration
        scoring_config: Configuration for annotation weights
        save_results: Whether to save results to file
        results_dir: Directory for results (default: RESULTS_PATH)
        
    Returns:
        MultiScoreResults with results for all score types
    """
    logger.info(f"Starting Experiment I (Multi-Score) for model: {model_key}")
    logger.info(f"Score types: {list(SCORE_TYPES.keys())}")

    if idiom_ids is None:
        idiom_ids = discover_idioms()

    krippendorff_alpha = calculate_krippendorff_alpha_for_irfl(
        idiom_ids=idiom_ids,
        scoring_config=scoring_config,
    )
    
    results_by_score_type: Dict[str, ExperimentINumericalResults] = {}
    
    for score_type, description in SCORE_TYPES.items():
        logger.info(f"\n--- Running with {description} ---")
        
        result = run_experiment_i_numerical(
            model_key=model_key,
            score_field=score_type,
            idiom_ids=idiom_ids,
            config=config,
            scoring_config=scoring_config,
            precomputed_krippendorff_alpha=krippendorff_alpha,
            save_results=save_results,
            results_dir=results_dir,
        )
        
        results_by_score_type[score_type] = result
    
    multi_results = MultiScoreResults(
        model_key=model_key,
        rbo_p=config.rbo_p,
        timestamp=datetime.now().isoformat(),
        scoring_weights=scoring_config.weights,
        krippendorff_alpha=krippendorff_alpha,
        results_by_score_type=results_by_score_type,
    )
    
    # Save multi-score summary
    if save_results:
        save_multi_score_results(multi_results, results_dir)
    
    return multi_results


def save_multi_score_results(
    results: MultiScoreResults,
    results_dir: Optional[Path] = None,
) -> Path:
    """Save multi-score comparison results to JSON file.
    
    Args:
        results: The multi-score results
        results_dir: Directory for results
        
    Returns:
        Path to saved file
    """
    if results_dir is None:
        results_dir = RESULTS_PATH / "experiment1_numerical"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with model and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp1_multi_score_{results.model_key}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Create comparison summary
    comparison_data = {
        "model_key": results.model_key,
        "rbo_p": results.rbo_p,
        "timestamp": results.timestamp,
        "scoring_weights": results.scoring_weights,
        "krippendorff_alpha": results.krippendorff_alpha,
        "score_types": list(SCORE_TYPES.keys()),
        "comparison": {},
    }
    
    # Extract key metrics for comparison
    for score_type, exp_results in results.results_by_score_type.items():
        stats = exp_results.aggregate_stats
        comparison_data["comparison"][score_type] = {
            "rbo_standard": stats.get("mean_rbo_standard", 0.0),
            "ta_rbo": stats.get("mean_ta_rbo", 0.0),
            "tta_rbo": stats.get("mean_tta_rbo", 0.0),
            "count": stats.get("count", 0),
        }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Multi-score results saved to: {filepath}")
    
    return filepath


def save_numerical_experiment_results(
    results: ExperimentINumericalResults,
    results_dir: Optional[Path] = None,
) -> Path:
    """Save numerical experiment results to JSON file.
    
    Args:
        results: The experiment results
        results_dir: Directory for results
        
    Returns:
        Path to saved file
    """
    if results_dir is None:
        results_dir = RESULTS_PATH / "experiment1_numerical"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with model and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp1_num_{results.model_key}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Convert to serializable format
    data = {
        "model_key": results.model_key,
        "score_field": results.score_field,
        "rbo_p": results.rbo_p,
        "timestamp": results.timestamp,
        "scoring_weights": results.scoring_weights,
        "krippendorff_alpha": results.krippendorff_alpha,
        "aggregate_stats": results.aggregate_stats,
        "idiom_results": [asdict(r) for r in results.idiom_results],
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Numerical results saved to: {filepath}")
    
    # Also save a summary file
    summary_filepath = results_dir / f"exp1_num_{results.model_key}_latest_summary.json"
    summary_data = {
        "model_key": results.model_key,
        "score_field": results.score_field,
        "rbo_p": results.rbo_p,
        "timestamp": results.timestamp,
        "scoring_weights": results.scoring_weights,
        "krippendorff_alpha": results.krippendorff_alpha,
        "aggregate_stats": results.aggregate_stats,
    }
    
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def print_numerical_experiment_summary(results: ExperimentINumericalResults) -> None:
    """Print a summary of numerical experiment results.
    
    Args:
        results: The experiment results
    """
    stats = results.aggregate_stats
    
    print("\n" + "=" * 80)
    print("EXPERIMENT I (NUMERICAL): RANKING ALIGNMENT - RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nModel: {results.model_key}")
    print(f"Score Field: {results.score_field}")
    print(f"RBO Parameter (p): {results.rbo_p}")
    print(f"Timestamp: {results.timestamp}")
    print(f"\nScoring Weights: {results.scoring_weights}")
    if results.krippendorff_alpha is None:
        print("Krippendorff's Alpha (ordinal): N/A")
    else:
        alpha = results.krippendorff_alpha
        if alpha >= 0.8:
            alpha_interp = "Excellent"
        elif alpha >= 0.667:
            alpha_interp = "Substantial"
        elif alpha >= 0.4:
            alpha_interp = "Moderate"
        else:
            alpha_interp = "Poor"
        print(f"Krippendorff's Alpha (ordinal): {alpha:.4f} ({alpha_interp})")
    
    print("\n--- Data Statistics ---")
    print(f"  Idioms processed: {stats.get('count', 0)}")
    print(f"  Total images: {stats.get('total_images', 0)}")
    print(f"  Total tie groups: {stats.get('total_tie_groups', 0)}")
    
    print("\n--- RBO Scores (Standard) ---")
    print(f"  Mean:   {stats.get('mean_rbo_standard', 0):.4f}")
    print(f"  Std:    {stats.get('std_rbo_standard', 0):.4f}")
    print(f"  Median: {stats.get('median_rbo_standard', 0):.4f}")
    print(f"  Range:  [{stats.get('min_rbo_standard', 0):.4f}, {stats.get('max_rbo_standard', 0):.4f}]")
    
    print("\n--- RBO Scores (Tie-Aware) ---")
    print(f"  Mean:   {stats.get('mean_ta_rbo', 0):.4f}")
    print(f"  Std:    {stats.get('std_ta_rbo', 0):.4f}")
    print(f"  Median: {stats.get('median_ta_rbo', 0):.4f}")
    print(f"  Range:  [{stats.get('min_ta_rbo', 0):.4f}, {stats.get('max_ta_rbo', 0):.4f}]")

    print("\n--- RBO Scores (Threshold Tie-Aware) ---")
    print(f"  Mean:   {stats.get('mean_tta_rbo', 0):.4f}")
    print(f"  Std:    {stats.get('std_tta_rbo', 0):.4f}")
    print(f"  Median: {stats.get('median_tta_rbo', 0):.4f}")
    print(f"  Range:  [{stats.get('min_tta_rbo', 0):.4f}, {stats.get('max_tta_rbo', 0):.4f}]")
    
    print("\n--- Interpretation ---")
    mean_rbo = stats.get('mean_ta_rbo', 0)
    
    # RBO interpretation
    if mean_rbo >= 0.9:
        rbo_interp = "Excellent"
    elif mean_rbo >= 0.7:
        rbo_interp = "Good"
    elif mean_rbo >= 0.5:
        rbo_interp = "Moderate"
    else:
        rbo_interp = "Poor"
    
    print(f"  RBO alignment:      {rbo_interp} ({mean_rbo:.4f})")
    
    print("\n" + "=" * 80)


def print_multi_score_comparison(results: MultiScoreResults) -> None:
    """Print side-by-side comparison of metrics across all score types.
    
    Output format:
    ================================================================================
    EXPERIMENT I: MULTI-SCORE COMPARISON
    ================================================================================
                              s_pot         s_fid      fig_lit_avg  entity_action_avg
    --------------------------------------------------------------------------------
    RBO (Standard)            0.xxxx        0.xxxx        0.xxxx
    TA-RBO                    0.xxxx        0.xxxx        0.xxxx
    TTA-RBO                   0.xxxx        0.xxxx        0.xxxx
    ================================================================================
    
    Args:
        results: MultiScoreResults with data for all score types
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT I: MULTI-SCORE COMPARISON")
    print("=" * 80)
    print(f"\nModel: {results.model_key}")
    print(f"RBO Parameter (p): {results.rbo_p}")
    print(f"Timestamp: {results.timestamp}")
    print(f"Scoring Weights: {results.scoring_weights}")
    if results.krippendorff_alpha is None:
        print("Krippendorff's Alpha (ordinal): N/A")
    else:
        print(f"Krippendorff's Alpha (ordinal): {results.krippendorff_alpha:.4f}")
    
    # Get score types in order
    score_types = list(SCORE_TYPES.keys())
    
    # Get first result to extract idiom count
    first_result = next(iter(results.results_by_score_type.values()))
    idiom_count = first_result.aggregate_stats.get("count", 0)
    total_images = first_result.aggregate_stats.get("total_images", 0)
    
    print(f"\nIdioms processed: {idiom_count}")
    print(f"Total images: {total_images}")
    
    # Column headers
    col_width = 14
    metric_col_width = 22
    
    print("\n" + "-" * 80)
    
    # Header row
    header = f"{'Metric':<{metric_col_width}}"
    for score_type in score_types:
        header += f"{SCORE_TYPES[score_type]:>{col_width}}"
    print(header)
    
    print("-" * 80)
    
    # Metric rows
    metrics = [
        ("RBO (Standard)", "mean_rbo_standard"),
        ("TA-RBO", "mean_ta_rbo"),
        ("TTA-RBO", "mean_tta_rbo"),
    ]
    
    for metric_name, stat_key in metrics:
        row = f"{metric_name:<{metric_col_width}}"
        for score_type in score_types:
            stats = results.results_by_score_type[score_type].aggregate_stats
            value = stats.get(stat_key, 0.0)
            row += f"{value:>{col_width}.4f}"
        print(row)
    
    print("-" * 80)
    
    # Find best score type for each metric
    print("\n--- Best Score Type by Metric ---")
    
    # For all selected RBO metrics, higher is better.
    higher_is_better = ["mean_rbo_standard", "mean_ta_rbo", "mean_tta_rbo"]
    
    for metric_name, stat_key in metrics:
        values = {
            score_type: results.results_by_score_type[score_type].aggregate_stats.get(stat_key, 0.0)
            for score_type in score_types
        }
        
        if stat_key in higher_is_better:
            best_type = max(values, key=values.get)
        else:
            best_type = min(values, key=values.get)
        
        print(f"  {metric_name:<20}: {SCORE_TYPES[best_type]} ({values[best_type]:.4f})")
    
    # Overall interpretation
    print("\n--- Overall Interpretation ---")
    
    # Average across key RBO metrics.
    score_rankings: Dict[str, float] = {}
    for score_type in score_types:
        stats = results.results_by_score_type[score_type].aggregate_stats
        combined = (
            stats.get("mean_rbo_standard", 0.0) * 0.34 +
            stats.get("mean_ta_rbo", 0.0) * 0.33 +
            stats.get("mean_tta_rbo", 0.0) * 0.33
        )
        score_rankings[score_type] = combined
    
    # Sort by combined score
    ranked = sorted(score_rankings.items(), key=lambda x: x[1], reverse=True)
    
    print("\n  Score Type Ranking (by combined metrics):")
    for i, (score_type, combined_score) in enumerate(ranked, 1):
        print(f"    {i}. {SCORE_TYPES[score_type]}: {combined_score:.4f}")
    
    best_overall = ranked[0][0]
    print(f"\n  Recommended score type: {SCORE_TYPES[best_overall]}")
    
    print("\n" + "=" * 80)


def generate_multi_score_latex_table(results: MultiScoreResults) -> str:
    """Generate LaTeX table for multi-score comparison.
    
    Args:
        results: MultiScoreResults with data for all score types
        
    Returns:
        LaTeX table string
    """
    score_types = list(SCORE_TYPES.keys())
    
    # Build header
    header_cols = " & ".join([f"\\textbf{{{SCORE_TYPES[st]}}}" for st in score_types])
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Experiment I: Multi-Score Comparison for """ + results.model_key + r"""}
\begin{tabular}{l""" + "c" * len(score_types) + r"""}
\toprule
\textbf{Metric} & """ + header_cols + r""" \\
\midrule
"""
    
    metrics = [
        ("RBO (Standard)", "mean_rbo_standard"),
        ("TA-RBO", "mean_ta_rbo"),
        ("TTA-RBO", "mean_tta_rbo"),
    ]
    
    for metric_name, stat_key in metrics:
        values = []
        for score_type in score_types:
            stats = results.results_by_score_type[score_type].aggregate_stats
            value = stats.get(stat_key, 0.0)
            values.append(f"{value:.4f}")
        
        latex += metric_name + " & " + " & ".join(values) + r" \\" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\label{tab:exp1_multi_score_""" + results.model_key.replace("_", "-") + r"""}
\end{table}
"""
    
    return latex


# =============================================================================
# RESULTS SAVING AND LOADING
# =============================================================================

def save_experiment_results(
    results: ExperimentIResults,
    results_dir: Optional[Path] = None,
) -> Path:
    """Save experiment results to JSON file.
    
    Args:
        results: The experiment results
        results_dir: Directory for results
        
    Returns:
        Path to saved file
    """
    if results_dir is None:
        results_dir = RESULTS_PATH / "experiment1"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with model and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp1_{results.model_key}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Convert to serializable format
    data = {
        "model_key": results.model_key,
        "score_field": results.score_field,
        "rbo_p": results.rbo_p,
        "timestamp": results.timestamp,
        "aggregate_stats": results.aggregate_stats,
        "idiom_results": [asdict(r) for r in results.idiom_results],
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {filepath}")
    
    # Also save a summary file
    summary_filepath = results_dir / f"exp1_{results.model_key}_latest_summary.json"
    summary_data = {
        "model_key": results.model_key,
        "score_field": results.score_field,
        "rbo_p": results.rbo_p,
        "timestamp": results.timestamp,
        "aggregate_stats": results.aggregate_stats,
    }
    
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def load_experiment_results(filepath: Path) -> ExperimentIResults:
    """Load experiment results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        ExperimentIResults object
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    idiom_results = [
        IdiomRankingResult(**r) for r in data["idiom_results"]
    ]
    
    return ExperimentIResults(
        model_key=data["model_key"],
        score_field=data["score_field"],
        rbo_p=data["rbo_p"],
        timestamp=data["timestamp"],
        idiom_results=idiom_results,
        aggregate_stats=data["aggregate_stats"],
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_experiment_summary(results: ExperimentIResults) -> None:
    """Print a summary of experiment results.
    
    Args:
        results: The experiment results
    """
    stats = results.aggregate_stats
    
    print("\n" + "=" * 70)
    print("EXPERIMENT I: RANKING ALIGNMENT - RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nModel: {results.model_key}")
    print(f"Score Field: {results.score_field}")
    print(f"RBO Parameter (p): {results.rbo_p}")
    print(f"Timestamp: {results.timestamp}")
    
    print("\n--- Data Statistics ---")
    print(f"  Idioms processed: {stats.get('count', 0)}")
    print(f"  Total images: {stats.get('total_images', 0)}")
    print(f"    - I_fig: {stats.get('total_fig', 0)}")
    print(f"    - I_lit: {stats.get('total_lit', 0)}")
    print(f"    - I_rand: {stats.get('total_rand', 0)}")
    
    print("\n--- RBO Score Statistics ---")
    print(f"  Mean RBO: {stats.get('mean_rbo', 0):.4f}")
    print(f"  Std Dev:  {stats.get('std_rbo', 0):.4f}")
    print(f"  Min RBO:  {stats.get('min_rbo', 0):.4f}")
    print(f"  Max RBO:  {stats.get('max_rbo', 0):.4f}")
    print(f"  Median:   {stats.get('median_rbo', 0):.4f}")
    print(f"  P25:      {stats.get('p25_rbo', 0):.4f}")
    print(f"  P75:      {stats.get('p75_rbo', 0):.4f}")
    
    print("\n--- Interpretation ---")
    mean_rbo = stats.get('mean_rbo', 0)
    if mean_rbo >= 0.9:
        interpretation = "Excellent alignment with human judgment"
    elif mean_rbo >= 0.7:
        interpretation = "Good alignment with human judgment"
    elif mean_rbo >= 0.5:
        interpretation = "Moderate alignment with human judgment"
    else:
        interpretation = "Poor alignment with human judgment"
    print(f"  {interpretation}")
    
    print("\n" + "=" * 70)


def generate_latex_table(results: ExperimentIResults) -> str:
    """Generate LaTeX table for the results.
    
    Args:
        results: The experiment results
        
    Returns:
        LaTeX table string
    """
    stats = results.aggregate_stats
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Experiment I: Ranking Alignment Results for """ + results.model_key + r"""}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Idioms Evaluated & """ + str(stats.get('count', 0)) + r""" \\
Total Images & """ + str(stats.get('total_images', 0)) + r""" \\
\midrule
Mean RBO & """ + f"{stats.get('mean_rbo', 0):.4f}" + r""" \\
Std. Dev. & """ + f"{stats.get('std_rbo', 0):.4f}" + r""" \\
Median & """ + f"{stats.get('median_rbo', 0):.4f}" + r""" \\
Min & """ + f"{stats.get('min_rbo', 0):.4f}" + r""" \\
Max & """ + f"{stats.get('max_rbo', 0):.4f}" + r""" \\
\bottomrule
\end{tabular}
\label{tab:exp1_""" + results.model_key.replace("_", "-") + r"""}
\end{table}
"""
    return latex


# =============================================================================
# MULTI-MODEL COMPARISON
# =============================================================================

def compare_models(
    model_keys: List[str],
    idiom_ids: Optional[List[int]] = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> Dict[str, ExperimentIResults]:
    """Run Experiment I for multiple models and compare.
    
    Args:
        model_keys: List of model keys to compare
        idiom_ids: Specific idioms to process
        config: Experiment configuration
        
    Returns:
        Dictionary mapping model_key to results
    """
    all_results = {}
    
    for model_key in model_keys:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_key}")
        logger.info(f"{'='*50}")
        
        results = run_experiment_i(
            model_key=model_key,
            idiom_ids=idiom_ids,
            config=config,
            save_results=True,
        )
        
        all_results[model_key] = results
        print_experiment_summary(results)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<40} {'Mean RBO':>10} {'Std':>8} {'Median':>10}")
    print("-" * 70)
    
    for model_key, results in all_results.items():
        stats = results.aggregate_stats
        print(
            f"{model_key:<40} "
            f"{stats.get('mean_rbo', 0):>10.4f} "
            f"{stats.get('std_rbo', 0):>8.4f} "
            f"{stats.get('median_rbo', 0):>10.4f}"
        )
    
    print("=" * 70)
    
    return all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment I: Ranking Alignment"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model key to evaluate (default: first available)"
    )
    parser.add_argument(
        "--all-models", "-a",
        action="store_true",
        help="Evaluate all available models"
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=0.9,
        help="RBO persistence parameter (default: 0.9)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of idioms to process"
    )
    parser.add_argument(
        "--idiom-ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Specific idiom IDs to process. Supports both "
            "'--idiom-ids 1 2 3' and '--idiom-ids 1,2,3'. "
            "When set, it overrides --limit."
        ),
    )
    parser.add_argument(
        "--numerical", "-n",
        action="store_true",
        help="Use numerical scoring (RBO, TA-RBO, TTA-RBO with Krippendorff alpha)"
    )
    parser.add_argument(
        "--multi-score", "-M",
        action="store_true",
        help="Compare all score types (s_pot, s_fid, fig_lit_avg, entity_action_avg)"
    )
    
    args = parser.parse_args()
    
    # Configure experiment
    config = ExperimentConfig(rbo_p=args.rbo_p)
    
    # Get idiom IDs
    if args.idiom_ids:
        parsed_ids: List[int] = []
        for raw in args.idiom_ids:
            for token in raw.split(","):
                token = token.strip()
                if token:
                    parsed_ids.append(int(token))
        idiom_ids = parsed_ids
    else:
        idiom_ids = discover_idioms()
        if args.limit:
            idiom_ids = idiom_ids[:args.limit]
    
    # Determine which models to evaluate
    available_models = list_available_models()
    
    if args.all_models:
        model_keys = available_models
    elif args.model:
        if args.model not in available_models:
            print(f"Error: Model '{args.model}' not found.")
            print(f"Available models: {available_models}")
            exit(1)
        model_keys = [args.model]
    else:
        model_keys = [available_models[0]] if available_models else []
    
    if not model_keys:
        print("No models available. Check config.py MODEL_CONFIGS.")
        exit(1)
    
    # Run experiment(s)
    if args.multi_score:
        # Multi-score comparison mode
        print("\n*** MULTI-SCORE COMPARISON MODE ***")
        print(f"Score types: {list(SCORE_TYPES.keys())}")
        print(f"Scoring weights: {DEFAULT_SCORING_CONFIG.weights}")
        print()
        
        for model_key in model_keys:
            results = run_experiment_i_multi_score(
                model_key=model_key,
                idiom_ids=idiom_ids,
                config=config,
            )
            print_multi_score_comparison(results)
    elif args.numerical:
        # Use new numerical scoring system
        print("\n*** Using NUMERICAL SCORING system ***")
        print(f"Scoring weights: {DEFAULT_SCORING_CONFIG.weights}")
        print()
        
        for model_key in model_keys:
            results = run_experiment_i_numerical(
                model_key=model_key,
                idiom_ids=idiom_ids,
                config=config,
            )
            print_numerical_experiment_summary(results)
    else:
        # Use original categorical scoring system
        if len(model_keys) == 1:
            results = run_experiment_i(
                model_key=model_keys[0],
                idiom_ids=idiom_ids,
                config=config,
            )
            print_experiment_summary(results)
        else:
            compare_models(model_keys, idiom_ids, config)
