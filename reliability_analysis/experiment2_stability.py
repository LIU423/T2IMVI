"""
Experiment II: Score Stability Analysis (Test-Retest)

This experiment assesses the stability of the scoring mechanism against
minor perturbations using the SimPD (Similarity of Promotion and Demotion) metric.

Methodology:
1. Run A: Standard evaluation (baseline model output)
2. Run B: Evaluation with perturbations (different model/prompt/etc.)
3. Calculate SimPD to measure rank volatility

Low SimPD = high volatility (unstable), High SimPD = stable rankings (reliable)

This module is designed with placeholders for different perturbation strategies.
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
    PERTURBATION_CONFIGS,
    PerturbationConfig,
    get_model_config,
    get_perturbation_config,
    list_available_models,
    list_available_perturbations,
)
from data_loader import (
    discover_idioms,
    load_combined_data_for_idiom,
    ImageData,
)
from classification import (
    classify_idiom_images,
    create_model_ranked_list,
)
from metrics import (
    simpd,
    simpd_base,
    calculate_simpd_detailed,
    SimPDResult,
    extract_ranking_order,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class IdiomStabilityResult:
    """Result for a single idiom's stability analysis.
    
    Attributes:
        idiom_id: The idiom ID
        simpd_score: SimPD score comparing Run A vs Run B
        simpd_variant: Which SimPD variant was used
        num_images: Total number of images with both runs
        sum_promotions: Total promotion magnitude
        sum_demotions: Total demotion magnitude
        run_a_model: Model used for Run A
        run_b_model: Model used for Run B
        perturbation_type: Type of perturbation applied
    """
    idiom_id: int
    simpd_score: float
    simpd_variant: str
    num_images: int
    sum_promotions: int
    sum_demotions: int
    run_a_model: str
    run_b_model: str
    perturbation_type: str


@dataclass
class ExperimentIIResults:
    """Complete results for Experiment II.
    
    Attributes:
        run_a_model: Model used for Run A (baseline)
        run_b_model: Model used for Run B (perturbed)
        perturbation_type: Type of perturbation
        perturbation_name: Name of the perturbation configuration
        simpd_variant: Which SimPD variant was used
        timestamp: When the experiment was run
        idiom_results: Results for each idiom
        aggregate_stats: Aggregate statistics
    """
    run_a_model: str
    run_b_model: str
    perturbation_type: str
    perturbation_name: str
    simpd_variant: str
    timestamp: str
    idiom_results: List[IdiomStabilityResult]
    aggregate_stats: Dict[str, float]


# =============================================================================
# STABILITY COMPARISON
# =============================================================================

def compare_stability_for_idiom(
    idiom_id: int,
    image_data_list: List[ImageData],
    run_a_model: str,
    run_b_model: str,
    score_field_a: str = "figurative_score",
    score_field_b: str = "figurative_score",
    simpd_variant: str = "base",
    perturbation_type: str = "model",
) -> Optional[IdiomStabilityResult]:
    """Compare ranking stability between two runs for a single idiom.
    
    Args:
        idiom_id: The idiom ID
        image_data_list: Combined annotation and model output data
        run_a_model: Model key for Run A (baseline)
        run_b_model: Model key for Run B (perturbed)
        score_field_a: Score field for Run A
        score_field_b: Score field for Run B
        simpd_variant: Which SimPD variant to use
        perturbation_type: Type of perturbation
        
    Returns:
        IdiomStabilityResult if successful, None if insufficient data
    """
    # Filter to images that have BOTH model outputs
    valid_images = [
        img for img in image_data_list
        if run_a_model in img.model_outputs and run_b_model in img.model_outputs
    ]
    
    if len(valid_images) < 2:
        logger.debug(
            f"Idiom {idiom_id}: Not enough images with both model outputs "
            f"({len(valid_images)} < 2)"
        )
        return None
    
    # Classify images for reference
    summary = classify_idiom_images(valid_images)
    
    # Create ranked lists for both runs
    ranking_a = create_model_ranked_list(
        valid_images, run_a_model, summary, score_field_a
    )
    ranking_b = create_model_ranked_list(
        valid_images, run_b_model, summary, score_field_b
    )
    
    if not ranking_a or not ranking_b:
        logger.debug(f"Idiom {idiom_id}: Empty ranking lists")
        return None
    
    # Extract just image IDs for SimPD comparison
    list_a = extract_ranking_order(ranking_a)
    list_b = extract_ranking_order(ranking_b)
    
    # Calculate SimPD with detailed results
    detailed = calculate_simpd_detailed(list_a, list_b, simpd_variant)
    
    return IdiomStabilityResult(
        idiom_id=idiom_id,
        simpd_score=detailed.score,
        simpd_variant=detailed.variant,
        num_images=len(valid_images),
        sum_promotions=detailed.sum_promotions,
        sum_demotions=detailed.sum_demotions,
        run_a_model=run_a_model,
        run_b_model=run_b_model,
        perturbation_type=perturbation_type,
    )


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_experiment_ii_with_perturbation(
    perturbation_key: str,
    idiom_ids: Optional[List[int]] = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    save_results: bool = True,
    results_dir: Optional[Path] = None,
) -> ExperimentIIResults:
    """Run Experiment II using a predefined perturbation configuration.
    
    Args:
        perturbation_key: Key in PERTURBATION_CONFIGS
        idiom_ids: Specific idioms to process
        config: Experiment configuration
        save_results: Whether to save results
        results_dir: Directory for results
        
    Returns:
        ExperimentIIResults with all results and statistics
    """
    perturbation = get_perturbation_config(perturbation_key)
    
    return run_experiment_ii(
        run_a_model=perturbation.run_a_model,
        run_b_model=perturbation.run_b_model,
        perturbation_name=perturbation.name,
        perturbation_type=perturbation.perturbation_type,
        idiom_ids=idiom_ids,
        config=config,
        save_results=save_results,
        results_dir=results_dir,
    )


def run_experiment_ii(
    run_a_model: str,
    run_b_model: str,
    perturbation_name: str = "custom",
    perturbation_type: str = "model",
    score_field_a: Optional[str] = None,
    score_field_b: Optional[str] = None,
    idiom_ids: Optional[List[int]] = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    save_results: bool = True,
    results_dir: Optional[Path] = None,
) -> ExperimentIIResults:
    """Run Experiment II: Score Stability Analysis.
    
    Args:
        run_a_model: Model key for Run A (baseline)
        run_b_model: Model key for Run B (perturbed)
        perturbation_name: Name for this perturbation comparison
        perturbation_type: Type of perturbation
        score_field_a: Score field for Run A
        score_field_b: Score field for Run B
        idiom_ids: Specific idioms to process
        config: Experiment configuration
        save_results: Whether to save results
        results_dir: Directory for results
        
    Returns:
        ExperimentIIResults with all results and statistics
    """
    logger.info(f"Starting Experiment II: {perturbation_name}")
    logger.info(f"  Run A: {run_a_model}")
    logger.info(f"  Run B: {run_b_model}")
    
    # Get model configurations
    config_a = get_model_config(run_a_model)
    config_b = get_model_config(run_b_model)
    
    if score_field_a is None:
        score_field_a = config_a.score_field
    if score_field_b is None:
        score_field_b = config_b.score_field
    
    # Discover idioms if not specified
    if idiom_ids is None:
        idiom_ids = discover_idioms()
    
    logger.info(f"Processing {len(idiom_ids)} idioms...")
    
    # Process each idiom
    idiom_results: List[IdiomStabilityResult] = []
    
    for i, idiom_id in enumerate(idiom_ids):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(idiom_ids)}")
        
        # Load data for this idiom (need both models)
        image_data = load_combined_data_for_idiom(
            idiom_id, [run_a_model, run_b_model]
        )
        
        if not image_data:
            logger.debug(f"  Idiom {idiom_id}: No data found")
            continue
        
        # Compare stability
        result = compare_stability_for_idiom(
            idiom_id=idiom_id,
            image_data_list=image_data,
            run_a_model=run_a_model,
            run_b_model=run_b_model,
            score_field_a=score_field_a,
            score_field_b=score_field_b,
            simpd_variant=config.use_simpd_variant,
            perturbation_type=perturbation_type,
        )
        
        if result:
            idiom_results.append(result)
    
    logger.info(f"Processed {len(idiom_results)} idioms successfully")
    
    # Calculate aggregate statistics
    aggregate_stats = calculate_stability_stats(idiom_results)
    
    # Create results object
    results = ExperimentIIResults(
        run_a_model=run_a_model,
        run_b_model=run_b_model,
        perturbation_type=perturbation_type,
        perturbation_name=perturbation_name,
        simpd_variant=config.use_simpd_variant,
        timestamp=datetime.now().isoformat(),
        idiom_results=idiom_results,
        aggregate_stats=aggregate_stats,
    )
    
    # Save results
    if save_results:
        save_stability_results(results, results_dir)
    
    return results


def calculate_stability_stats(
    idiom_results: List[IdiomStabilityResult]
) -> Dict[str, float]:
    """Calculate aggregate statistics from stability results.
    
    Args:
        idiom_results: List of IdiomStabilityResult
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not idiom_results:
        return {
            "count": 0,
            "mean_simpd": 0.0,
            "std_simpd": 0.0,
            "min_simpd": 0.0,
            "max_simpd": 0.0,
            "median_simpd": 0.0,
        }
    
    simpd_scores = [r.simpd_score for r in idiom_results]
    total_images = sum(r.num_images for r in idiom_results)
    total_promotions = sum(r.sum_promotions for r in idiom_results)
    total_demotions = sum(r.sum_demotions for r in idiom_results)
    
    stats = {
        "count": len(idiom_results),
        "total_images": total_images,
        "total_promotions": total_promotions,
        "total_demotions": total_demotions,
        "mean_simpd": statistics.mean(simpd_scores),
        "std_simpd": statistics.stdev(simpd_scores) if len(simpd_scores) > 1 else 0.0,
        "min_simpd": min(simpd_scores),
        "max_simpd": max(simpd_scores),
        "median_simpd": statistics.median(simpd_scores),
    }
    
    # Calculate percentiles
    sorted_scores = sorted(simpd_scores)
    n = len(sorted_scores)
    stats["p25_simpd"] = sorted_scores[int(n * 0.25)] if n >= 4 else stats["min_simpd"]
    stats["p75_simpd"] = sorted_scores[int(n * 0.75)] if n >= 4 else stats["max_simpd"]
    
    # Calculate volatility index (inverse of stability)
    stats["volatility_index"] = 1 - stats["mean_simpd"]
    
    return stats


# =============================================================================
# RESULTS SAVING AND LOADING
# =============================================================================

def save_stability_results(
    results: ExperimentIIResults,
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
        results_dir = RESULTS_PATH / "experiment2"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = results.perturbation_name.replace(" ", "_").replace("/", "-")
    filename = f"exp2_{safe_name}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Convert to serializable format
    data = {
        "run_a_model": results.run_a_model,
        "run_b_model": results.run_b_model,
        "perturbation_type": results.perturbation_type,
        "perturbation_name": results.perturbation_name,
        "simpd_variant": results.simpd_variant,
        "timestamp": results.timestamp,
        "aggregate_stats": results.aggregate_stats,
        "idiom_results": [asdict(r) for r in results.idiom_results],
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {filepath}")
    
    # Also save summary
    summary_filepath = results_dir / f"exp2_{safe_name}_latest_summary.json"
    summary_data = {
        "run_a_model": results.run_a_model,
        "run_b_model": results.run_b_model,
        "perturbation_type": results.perturbation_type,
        "perturbation_name": results.perturbation_name,
        "simpd_variant": results.simpd_variant,
        "timestamp": results.timestamp,
        "aggregate_stats": results.aggregate_stats,
    }
    
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def load_stability_results(filepath: Path) -> ExperimentIIResults:
    """Load experiment results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        ExperimentIIResults object
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    idiom_results = [
        IdiomStabilityResult(**r) for r in data["idiom_results"]
    ]
    
    return ExperimentIIResults(
        run_a_model=data["run_a_model"],
        run_b_model=data["run_b_model"],
        perturbation_type=data["perturbation_type"],
        perturbation_name=data["perturbation_name"],
        simpd_variant=data["simpd_variant"],
        timestamp=data["timestamp"],
        idiom_results=idiom_results,
        aggregate_stats=data["aggregate_stats"],
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_stability_summary(results: ExperimentIIResults) -> None:
    """Print a summary of stability analysis results.
    
    Args:
        results: The experiment results
    """
    stats = results.aggregate_stats
    
    print("\n" + "=" * 70)
    print("EXPERIMENT II: SCORE STABILITY ANALYSIS - RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nPerturbation: {results.perturbation_name}")
    print(f"Type: {results.perturbation_type}")
    print(f"Run A Model: {results.run_a_model}")
    print(f"Run B Model: {results.run_b_model}")
    print(f"SimPD Variant: {results.simpd_variant}")
    print(f"Timestamp: {results.timestamp}")
    
    print("\n--- Data Statistics ---")
    print(f"  Idioms processed: {stats.get('count', 0)}")
    print(f"  Total images: {stats.get('total_images', 0)}")
    print(f"  Total promotions: {stats.get('total_promotions', 0)}")
    print(f"  Total demotions: {stats.get('total_demotions', 0)}")
    
    print("\n--- SimPD Score Statistics ---")
    print(f"  Mean SimPD: {stats.get('mean_simpd', 0):.4f}")
    print(f"  Std Dev:    {stats.get('std_simpd', 0):.4f}")
    print(f"  Min SimPD:  {stats.get('min_simpd', 0):.4f}")
    print(f"  Max SimPD:  {stats.get('max_simpd', 0):.4f}")
    print(f"  Median:     {stats.get('median_simpd', 0):.4f}")
    print(f"  P25:        {stats.get('p25_simpd', 0):.4f}")
    print(f"  P75:        {stats.get('p75_simpd', 0):.4f}")
    
    print(f"\n--- Volatility Index ---")
    print(f"  Volatility: {stats.get('volatility_index', 0):.4f}")
    
    print("\n--- Interpretation ---")
    mean_simpd = stats.get('mean_simpd', 0)
    if mean_simpd >= 0.9:
        interpretation = "Excellent stability - rankings highly consistent"
    elif mean_simpd >= 0.7:
        interpretation = "Good stability - rankings mostly consistent"
    elif mean_simpd >= 0.5:
        interpretation = "Moderate stability - some rank volatility"
    else:
        interpretation = "Poor stability - high rank volatility"
    print(f"  {interpretation}")
    
    print("\n" + "=" * 70)


def generate_stability_latex_table(results: ExperimentIIResults) -> str:
    """Generate LaTeX table for stability results.
    
    Args:
        results: The experiment results
        
    Returns:
        LaTeX table string
    """
    stats = results.aggregate_stats
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Experiment II: Score Stability Analysis - """ + results.perturbation_name + r"""}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Run A Model & """ + results.run_a_model + r""" \\
Run B Model & """ + results.run_b_model + r""" \\
Perturbation Type & """ + results.perturbation_type + r""" \\
\midrule
Idioms Evaluated & """ + str(stats.get('count', 0)) + r""" \\
Total Images & """ + str(stats.get('total_images', 0)) + r""" \\
\midrule
Mean SimPD & """ + f"{stats.get('mean_simpd', 0):.4f}" + r""" \\
Std. Dev. & """ + f"{stats.get('std_simpd', 0):.4f}" + r""" \\
Median & """ + f"{stats.get('median_simpd', 0):.4f}" + r""" \\
Volatility Index & """ + f"{stats.get('volatility_index', 0):.4f}" + r""" \\
\bottomrule
\end{tabular}
\label{tab:exp2_stability}
\end{table}
"""
    return latex


# =============================================================================
# SELF-STABILITY TEST (SAME MODEL COMPARISON)
# =============================================================================

def run_self_stability_test(
    model_key: str,
    idiom_ids: Optional[List[int]] = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> ExperimentIIResults:
    """Run a self-stability test (same model compared to itself).
    
    This serves as a sanity check - the result should be SimPD = 1.0
    since the same model should produce identical rankings.
    
    Args:
        model_key: Model to test
        idiom_ids: Specific idioms to process
        config: Experiment configuration
        
    Returns:
        ExperimentIIResults (should show perfect stability)
    """
    logger.info(f"Running self-stability test for: {model_key}")
    
    return run_experiment_ii(
        run_a_model=model_key,
        run_b_model=model_key,
        perturbation_name=f"Self-test ({model_key})",
        perturbation_type="self",
        idiom_ids=idiom_ids,
        config=config,
        save_results=True,
    )


# =============================================================================
# PLACEHOLDER FOR CUSTOM PERTURBATION EXPERIMENTS
# =============================================================================

# -----------------------------------------------------------------------------
# TO ADD A NEW PERTURBATION EXPERIMENT:
# 
# 1. Add a new PerturbationConfig in config.py:
#
#    "my_perturbation": PerturbationConfig(
#        name="My Custom Perturbation",
#        perturbation_type="prompt",  # or "image", "model", "combined"
#        run_a_model="baseline_model",
#        run_b_model="perturbed_model",
#        description="Description of what this tests",
#    ),
#
# 2. Add the corresponding ModelConfig for run_b_model in config.py
#
# 3. Generate the perturbed model outputs using your perturbation strategy
#    (e.g., run the model with modified prompts or noisy images)
#
# 4. Run the experiment:
#    
#    results = run_experiment_ii_with_perturbation("my_perturbation")
#    print_stability_summary(results)
#
# -----------------------------------------------------------------------------


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment II: Score Stability Analysis"
    )
    parser.add_argument(
        "--perturbation", "-p",
        type=str,
        default=None,
        help="Perturbation config key to use"
    )
    parser.add_argument(
        "--run-a", "-a",
        type=str,
        default=None,
        help="Model key for Run A (baseline)"
    )
    parser.add_argument(
        "--run-b", "-b",
        type=str,
        default=None,
        help="Model key for Run B (perturbed)"
    )
    parser.add_argument(
        "--self-test",
        type=str,
        default=None,
        help="Run self-stability test for specified model"
    )
    parser.add_argument(
        "--simpd-variant",
        type=str,
        choices=["base", "F", "A", "tA"],
        default="base",
        help="SimPD variant to use (default: base)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of idioms to process"
    )
    parser.add_argument(
        "--list-perturbations",
        action="store_true",
        help="List available perturbation configurations"
    )
    
    args = parser.parse_args()
    
    # List perturbations if requested
    if args.list_perturbations:
        print("\nAvailable perturbation configurations:")
        perturbations = list_available_perturbations()
        if perturbations:
            for key in perturbations:
                config = get_perturbation_config(key)
                print(f"  {key}: {config.name}")
                print(f"      Type: {config.perturbation_type}")
                print(f"      Run A: {config.run_a_model}")
                print(f"      Run B: {config.run_b_model}")
        else:
            print("  No perturbations configured. Add them in config.py")
        exit(0)
    
    # Configure experiment
    config = ExperimentConfig(use_simpd_variant=args.simpd_variant)
    
    # Get idiom IDs
    idiom_ids = discover_idioms()
    if args.limit:
        idiom_ids = idiom_ids[:args.limit]
    
    # Determine what to run
    if args.self_test:
        # Self-stability test
        if args.self_test not in list_available_models():
            print(f"Error: Model '{args.self_test}' not found.")
            print(f"Available: {list_available_models()}")
            exit(1)
        
        results = run_self_stability_test(
            model_key=args.self_test,
            idiom_ids=idiom_ids,
            config=config,
        )
        print_stability_summary(results)
        
    elif args.perturbation:
        # Run with predefined perturbation
        if args.perturbation not in list_available_perturbations():
            print(f"Error: Perturbation '{args.perturbation}' not found.")
            print(f"Available: {list_available_perturbations()}")
            print("Use --list-perturbations to see all options.")
            exit(1)
        
        results = run_experiment_ii_with_perturbation(
            perturbation_key=args.perturbation,
            idiom_ids=idiom_ids,
            config=config,
        )
        print_stability_summary(results)
        
    elif args.run_a and args.run_b:
        # Custom comparison
        available = list_available_models()
        if args.run_a not in available:
            print(f"Error: Model '{args.run_a}' not found.")
            exit(1)
        if args.run_b not in available:
            print(f"Error: Model '{args.run_b}' not found.")
            exit(1)
        
        results = run_experiment_ii(
            run_a_model=args.run_a,
            run_b_model=args.run_b,
            perturbation_name=f"{args.run_a} vs {args.run_b}",
            idiom_ids=idiom_ids,
            config=config,
        )
        print_stability_summary(results)
        
    else:
        # Default: show help
        print("Experiment II: Score Stability Analysis")
        print("\nUsage options:")
        print("  --self-test MODEL    Run self-stability test")
        print("  --perturbation KEY   Run with predefined perturbation config")
        print("  --run-a A --run-b B  Compare two specific models")
        print("  --list-perturbations List available perturbation configs")
        print("\nNote: Configure perturbations in config.py PERTURBATION_CONFIGS")
        
        # If only one model available, run self-test as demo
        models = list_available_models()
        if models:
            print(f"\nRunning self-test demo with {models[0]}...")
            results = run_self_stability_test(
                model_key=models[0],
                idiom_ids=idiom_ids[:5] if idiom_ids else None,  # Limit for demo
                config=config,
            )
            print_stability_summary(results)
