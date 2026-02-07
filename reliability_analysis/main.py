"""
T2IMVI Reliability Analysis - Main Entry Point

This module provides a unified entry point to run both reliability experiments:
- Experiment I:  Ranking Alignment (RBO metric)
- Experiment II: Score Stability (SimPD metric)

Usage:
    python main.py --help                     # Show help
    python main.py exp1 --model MODEL         # Run Experiment I
    python main.py exp2 --self-test MODEL     # Run Experiment II self-test
    python main.py all --model MODEL          # Run both experiments
    python main.py verify                     # Verify data loading
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    INPUT_PATH,
    OUTPUT_PATH,
    RESULTS_PATH,
    list_available_models,
    list_available_perturbations,
    export_config,
    get_model_config,
    DEFAULT_EXPERIMENT_CONFIG,
    ExperimentConfig,
)
from data_loader import (
    discover_idioms,
    discover_images_for_idiom,
    load_annotation,
    load_model_output,
    load_combined_data_for_idiom,
)
from classification import (
    classify_annotation,
    classify_idiom_images,
    ImageClass,
)
from experiment1_ranking import (
    run_experiment_i,
    compare_models,
    print_experiment_summary,
    generate_latex_table,
)
from experiment2_stability import (
    run_experiment_ii,
    run_self_stability_test,
    run_experiment_ii_with_perturbation,
    print_stability_summary,
    generate_stability_latex_table,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_data_loading(model_key: str, limit: int = 3) -> bool:
    """Verify that data loading works correctly.
    
    Args:
        model_key: Model to verify
        limit: Number of idioms to check
        
    Returns:
        True if verification passes
    """
    print("\n" + "=" * 70)
    print("DATA LOADING VERIFICATION")
    print("=" * 70)
    
    # Check paths exist
    print(f"\n[1] Checking paths...")
    print(f"    Input path: {INPUT_PATH}")
    print(f"    Output path: {OUTPUT_PATH}")
    
    if not INPUT_PATH.exists():
        print(f"    ERROR: Input path does not exist!")
        return False
    print(f"    Input path exists: OK")
    
    if not OUTPUT_PATH.exists():
        print(f"    ERROR: Output path does not exist!")
        return False
    print(f"    Output path exists: OK")
    
    # Discover idioms
    print(f"\n[2] Discovering idioms...")
    idiom_ids = discover_idioms()
    if not idiom_ids:
        print("    ERROR: No idioms found!")
        return False
    print(f"    Found {len(idiom_ids)} idioms: {idiom_ids[:5]}...")
    
    # Check model output path
    print(f"\n[3] Checking model '{model_key}'...")
    model_config = get_model_config(model_key)
    model_output_path = model_config.get_output_path()
    print(f"    Model output path: {model_output_path}")
    
    if not model_output_path.exists():
        print(f"    ERROR: Model output path does not exist!")
        return False
    print(f"    Model output path exists: OK")
    
    # Load sample data
    print(f"\n[4] Loading sample data (first {limit} idioms)...")
    success_count = 0
    
    for idiom_id in idiom_ids[:limit]:
        image_ids = discover_images_for_idiom(idiom_id)
        if not image_ids:
            print(f"    Idiom {idiom_id}: No images found")
            continue
            
        # Load first image
        first_image_id = image_ids[0]
        
        # Load annotation
        annotation = load_annotation(idiom_id, first_image_id)
        if annotation:
            print(f"    Idiom {idiom_id}, Image {first_image_id}:")
            print(f"        Annotation: {annotation.annotations[:3]}... -> {annotation.category}")
            
            # Classify
            result = classify_annotation(annotation)
            print(f"        Classification: {result.image_class.value}")
        else:
            print(f"    Idiom {idiom_id}, Image {first_image_id}: Annotation load failed")
            continue
        
        # Load model output
        model_output = load_model_output(model_key, idiom_id, first_image_id)
        if model_output:
            print(f"        Model scores: fig={model_output.figurative_score:.4f}, lit={model_output.literal_score:.4f}")
            success_count += 1
        else:
            print(f"        Model output: Not found")
    
    # Summary
    print(f"\n[5] Verification Summary")
    print(f"    Idioms checked: {min(limit, len(idiom_ids))}")
    print(f"    Successful loads: {success_count}")
    
    if success_count > 0:
        print("\n    VERIFICATION PASSED")
        return True
    else:
        print("\n    VERIFICATION FAILED - No data could be loaded")
        return False


def verify_classification(model_key: str, idiom_id: Optional[int] = None) -> None:
    """Verify classification logic for a specific idiom.
    
    Args:
        model_key: Model to use
        idiom_id: Specific idiom to check (default: first available)
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION VERIFICATION")
    print("=" * 70)
    
    idiom_ids = discover_idioms()
    if idiom_id is None:
        idiom_id = idiom_ids[0]
    
    print(f"\nAnalyzing idiom {idiom_id}...")
    
    # Load combined data
    image_data = load_combined_data_for_idiom(idiom_id, [model_key])
    
    if not image_data:
        print("  No data found for this idiom")
        return
    
    # Classify
    summary = classify_idiom_images(image_data)
    
    print(f"\n  Total images: {summary.total_count}")
    print(f"  I_fig count:  {summary.fig_count}")
    print(f"  I_lit count:  {summary.lit_count}")
    print(f"  I_rand count: {summary.rand_count}")
    
    print("\n  Detailed classifications:")
    for img_id, img_class in list(summary.classifications.items())[:10]:
        # Find the image data
        img_data = next((d for d in image_data if d.image_id == img_id), None)
        if img_data and img_data.annotation:
            print(f"    Image {img_id}: {img_class.value} (annotations: {img_data.annotation.annotations})")


# =============================================================================
# MAIN COMMANDS
# =============================================================================

def cmd_experiment1(args) -> None:
    """Run Experiment I: Ranking Alignment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT I: RANKING ALIGNMENT")
    print("=" * 70)
    
    # Configure
    config = ExperimentConfig(rbo_p=args.rbo_p)
    
    # Get idiom IDs
    idiom_ids = discover_idioms()
    if args.limit:
        idiom_ids = idiom_ids[:args.limit]
    
    # Determine models
    available = list_available_models()
    
    if args.all_models:
        model_keys = available
    elif args.model:
        if args.model not in available:
            print(f"Error: Model '{args.model}' not found.")
            print(f"Available: {available}")
            return
        model_keys = [args.model]
    else:
        model_keys = [available[0]] if available else []
    
    if not model_keys:
        print("No models configured. Add models in config.py MODEL_CONFIGS.")
        return
    
    # Run
    if len(model_keys) == 1:
        results = run_experiment_i(
            model_key=model_keys[0],
            idiom_ids=idiom_ids,
            config=config,
        )
        print_experiment_summary(results)
        
        if args.latex:
            print("\n--- LaTeX Table ---")
            print(generate_latex_table(results))
    else:
        compare_models(model_keys, idiom_ids, config)


def cmd_experiment2(args) -> None:
    """Run Experiment II: Score Stability."""
    print("\n" + "=" * 70)
    print("EXPERIMENT II: SCORE STABILITY")
    print("=" * 70)
    
    # Configure
    config = ExperimentConfig(use_simpd_variant=args.simpd_variant)
    
    # Get idiom IDs
    idiom_ids = discover_idioms()
    if args.limit:
        idiom_ids = idiom_ids[:args.limit]
    
    # Run based on mode
    if args.self_test:
        # Self-stability test
        available = list_available_models()
        if args.self_test not in available:
            print(f"Error: Model '{args.self_test}' not found.")
            print(f"Available: {available}")
            return
        
        results = run_self_stability_test(
            model_key=args.self_test,
            idiom_ids=idiom_ids,
            config=config,
        )
        print_stability_summary(results)
        
    elif args.perturbation:
        # Predefined perturbation
        available = list_available_perturbations()
        if args.perturbation not in available:
            print(f"Error: Perturbation '{args.perturbation}' not found.")
            print(f"Available: {available}")
            print("Add perturbations in config.py PERTURBATION_CONFIGS.")
            return
        
        results = run_experiment_ii_with_perturbation(
            perturbation_key=args.perturbation,
            idiom_ids=idiom_ids,
            config=config,
        )
        print_stability_summary(results)
        
    elif args.run_a and args.run_b:
        # Custom comparison
        available = list_available_models()
        if args.run_a not in available or args.run_b not in available:
            print("Error: Model not found.")
            print(f"Available: {available}")
            return
        
        results = run_experiment_ii(
            run_a_model=args.run_a,
            run_b_model=args.run_b,
            perturbation_name=f"{args.run_a} vs {args.run_b}",
            idiom_ids=idiom_ids,
            config=config,
        )
        print_stability_summary(results)
        
    else:
        # Default: self-test with first available model
        available = list_available_models()
        if available:
            print(f"Running self-test with: {available[0]}")
            results = run_self_stability_test(
                model_key=available[0],
                idiom_ids=idiom_ids[:5],  # Limited for demo
                config=config,
            )
            print_stability_summary(results)
        else:
            print("No models configured.")
    
    if args.latex and 'results' in locals():
        print("\n--- LaTeX Table ---")
        print(generate_stability_latex_table(results))


def cmd_run_all(args) -> None:
    """Run both experiments."""
    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    
    # Configure
    exp_config = ExperimentConfig(
        rbo_p=args.rbo_p,
        use_simpd_variant=args.simpd_variant,
    )
    
    # Get idiom IDs
    idiom_ids = discover_idioms()
    if args.limit:
        idiom_ids = idiom_ids[:args.limit]
    
    # Determine model
    available = list_available_models()
    if args.model:
        if args.model not in available:
            print(f"Error: Model '{args.model}' not found.")
            return
        model_key = args.model
    else:
        model_key = available[0] if available else None
    
    if not model_key:
        print("No models configured.")
        return
    
    # Run Experiment I
    print("\n>>> Running Experiment I: Ranking Alignment <<<\n")
    exp1_results = run_experiment_i(
        model_key=model_key,
        idiom_ids=idiom_ids,
        config=exp_config,
    )
    print_experiment_summary(exp1_results)
    
    # Run Experiment II (self-test)
    print("\n>>> Running Experiment II: Score Stability (Self-Test) <<<\n")
    exp2_results = run_self_stability_test(
        model_key=model_key,
        idiom_ids=idiom_ids,
        config=exp_config,
    )
    print_stability_summary(exp2_results)
    
    # Combined summary
    print("\n" + "=" * 70)
    print("COMBINED RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nModel: {model_key}")
    print(f"Idioms processed: {len(idiom_ids)}")
    print(f"\nExperiment I  (Ranking Alignment): Mean RBO = {exp1_results.aggregate_stats.get('mean_rbo', 0):.4f}")
    print(f"Experiment II (Score Stability):   Mean SimPD = {exp2_results.aggregate_stats.get('mean_simpd', 0):.4f}")


def cmd_verify(args) -> None:
    """Verify data loading and setup."""
    # Determine model
    available = list_available_models()
    model_key = args.model if args.model else (available[0] if available else None)
    
    if not model_key:
        print("No models configured. Add models in config.py MODEL_CONFIGS.")
        return
    
    # Run verification
    success = verify_data_loading(model_key, limit=args.limit or 3)
    
    if args.classify and success:
        verify_classification(model_key, args.idiom)


def cmd_config(args) -> None:
    """Show or export configuration."""
    print("\n" + "=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    
    print(f"\nInput Path:  {INPUT_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Results Path: {RESULTS_PATH}")
    
    print(f"\nAvailable Models ({len(list_available_models())}):")
    for key in list_available_models():
        config = get_model_config(key)
        print(f"  - {key}: {config.name}")
    
    perturbations = list_available_perturbations()
    print(f"\nAvailable Perturbations ({len(perturbations)}):")
    if perturbations:
        for key in perturbations:
            print(f"  - {key}")
    else:
        print("  (none configured - add in config.py PERTURBATION_CONFIGS)")
    
    print(f"\nExperiment Settings:")
    print(f"  RBO p-parameter: {DEFAULT_EXPERIMENT_CONFIG.rbo_p}")
    print(f"  SimPD variant: {DEFAULT_EXPERIMENT_CONFIG.use_simpd_variant}")
    
    if args.export:
        export_path = RESULTS_PATH / "config_export.json"
        export_config(export_path)
        print(f"\nConfiguration exported to: {export_path}")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="T2IMVI Reliability Analysis Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py verify                        # Verify data loading
    python main.py exp1 -m qwen3_vl_2b_T2IMVI    # Run Experiment I
    python main.py exp2 --self-test qwen3_vl_2b_T2IMVI  # Run Experiment II self-test
    python main.py all -m qwen3_vl_2b_T2IMVI     # Run both experiments
    python main.py config --export               # Export configuration
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ---------------------------------------------------------------------
    # exp1 - Experiment I
    # ---------------------------------------------------------------------
    exp1_parser = subparsers.add_parser("exp1", help="Run Experiment I: Ranking Alignment")
    exp1_parser.add_argument("-m", "--model", type=str, help="Model key to evaluate")
    exp1_parser.add_argument("-a", "--all-models", action="store_true", help="Evaluate all models")
    exp1_parser.add_argument("--rbo-p", type=float, default=0.9, help="RBO p-parameter (default: 0.9)")
    exp1_parser.add_argument("--limit", type=int, help="Limit number of idioms")
    exp1_parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    
    # ---------------------------------------------------------------------
    # exp2 - Experiment II
    # ---------------------------------------------------------------------
    exp2_parser = subparsers.add_parser("exp2", help="Run Experiment II: Score Stability")
    exp2_parser.add_argument("--self-test", type=str, help="Run self-stability test for model")
    exp2_parser.add_argument("-p", "--perturbation", type=str, help="Perturbation config key")
    exp2_parser.add_argument("--run-a", type=str, help="Model for Run A (baseline)")
    exp2_parser.add_argument("--run-b", type=str, help="Model for Run B (perturbed)")
    exp2_parser.add_argument("--simpd-variant", type=str, default="base",
                            choices=["base", "F", "A", "tA"], help="SimPD variant")
    exp2_parser.add_argument("--limit", type=int, help="Limit number of idioms")
    exp2_parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    
    # ---------------------------------------------------------------------
    # all - Run both experiments
    # ---------------------------------------------------------------------
    all_parser = subparsers.add_parser("all", help="Run both experiments")
    all_parser.add_argument("-m", "--model", type=str, help="Model key to evaluate")
    all_parser.add_argument("--rbo-p", type=float, default=0.9, help="RBO p-parameter")
    all_parser.add_argument("--simpd-variant", type=str, default="base",
                           choices=["base", "F", "A", "tA"], help="SimPD variant")
    all_parser.add_argument("--limit", type=int, help="Limit number of idioms")
    
    # ---------------------------------------------------------------------
    # verify - Verify data loading
    # ---------------------------------------------------------------------
    verify_parser = subparsers.add_parser("verify", help="Verify data loading and setup")
    verify_parser.add_argument("-m", "--model", type=str, help="Model to verify")
    verify_parser.add_argument("--limit", type=int, default=3, help="Number of idioms to check")
    verify_parser.add_argument("--classify", action="store_true", help="Also verify classification")
    verify_parser.add_argument("--idiom", type=int, help="Specific idiom for classification check")
    
    # ---------------------------------------------------------------------
    # config - Show/export configuration
    # ---------------------------------------------------------------------
    config_parser = subparsers.add_parser("config", help="Show or export configuration")
    config_parser.add_argument("--export", action="store_true", help="Export config to JSON")
    
    return parser


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\n" + "-" * 70)
        print("Quick Start:")
        print("  python main.py verify                    # Check data loading")
        print("  python main.py exp1 -m qwen3_vl_2b_T2IMVI  # Run Experiment I")
        print("  python main.py exp2 --self-test qwen3_vl_2b_T2IMVI  # Self-test")
        return
    
    # Dispatch to command handler
    if args.command == "exp1":
        cmd_experiment1(args)
    elif args.command == "exp2":
        cmd_experiment2(args)
    elif args.command == "all":
        cmd_run_all(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "config":
        cmd_config(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
