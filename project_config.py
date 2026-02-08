"""
T2IMVI Project Configuration - Centralized Path Management

This module provides centralized path configuration for the entire T2IMVI project.
All path constants are derived from the project root, which is automatically detected
based on this file's location.

USAGE:
    # In any submodule, import the paths you need:
    from project_config import PROJECT_ROOT, DATA_DIR, OUTPUT_DIR
    
    # Or import all paths:
    from project_config import *

MIGRATION NOTES:
    When moving the project to a new location, no changes are needed.
    The PROJECT_ROOT is automatically detected using __file__.

Author: T2IMVI Team
"""

from pathlib import Path
from typing import Dict, Any
import sys


# =============================================================================
# PROJECT ROOT - Auto-detected from this file's location
# =============================================================================

# This file must be located at the project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add project root to Python path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# DATA DIRECTORIES
# =============================================================================

# Main data directory
DATA_DIR = PROJECT_ROOT / "data"

# Input data directories
INPUT_DIR = DATA_DIR / "input"
INPUT_IRFL_DIR = INPUT_DIR / "IRFL"
INPUT_IRFL_NON_NONE_DIR = INPUT_IRFL_DIR / "non_none"
INPUT_IRFL_MATCHED_IMAGES_DIR = INPUT_IRFL_DIR / "matched_images"

# Prompt templates directory
PROMPT_DIR = DATA_DIR / "prompt"

# Output directories
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_IRFL_DIR = OUTPUT_DIR / "IRFL"
OUTPUT_PHASE0_DIR = OUTPUT_DIR / "phase0"
OUTPUT_PHASE1_EXTRACTION_DIR = OUTPUT_DIR / "phase1_extraction"
OUTPUT_PHASE1_EXTRACTION_OUTPUT_DIR = OUTPUT_PHASE1_EXTRACTION_DIR / "output"
OUTPUT_PHASE1_EXTRACTION_CHECKPOINTS_DIR = OUTPUT_PHASE1_EXTRACTION_DIR / "checkpoints"


# =============================================================================
# SUBPROJECT DIRECTORIES
# =============================================================================

# Reliability analysis directory
RELIABILITY_ANALYSIS_DIR = PROJECT_ROOT / "reliability_analysis"
RELIABILITY_ANALYSIS_RESULTS_DIR = DATA_DIR / "reliability_analysis" / "results"

# Quantification pipeline directory
QUANTIFICATION_PIPELINE_DIR = PROJECT_ROOT / "quantification_pipeline"


# =============================================================================
# COMMON DATA FILES
# =============================================================================

# Phase 0 prompts
PHASE0_IMAGEABILITY_PROMPT = PROMPT_DIR / "phase0_imageability.txt"
PHASE0_TRANSPARENCY_PROMPT = PROMPT_DIR / "phase0_transparency.txt"

# Phase 1 prompts
PHASE1_LITERAL_EXTRACTION_PROMPT = PROMPT_DIR / "phase1_literal_extraction_specialist.txt"
PHASE1_FIGURATIVE_EXTRACTION_PROMPT = PROMPT_DIR / "phase1_figurative_extraction_specialist.txt"
PHASE1_LITERAL_VERIFIER_PROMPT = PROMPT_DIR / "phase1_literal_verifier_specialist.txt"
PHASE1_FIGURATIVE_VERIFIER_PROMPT = PROMPT_DIR / "phase1_figurative_verifier_specialist.txt"

# Phase 2 prompts
PHASE2_AEA_PROMPT = PROMPT_DIR / "phase2_aea.txt"
PHASE2_IU_RELATIONSHIPS_PROMPT = PROMPT_DIR / "phase2_iu_relationships.txt"
PHASE2_IU_WITHOUT_RELATIONSHIPS_PROMPT = PROMPT_DIR / "phase2_iu_without_relationships.txt"

# Common input files
UNIQUE_IDIOMS_FILE = INPUT_IRFL_NON_NONE_DIR / "unique_idioms.json"


# =============================================================================
# DEFAULT MODEL OUTPUT PATHS (Templates)
# =============================================================================

def get_model_output_dir(model_prefix: str = "qwen3_vl_8b", strategy: str = "T2IMVI") -> Path:
    """
    Get the output directory for a specific model/strategy combination.
    
    Args:
        model_prefix: Model identifier (e.g., "qwen3_vl_2b", "llava_1_5")
        strategy: Strategy name (e.g., "T2IMVI")
        
    Returns:
        Path to model output directory: data/output/IRFL/<model_prefix>_<strategy>/
    """
    return OUTPUT_IRFL_DIR / f"{model_prefix}_{strategy}"


def get_phase0_output_file(model_name: str = "test") -> Path:
    """
    Get the phase0 output file path for a specific model.
    
    Args:
        model_name: Model name (e.g., "qwen3-0.6b")
        
    Returns:
        Path to phase0 output: data/output/phase0/phase0_<model_name>.json
    """
    # Sanitize model name for filename
    model_safe = model_name.replace("/", "_").replace("\\", "_")
    return OUTPUT_PHASE0_DIR / f"phase0_{model_safe}.json"


# =============================================================================
# CONFIGURATION EXPORT (for reproducibility)
# =============================================================================

def get_all_paths() -> Dict[str, str]:
    """
    Get all configured paths as a dictionary.
    Useful for logging and reproducibility.
    
    Returns:
        Dictionary mapping path names to their string values
    """
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "INPUT_DIR": str(INPUT_DIR),
        "INPUT_IRFL_DIR": str(INPUT_IRFL_DIR),
        "INPUT_IRFL_NON_NONE_DIR": str(INPUT_IRFL_NON_NONE_DIR),
        "INPUT_IRFL_MATCHED_IMAGES_DIR": str(INPUT_IRFL_MATCHED_IMAGES_DIR),
        "PROMPT_DIR": str(PROMPT_DIR),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "OUTPUT_IRFL_DIR": str(OUTPUT_IRFL_DIR),
        "OUTPUT_PHASE0_DIR": str(OUTPUT_PHASE0_DIR),
        "OUTPUT_PHASE1_EXTRACTION_DIR": str(OUTPUT_PHASE1_EXTRACTION_DIR),
        "RELIABILITY_ANALYSIS_DIR": str(RELIABILITY_ANALYSIS_DIR),
        "RELIABILITY_ANALYSIS_RESULTS_DIR": str(RELIABILITY_ANALYSIS_RESULTS_DIR),
        "QUANTIFICATION_PIPELINE_DIR": str(QUANTIFICATION_PIPELINE_DIR),
    }


def print_config():
    """Print all configured paths to stdout."""
    print("=" * 70)
    print("T2IMVI Project Configuration")
    print("=" * 70)
    for name, path in get_all_paths().items():
        print(f"  {name}: {path}")
    print("=" * 70)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_paths(create_missing: bool = False) -> bool:
    """
    Validate that critical paths exist.
    
    Args:
        create_missing: If True, create missing directories
        
    Returns:
        True if all critical paths exist (or were created)
    """
    critical_dirs = [
        DATA_DIR,
        INPUT_DIR,
        OUTPUT_DIR,
        PROMPT_DIR,
    ]
    
    all_exist = True
    for path in critical_dirs:
        if not path.exists():
            if create_missing:
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created: {path}")
            else:
                print(f"Missing: {path}")
                all_exist = False
    
    return all_exist


if __name__ == "__main__":
    print_config()
    print("\nValidating paths...")
    if validate_paths():
        print("All critical paths exist.")
    else:
        print("Some paths are missing. Run with create_missing=True to create them.")
