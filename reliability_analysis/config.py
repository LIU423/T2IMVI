"""
Configuration module for T2IMVI Reliability Analysis experiments.

This module provides centralized configuration management with placeholders
for model/strategy substitution. Modify the MODEL_CONFIGS to add new models
or strategies for comparison.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import sys

# Add project root to path for centralized config import
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import centralized path configuration
from project_config import (
    DATA_DIR as _DATA_DIR,
    INPUT_IRFL_MATCHED_IMAGES_DIR,
    OUTPUT_IRFL_DIR,
    RELIABILITY_ANALYSIS_RESULTS_DIR,
)


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths - derived from centralized project_config
BASE_DATA_PATH = _DATA_DIR
INPUT_PATH = INPUT_IRFL_MATCHED_IMAGES_DIR
OUTPUT_PATH = OUTPUT_IRFL_DIR

# Results output path
RESULTS_PATH = RELIABILITY_ANALYSIS_RESULTS_DIR


# =============================================================================
# MODEL/STRATEGY CONFIGURATION - MODIFY THIS SECTION TO ADD NEW MODELS
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model/strategy.
    
    Attributes:
        name: Human-readable name for the model
        strategy_id: Directory name in the output path (e.g., 'qwen3_vl_2b_T2IMVI')
        description: Optional description of the model/strategy
        score_field: Which field to use for ranking (default: 'figurative_score')
        additional_params: Any additional parameters specific to this model
    """
    name: str
    strategy_id: str
    description: str = ""
    score_field: str = "figurative_score"
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_output_path(self) -> Path:
        """Get the output path for this model's results."""
        return OUTPUT_PATH / self.strategy_id


# =============================================================================
# REGISTERED MODELS - ADD YOUR MODELS HERE
# =============================================================================

# Placeholder model configurations - ADD NEW MODELS BY EXTENDING THIS DICTIONARY
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # -------------------------------------------------------------------------
    # BASELINE MODEL (currently available)
    # -------------------------------------------------------------------------
    "qwen3_vl_2b_T2IMVI": ModelConfig(
        name="Qwen3-VL-2B T2IMVI",
        strategy_id="qwen3_vl_2b_T2IMVI",
        description="Qwen3 Vision-Language 2B model with T2IMVI strategy",
        score_field="figurative_score",
    ),
    "qwen3_vl_30b_a3b_instruct_T2IMVI": ModelConfig(
        name="Qwen3-VL-30B-A3B-Instruct T2IMVI",
        strategy_id="qwen3_vl_30b_a3b_instruct_T2IMVI",
        description="Qwen3-VL-30B-A3B-Instruct model with T2IMVI strategy",
        score_field="figurative_score",
    ),
    
    # -------------------------------------------------------------------------
    # PLACEHOLDER MODELS - UNCOMMENT AND MODIFY AS NEEDED
    # -------------------------------------------------------------------------
    
    # "model_variant_1": ModelConfig(
    #     name="Model Variant 1",
    #     strategy_id="<model_name>_<strategy>",  # e.g., "llava_v1.5_T2IMVI"
    #     description="Description of this model variant",
    #     score_field="figurative_score",
    # ),
    
    # "model_variant_2": ModelConfig(
    #     name="Model Variant 2", 
    #     strategy_id="<model_name>_<strategy>",
    #     description="Description of this model variant",
    #     score_field="figurative_score",
    # ),
    
    # "model_with_custom_score": ModelConfig(
    #     name="Model with Custom Score",
    #     strategy_id="<model_name>_<strategy>",
    #     description="Model using a different score field",
    #     score_field="S_fid",  # Use a different score field
    # ),
}


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for experiments.
    
    Attributes:
        rbo_p: RBO top-weighting parameter (0 < p < 1, higher = more top-weight)
        simpd_beta: SimPD F-measure beta parameter for SimPD-A variant
        use_simpd_variant: Which SimPD variant to use ('base', 'F', 'A', 'tA')
    """
    # RBO Configuration
    rbo_p: float = 0.9  # Top-weighting parameter for RBO
    
    # SimPD Configuration
    simpd_beta: float = 1.0  # Beta for F-measure in SimPD-A
    use_simpd_variant: str = "base"  # Options: 'base', 'F', 'A', 'tA'
    

# Default experiment configuration
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


# =============================================================================
# EXPERIMENT II: PERTURBATION STRATEGIES - PLACEHOLDER FOR RUN B
# =============================================================================

@dataclass
class PerturbationConfig:
    """Configuration for perturbation strategies in Experiment II.
    
    This defines how Run B (perturbed run) should differ from Run A (baseline).
    
    Attributes:
        name: Human-readable name for this perturbation
        perturbation_type: Type of perturbation ('model', 'prompt', 'image', 'combined')
        run_a_model: Model config key for Run A (baseline)
        run_b_model: Model config key for Run B (perturbed)
        description: Description of what this perturbation tests
    """
    name: str
    perturbation_type: str
    run_a_model: str
    run_b_model: str
    description: str = ""


# Placeholder perturbation configurations - ADD YOUR PERTURBATION STRATEGIES HERE
PERTURBATION_CONFIGS: Dict[str, PerturbationConfig] = {
    # -------------------------------------------------------------------------
    # PLACEHOLDER PERTURBATION STRATEGIES
    # -------------------------------------------------------------------------
    
    # Example: Compare same model with different prompts
    # "prompt_synonym": PerturbationConfig(
    #     name="Prompt Synonym Substitution",
    #     perturbation_type="prompt",
    #     run_a_model="qwen3_vl_2b_T2IMVI",
    #     run_b_model="qwen3_vl_2b_T2IMVI_synonym",  # Need to add this model config
    #     description="Test stability with synonym substitution in prompts",
    # ),
    
    # Example: Compare same model with image noise
    # "image_noise": PerturbationConfig(
    #     name="Gaussian Noise Injection",
    #     perturbation_type="image",
    #     run_a_model="qwen3_vl_2b_T2IMVI",
    #     run_b_model="qwen3_vl_2b_T2IMVI_noisy",  # Need to add this model config
    #     description="Test stability with Gaussian noise injected into images",
    # ),
    
    # Example: Compare different models
    # "model_comparison": PerturbationConfig(
    #     name="Model A vs Model B",
    #     perturbation_type="model",
    #     run_a_model="qwen3_vl_2b_T2IMVI",
    #     run_b_model="llava_v1.5_T2IMVI",
    #     description="Compare stability across different VLM architectures",
    # ),
}


# =============================================================================
# ANNOTATION SCORING CONFIGURATION (NUMERICAL)
# =============================================================================

# Annotation label scores for numerical scoring
# Priority: Figurative+Literal > Figurative > Literal > Partial Literal > None
ANNOTATION_SCORE_WEIGHTS: Dict[str, int] = {
    "Figurative+Literal": 20,
    "Figurative": 15,
    "Literal": 10,
    "Partial Literal": 5,
    "None": 0,
}


@dataclass
class ScoringConfig:
    """Configuration for numerical annotation scoring.
    
    Each annotation label is assigned a numerical weight.
    An image's human score = sum of weights for all 5 annotations.
    Score range: 0 (all None) to 100 (all Figurative+Literal).
    
    Attributes:
        weights: Mapping from annotation label to numerical score
        max_score: Maximum possible score (5 * max_weight)
        min_score: Minimum possible score (5 * min_weight)
    """
    weights: Dict[str, int] = field(default_factory=lambda: ANNOTATION_SCORE_WEIGHTS.copy())
    
    @property
    def max_score(self) -> int:
        """Maximum possible score (5 annotations * max weight)."""
        return 5 * max(self.weights.values())
    
    @property
    def min_score(self) -> int:
        """Minimum possible score (5 annotations * min weight)."""
        return 5 * min(self.weights.values())
    
    def get_label_score(self, label: str) -> int:
        """Get score for a single annotation label.
        
        Args:
            label: The annotation label
            
        Returns:
            The numerical score for this label
        """
        return self.weights.get(label, 0)
    
    def calculate_image_score(self, annotations: List[str]) -> int:
        """Calculate total score for an image from its annotations.
        
        Args:
            annotations: List of annotation labels (typically 5)
            
        Returns:
            Sum of scores for all annotations
        """
        return sum(self.get_label_score(label) for label in annotations)


# Default scoring configuration
DEFAULT_SCORING_CONFIG = ScoringConfig()


# =============================================================================
# ANNOTATION CLASSIFICATION CONFIGURATION (CATEGORICAL - LEGACY)
# =============================================================================

@dataclass
class ClassificationConfig:
    """Configuration for annotation classification into I_fig/I_lit/I_rand.
    
    Priority ordering (user specified):
    Figurative+Literal ≈ Figurative > Literal ≈ Partial Literal > None
    
    Attributes:
        fig_labels: Labels that count as "figurative" for I_fig classification
        lit_labels: Labels that count as "literal" for I_lit classification
        majority_threshold: Minimum count to be considered majority (default: 3 out of 5)
    """
    # Labels that contribute to I_fig classification
    fig_labels: List[str] = field(default_factory=lambda: ["Figurative", "Figurative+Literal"])
    
    # Labels that contribute to I_lit classification
    lit_labels: List[str] = field(default_factory=lambda: ["Literal"])
    
    # Minimum count for majority (3 out of 5 annotators)
    majority_threshold: int = 3
    
    # All possible annotation values (for validation)
    all_labels: List[str] = field(default_factory=lambda: [
        "None",
        "Figurative",
        "Literal",
        "Figurative+Literal",
        "Partial Literal"
    ])


# Default classification configuration
DEFAULT_CLASSIFICATION_CONFIG = ClassificationConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_config(model_key: str) -> ModelConfig:
    """Get a model configuration by key.
    
    Args:
        model_key: Key in MODEL_CONFIGS dictionary
        
    Returns:
        ModelConfig for the specified model
        
    Raises:
        ValueError: If model_key is not found
    """
    if model_key not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Model '{model_key}' not found. Available models: {available}"
        )
    return MODEL_CONFIGS[model_key]


def get_perturbation_config(perturbation_key: str) -> PerturbationConfig:
    """Get a perturbation configuration by key.
    
    Args:
        perturbation_key: Key in PERTURBATION_CONFIGS dictionary
        
    Returns:
        PerturbationConfig for the specified perturbation
        
    Raises:
        ValueError: If perturbation_key is not found
    """
    if perturbation_key not in PERTURBATION_CONFIGS:
        available = list(PERTURBATION_CONFIGS.keys())
        raise ValueError(
            f"Perturbation '{perturbation_key}' not found. Available: {available}"
        )
    return PERTURBATION_CONFIGS[perturbation_key]


def list_available_models() -> List[str]:
    """List all available model configuration keys."""
    return list(MODEL_CONFIGS.keys())


def list_available_perturbations() -> List[str]:
    """List all available perturbation configuration keys."""
    return list(PERTURBATION_CONFIGS.keys())


def get_annotation_path(idiom_id: int, image_id: int) -> Path:
    """Get the path to an annotation file.
    
    Args:
        idiom_id: The idiom ID
        image_id: The image ID
        
    Returns:
        Path to the annotation JSON file
    """
    return INPUT_PATH / str(idiom_id) / f"{image_id}.json"


def get_model_output_path(
    model_key: str, 
    idiom_id: int, 
    image_id: int
) -> Path:
    """Get the path to a model output file.
    
    Args:
        model_key: Key in MODEL_CONFIGS dictionary
        idiom_id: The idiom ID
        image_id: The image ID
        
    Returns:
        Path to the model output JSON file
    """
    config = get_model_config(model_key)
    return (
        config.get_output_path() / 
        f"idiom_{idiom_id}" / 
        f"image_{image_id}" / 
        "total_score.json"
    )


# =============================================================================
# CONFIG EXPORT FOR REPRODUCIBILITY
# =============================================================================

def export_config(output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Export current configuration as a dictionary for reproducibility.
    
    Args:
        output_path: If provided, save config to this JSON file
        
    Returns:
        Dictionary containing all configuration
    """
    config = {
        "paths": {
            "base_data_path": str(BASE_DATA_PATH),
            "input_path": str(INPUT_PATH),
            "output_path": str(OUTPUT_PATH),
            "results_path": str(RESULTS_PATH),
        },
        "models": {
            key: {
                "name": cfg.name,
                "strategy_id": cfg.strategy_id,
                "description": cfg.description,
                "score_field": cfg.score_field,
            }
            for key, cfg in MODEL_CONFIGS.items()
        },
        "experiment": {
            "rbo_p": DEFAULT_EXPERIMENT_CONFIG.rbo_p,
            "simpd_beta": DEFAULT_EXPERIMENT_CONFIG.simpd_beta,
            "simpd_variant": DEFAULT_EXPERIMENT_CONFIG.use_simpd_variant,
        },
        "classification": {
            "fig_labels": DEFAULT_CLASSIFICATION_CONFIG.fig_labels,
            "lit_labels": DEFAULT_CLASSIFICATION_CONFIG.lit_labels,
            "majority_threshold": DEFAULT_CLASSIFICATION_CONFIG.majority_threshold,
        },
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config


if __name__ == "__main__":
    # Print current configuration
    print("=" * 60)
    print("T2IMVI Reliability Analysis - Configuration")
    print("=" * 60)
    print(f"\nInput Path: {INPUT_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"\nAvailable Models: {list_available_models()}")
    print(f"Available Perturbations: {list_available_perturbations()}")
    print(f"\nRBO p-parameter: {DEFAULT_EXPERIMENT_CONFIG.rbo_p}")
    print(f"Classification threshold: {DEFAULT_CLASSIFICATION_CONFIG.majority_threshold}/5")
