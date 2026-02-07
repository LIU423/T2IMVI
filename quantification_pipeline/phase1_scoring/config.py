"""
Configuration management for phase1_scoring verification pipeline.

This module centralizes all configuration, paths, and model registry.
Modify this file to add new models or change default settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Type, Any, Optional, List
import sys

# Add project root to path for centralized config import
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import centralized path configuration
from project_config import (
    PROJECT_ROOT,
    INPUT_IRFL_MATCHED_IMAGES_DIR,
    OUTPUT_PHASE1_EXTRACTION_OUTPUT_DIR,
    PROMPT_DIR,
    PHASE1_FIGURATIVE_VERIFIER_PROMPT,
    PHASE1_LITERAL_VERIFIER_PROMPT,
    OUTPUT_IRFL_DIR,
)


# ============================================================================
# Path Configuration (Derived from centralized project_config)
# ============================================================================

# Input paths
INPUT_IMAGES_DIR = INPUT_IRFL_MATCHED_IMAGES_DIR
EXTRACTION_OUTPUT_DIR = OUTPUT_PHASE1_EXTRACTION_OUTPUT_DIR

# Prompt files
FIGURATIVE_PROMPT_FILE = PHASE1_FIGURATIVE_VERIFIER_PROMPT
LITERAL_PROMPT_FILE = PHASE1_LITERAL_VERIFIER_PROMPT

# Output paths
OUTPUT_BASE_DIR = OUTPUT_IRFL_DIR


# ============================================================================
# Model Registry
# ============================================================================

def get_model_class(model_name: str) -> Type:
    """
    Get model class by name. Add new models here.
    
    To add a new model:
    1. Create a new file implementing BaseVerifierModel
    2. Add an entry to MODEL_REGISTRY below
    
    Args:
        model_name: Name of the model (e.g., "qwen3-vl-2b")
        
    Returns:
        Model class type
        
    Raises:
        ValueError: If model name is not found in registry
    """
    MODEL_REGISTRY = {
        "qwen3-vl-2b": ("qwen_vl_model", "Qwen3VLModel"),
        # Add new models here:
        # "llava-7b": ("llava_model", "LlavaModel"),
        # "internvl-2b": ("internvl_model", "InternVLModel"),
    }
    
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    module_name, class_name = MODEL_REGISTRY[model_name]
    
    # Dynamic import
    import importlib
    module = importlib.import_module(f".{module_name}", package="quantification_pipeline.phase1_scoring.models")
    return getattr(module, class_name)


def get_available_models() -> List[str]:
    """Get list of available model names."""
    return ["qwen3-vl-2b"]


# ============================================================================
# Scoring Configuration
# ============================================================================

@dataclass
class ScoringConfig:
    """Configuration for element verification scoring."""
    
    # Model settings
    model_name: str = "qwen3-vl-2b"
    device: str = "cuda"
    torch_dtype: str = "float16"  # "float16", "float32", "bfloat16"
    
    # Processing settings
    save_interval: int = 10  # Save checkpoint every N images
    
    # Resume behavior
    resume_from_checkpoint: bool = True
    
    # Test mode
    test_mode: bool = False  # If True, only process first N idioms
    test_n_idioms: int = 1
    test_n_images: int = 2  # Process only N images per idiom in test mode
    
    # Scope settings
    idiom_ids: Optional[List[int]] = None  # If None, process all idioms
    
    # Paths (use defaults from module level)
    input_images_dir: Path = field(default_factory=lambda: INPUT_IMAGES_DIR)
    extraction_output_dir: Path = field(default_factory=lambda: EXTRACTION_OUTPUT_DIR)
    figurative_prompt_file: Path = field(default_factory=lambda: FIGURATIVE_PROMPT_FILE)
    literal_prompt_file: Path = field(default_factory=lambda: LITERAL_PROMPT_FILE)
    output_base_dir: Path = field(default_factory=lambda: OUTPUT_BASE_DIR)
    
    def get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        import torch
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)
    
    def get_output_dir(self) -> Path:
        """Get model-specific output directory."""
        # Format: <model>_T2IMVI
        model_suffix = self.model_name.replace("-", "_")
        return self.output_base_dir / f"{model_suffix}_T2IMVI"
    
    def get_checkpoint_file(self) -> Path:
        """Get checkpoint file path."""
        return self.get_output_dir() / "checkpoint.json"


# ============================================================================
# Default Configuration Instance
# ============================================================================

def get_default_config() -> ScoringConfig:
    """Get default scoring configuration."""
    return ScoringConfig()
