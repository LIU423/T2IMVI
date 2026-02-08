"""
Configuration management for phase1_scoring verification pipeline.

This module centralizes all configuration, paths, and model registry.
Modify this file to add new models or change default settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Type, Any, Optional, List
import sys
import os

# Add project root to path for centralized config import
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import centralized path configuration
from project_config import (
    PROJECT_ROOT,
    DATA_DIR,
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
        "qwen3-vl-32b-instruct": ("qwen_vl_model", "Qwen3VL32BInstructModel"),
        "qwen3-vl-30b-a3b-instruct": ("qwen_vl_model", "Qwen3VL30BA3BModel"),
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
    return ["qwen3-vl-2b", "qwen3-vl-32b-instruct", "qwen3-vl-30b-a3b-instruct"]


# ============================================================================
# Scoring Configuration
# ============================================================================

@dataclass
class ScoringConfig:
    """Configuration for element verification scoring."""
    
    # Model settings
    model_name: str = "qwen3-vl-2b"
    device: str = "cuda"  # "cuda", "cpu", "auto" (multi-GPU offload)
    torch_dtype: str = "float16"  # "float16", "float32", "bfloat16"
    
    # Processing settings
    save_interval: int = 10  # Save checkpoint every N images
    max_oom_attempts: int = 3  # Max attempts for OOM errors per image
    oom_retry_backoff_seconds: float = 1.0  # Sleep between OOM retries
    continue_on_error: bool = True  # Continue processing after a failed image
    
    # Resume behavior
    resume_from_checkpoint: bool = True
    
    # Test mode
    test_mode: bool = False  # If True, only process first N idioms
    test_n_idioms: int = 1
    test_n_images: int = 2  # Process only N images per idiom in test mode
    
    # Scope settings
    idiom_ids: Optional[List[int]] = None  # If None, process all idioms
    # Optional filter for exact failed-image reruns: {idiom_id: [image_num, ...]}
    target_image_nums_by_idiom: Optional[Dict[int, List[int]]] = None
    
    # Paths (use defaults from module level)
    input_images_dir: Path = field(default_factory=lambda: INPUT_IMAGES_DIR)
    extraction_output_dir: Path = field(default_factory=lambda: EXTRACTION_OUTPUT_DIR)
    figurative_prompt_file: Path = field(default_factory=lambda: FIGURATIVE_PROMPT_FILE)
    literal_prompt_file: Path = field(default_factory=lambda: LITERAL_PROMPT_FILE)
    output_base_dir: Path = field(default_factory=lambda: OUTPUT_BASE_DIR)
    failed_items_dir: Path = field(default_factory=lambda: DATA_DIR / "phase1_scoring" / "failed_items")
    
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
        suffix = os.environ.get("PHASE1_CHECKPOINT_SUFFIX", "").strip()
        if suffix:
            return self.get_output_dir() / f"checkpoint_{suffix}.json"
        return self.get_output_dir() / "checkpoint.json"

    def get_failed_items_file(self) -> Path:
        """Get failed-items log file path under data/."""
        model_suffix = self.model_name.replace("-", "_")
        model_dir = self.failed_items_dir / model_suffix
        suffix = (
            os.environ.get("PHASE1_FAILED_SUFFIX", "").strip()
            or os.environ.get("PHASE1_CHECKPOINT_SUFFIX", "").strip()
        )
        if suffix:
            return model_dir / f"failed_items_{suffix}.jsonl"
        return model_dir / "failed_items.jsonl"


# ============================================================================
# Default Configuration Instance
# ============================================================================

def get_default_config() -> ScoringConfig:
    """Get default scoring configuration."""
    return ScoringConfig()
