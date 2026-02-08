"""
Configuration management for transparency evaluation pipeline.

This module centralizes all configuration, paths, and model registry.
Modify this file to add new models or change default settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Type, Any
import sys

# Add project root to path for centralized config import
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import centralized path configuration
from project_config import (
    PROJECT_ROOT,
    INPUT_IRFL_NON_NONE_DIR,
    OUTPUT_PHASE0_DIR,
    PHASE0_TRANSPARENCY_PROMPT,
)


# ============================================================================
# Path Configuration (Derived from centralized project_config)
# ============================================================================

# Input paths
INPUT_DIR = INPUT_IRFL_NON_NONE_DIR
IDIOMS_FILE = INPUT_DIR / "unique_idioms.json"
PROMPT_FILE = PHASE0_TRANSPARENCY_PROMPT

# Output paths
OUTPUT_DIR = OUTPUT_PHASE0_DIR
OUTPUT_FILE = OUTPUT_DIR / "transparency_results.json"
CHECKPOINT_FILE = OUTPUT_DIR / "transparency_checkpoint.json"


# ============================================================================
# Model Registry
# ============================================================================

# Lazy import to avoid circular dependencies
def get_model_class(model_name: str) -> Type:
    """
    Get model class by name. Add new models here.
    
    To add a new model:
    1. Create a new file implementing BaseTransparencyModel
    2. Add an entry to MODEL_REGISTRY below
    """
    MODEL_REGISTRY = {
        "qwen3-0.6b": ("qwen_model", "Qwen3Model"),
        "qwen3-30b-a3b-instruct-2507": ("qwen_model", "Qwen3_30B_A3B_Instruct_2507"),
        "Qwen/Qwen3-30B-A3B-Instruct-2507": ("qwen_model", "Qwen3_30B_A3B_Instruct_2507"),
        # Add new models here:
        # "llama-7b": ("llama_model", "LlamaModel"),
        # "mistral-7b": ("mistral_model", "MistralModel"),
    }
    
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    module_name, class_name = MODEL_REGISTRY[model_name]
    
    # Dynamic import
    import importlib
    module = importlib.import_module(f".{module_name}", package=__package__)
    return getattr(module, class_name)


# ============================================================================
# Evaluation Configuration
# ============================================================================

@dataclass
class EvalConfig:
    """Configuration for transparency evaluation."""
    
    # Model settings
    model_name: str = "qwen3-0.6b"
    device: str = "auto"
    torch_dtype: str = "float16"  # "float16", "float32", "bfloat16"
    
    # Processing settings
    batch_size: int = 1  # Currently only single-item processing supported
    save_interval: int = 10  # Save checkpoint every N items
    
    # Resume behavior
    resume_from_checkpoint: bool = True
    
    # Test mode
    test_mode: bool = False  # If True, only process first N items
    test_n_items: int = 1
    
    # Paths (use defaults from module level)
    idioms_file: Path = field(default_factory=lambda: IDIOMS_FILE)
    prompt_file: Path = field(default_factory=lambda: PROMPT_FILE)
    output_file: Path = field(default_factory=lambda: OUTPUT_FILE)
    checkpoint_file: Path = field(default_factory=lambda: CHECKPOINT_FILE)
    
    def get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        import torch
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)


# ============================================================================
# Default Configuration Instance
# ============================================================================

def get_default_config() -> EvalConfig:
    """Get default evaluation configuration."""
    return EvalConfig()
