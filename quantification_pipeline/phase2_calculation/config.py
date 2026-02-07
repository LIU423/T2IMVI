"""
Configuration for Phase 2 AEA (Abstract Element Alignment) Calculation Pipeline.

This module provides:
- AEAConfig: Main configuration dataclass
- MODEL_REGISTRY: Registry of available VLM models
- Templated path utilities for dataset/model swapping
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import sys

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
    OUTPUT_DIR,
    PROMPT_DIR,
    PHASE2_AEA_PROMPT,
    PHASE2_IU_RELATIONSHIPS_PROMPT,
    PHASE2_IU_WITHOUT_RELATIONSHIPS_PROMPT,
)

# =============================================================================
# Path Configuration (Derived from centralized project_config)
# =============================================================================

# Dataset configuration - change these to swap datasets
DATASET_NAME = "IRFL"  # Template: easily replaceable

# Model configuration - change these to swap models
DEFAULT_MODEL_KEY = "qwen3-vl-2b"  # Key in MODEL_REGISTRY
DEFAULT_MODEL_OUTPUT_PREFIX = "qwen3_vl_2b"  # Used in output folder names

# Path templates - derived from centralized config
DATA_ROOT = DATA_DIR
INPUT_ROOT = DATA_ROOT / "input" / DATASET_NAME
OUTPUT_ROOT = DATA_ROOT / "output" / DATASET_NAME
PROMPT_ROOT = PROMPT_DIR

# Input paths
MATCHED_IMAGES_ROOT = INPUT_IRFL_MATCHED_IMAGES_DIR  # /<id>/<image_id>.jpeg

# Prompt files - use centralized paths
AEA_PROMPT_FILE = PHASE2_AEA_PROMPT
IU_RELATIONSHIPS_PROMPT_FILE = PHASE2_IU_RELATIONSHIPS_PROMPT
IU_WITHOUT_RELATIONSHIPS_PROMPT_FILE = PHASE2_IU_WITHOUT_RELATIONSHIPS_PROMPT

# Score threshold for entity/action validation
IU_SCORE_THRESHOLD = 0.1


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3-vl-2b": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "class": "Qwen3VLModel",
        "module": "quantification_pipeline.phase2_calculation.models.qwen_vl_model",
        "output_prefix": "qwen3_vl_2b",
        "description": "Qwen3-VL-2B-Instruct Vision-Language Model",
    },
    # Add more models here for easy swapping
    # "llava-1.5": {
    #     "model_id": "llava-hf/llava-1.5-7b-hf",
    #     "class": "LLaVAModel",
    #     "module": "quantification_pipeline.phase2_calculation.models.llava_model",
    #     "output_prefix": "llava_1_5",
    #     "description": "LLaVA 1.5 7B Vision-Language Model",
    # },
}


def get_model_class(model_key: str):
    """
    Dynamically import and return the model class.
    
    Args:
        model_key: Key in MODEL_REGISTRY
        
    Returns:
        The model class
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_key]
    module_path = model_info["module"]
    class_name = model_info["class"]
    
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_output_dir(model_key: str, dataset: str = DATASET_NAME) -> Path:
    """
    Get output directory for a model.
    
    Args:
        model_key: Key in MODEL_REGISTRY
        dataset: Dataset name (default: IRFL)
        
    Returns:
        Path to output directory: data/output/<dataset>/<model_prefix>_T2IMVI/
    """
    model_prefix = MODEL_REGISTRY[model_key]["output_prefix"]
    return DATA_ROOT / "output" / dataset / f"{model_prefix}_T2IMVI"


def get_phase1_output_dir(model_key: str, dataset: str = DATASET_NAME) -> Path:
    """
    Get Phase 1 scoring output directory (input for Phase 2).
    
    This is where figurative.json files with scores are located.
    
    Args:
        model_key: Key in MODEL_REGISTRY
        dataset: Dataset name (default: IRFL)
        
    Returns:
        Path to Phase 1 output: data/output/<dataset>/<model_prefix>_T2IMVI/
    """
    return get_output_dir(model_key, dataset)


def get_image_path(idiom_id: int, image_id: int, dataset: str = DATASET_NAME) -> Path:
    """
    Get path to an image file.
    
    Args:
        idiom_id: Idiom ID
        image_id: Image ID
        dataset: Dataset name
        
    Returns:
        Path to image: data/input/<dataset>/matched_images/<id>/<image_id>.jpeg
    """
    return DATA_ROOT / "input" / dataset / "matched_images" / str(idiom_id) / f"{image_id}.jpeg"


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class AEAConfig:
    """
    Configuration for AEA (Abstract Element Alignment) calculation.
    
    Attributes:
        model_key: Key in MODEL_REGISTRY for the VLM to use
        dataset: Dataset name (e.g., "IRFL")
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Number of images to process in parallel
        checkpoint_interval: Save checkpoint every N images
        resume: Whether to resume from checkpoint
        idiom_ids: List of idiom IDs to process (None = all)
        test_mode: If True, limit processing for testing
        test_n_images: Max images per idiom in test mode
        score_threshold: Min score for entities/actions. If all below, AEA=0.0
    """
    model_key: str = DEFAULT_MODEL_KEY
    dataset: str = DATASET_NAME
    device: str = "cuda"
    batch_size: int = 1  # VLMs typically process one at a time
    checkpoint_interval: int = 10
    resume: bool = True
    idiom_ids: Optional[List[int]] = None
    test_mode: bool = False
    test_n_images: int = 1
    score_threshold: float = IU_SCORE_THRESHOLD  # Default 0.1
    
    # Computed paths (set in __post_init__)
    phase1_output_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    images_root: Path = field(init=False)
    prompt_file: Path = field(init=False)
    checkpoint_file: Path = field(init=False)
    
    def __post_init__(self):
        """Compute derived paths after initialization."""
        self.phase1_output_dir = get_phase1_output_dir(self.model_key, self.dataset)
        self.output_dir = get_output_dir(self.model_key, self.dataset)
        self.images_root = DATA_ROOT / "input" / self.dataset / "matched_images"
        self.prompt_file = AEA_PROMPT_FILE
        self.checkpoint_file = self.output_dir / "aea_checkpoint.json"
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model info from registry."""
        return MODEL_REGISTRY[self.model_key]
    
    @property
    def model_id(self) -> str:
        """Get HuggingFace model ID."""
        return self.model_info["model_id"]
    
    def get_idiom_dir(self, idiom_id: int) -> Path:
        """Get directory for an idiom's output."""
        return self.phase1_output_dir / f"idiom_{idiom_id}"
    
    def get_image_dir(self, idiom_id: int, image_id: int) -> Path:
        """Get directory for an image's output."""
        return self.get_idiom_dir(idiom_id) / f"image_{image_id}"
    
    def get_figurative_json_path(self, idiom_id: int, image_id: int) -> Path:
        """Get path to figurative.json (Phase 1 output, Phase 2 input)."""
        return self.get_image_dir(idiom_id, image_id) / "figurative.json"
    
    def get_figurative_score_json_path(self, idiom_id: int, image_id: int) -> Path:
        """Get path to figurative_score.json (Phase 2 output)."""
        return self.get_image_dir(idiom_id, image_id) / "figurative_score.json"
    
    def get_image_path(self, idiom_id: int, image_id: int) -> Path:
        """Get path to source image."""
        return self.images_root / str(idiom_id) / f"{image_id}.jpeg"


# =============================================================================
# IU (Image Understanding) Configuration Dataclass
# =============================================================================

@dataclass
class IUConfig:
    """
    Configuration for IU (Image Understanding) calculation.
    
    Uses VQAScore methodology: P("yes") from yes/no questions about
    whether the image embodies the core abstract concept via relationships
    or entity-action pairs.
    
    Attributes:
        model_key: Key in MODEL_REGISTRY for the VLM to use
        dataset: Dataset name (e.g., "IRFL")
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Number of images to process in parallel
        checkpoint_interval: Save checkpoint every N images
        resume: Whether to resume from checkpoint
        idiom_ids: List of idiom IDs to process (None = all)
        test_mode: If True, limit processing for testing
        test_n_images: Max images per idiom in test mode
        score_threshold: Minimum score for entity/action to be valid (default: 0.1)
    """
    model_key: str = DEFAULT_MODEL_KEY
    dataset: str = DATASET_NAME
    device: str = "cuda"
    batch_size: int = 1  # VLMs typically process one at a time
    checkpoint_interval: int = 10
    resume: bool = True
    idiom_ids: Optional[List[int]] = None
    test_mode: bool = False
    test_n_images: int = 1
    score_threshold: float = IU_SCORE_THRESHOLD
    
    # Computed paths (set in __post_init__)
    phase1_output_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    images_root: Path = field(init=False)
    relationships_prompt_file: Path = field(init=False)
    without_relationships_prompt_file: Path = field(init=False)
    checkpoint_file: Path = field(init=False)
    
    def __post_init__(self):
        """Compute derived paths after initialization."""
        self.phase1_output_dir = get_phase1_output_dir(self.model_key, self.dataset)
        self.output_dir = get_output_dir(self.model_key, self.dataset)
        self.images_root = DATA_ROOT / "input" / self.dataset / "matched_images"
        self.relationships_prompt_file = IU_RELATIONSHIPS_PROMPT_FILE
        self.without_relationships_prompt_file = IU_WITHOUT_RELATIONSHIPS_PROMPT_FILE
        self.checkpoint_file = self.output_dir / "iu_checkpoint.json"
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model info from registry."""
        return MODEL_REGISTRY[self.model_key]
    
    @property
    def model_id(self) -> str:
        """Get HuggingFace model ID."""
        return self.model_info["model_id"]
    
    def get_idiom_dir(self, idiom_id: int) -> Path:
        """Get directory for an idiom's output."""
        return self.phase1_output_dir / f"idiom_{idiom_id}"
    
    def get_image_dir(self, idiom_id: int, image_id: int) -> Path:
        """Get directory for an image's output."""
        return self.get_idiom_dir(idiom_id) / f"image_{image_id}"
    
    def get_figurative_json_path(self, idiom_id: int, image_id: int) -> Path:
        """Get path to figurative.json (Phase 1 output, Phase 2 input)."""
        return self.get_image_dir(idiom_id, image_id) / "figurative.json"
    
    def get_figurative_score_json_path(self, idiom_id: int, image_id: int) -> Path:
        """Get path to figurative_score.json (Phase 2 output)."""
        return self.get_image_dir(idiom_id, image_id) / "figurative_score.json"
    
    def get_image_path(self, idiom_id: int, image_id: int) -> Path:
        """Get path to source image."""
        return self.images_root / str(idiom_id) / f"{image_id}.jpeg"


# =============================================================================
# Logging Configuration
# =============================================================================

import logging

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for the pipeline.
    
    Args:
        level: Logging level
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)
