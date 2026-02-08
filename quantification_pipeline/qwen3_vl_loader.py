"""
Shared Qwen3-VL architecture helpers.

Keeps special architecture handling in one modular place so phase1/phase2
model wrappers can stay mostly unchanged.
"""

import importlib
from typing import Type


QWEN3_VL_MOE_MODEL_IDS = {
    "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-30b-a3b-instruct",
}


def _normalize_model_id(model_id: str) -> str:
    return model_id.strip().lower()


def is_qwen3_vl_moe_model(model_id: str) -> bool:
    """Return True if model_id points to a Qwen3-VL MoE architecture model."""
    normalized = _normalize_model_id(model_id)
    return normalized in QWEN3_VL_MOE_MODEL_IDS


def get_qwen3_vl_generation_model_class(model_id: str) -> Type:
    """
    Resolve generation model class based on architecture.

    Default models use `Qwen3VLForConditionalGeneration`.
    Special MoE models (e.g., Qwen3-VL-30B-A3B-Instruct) use
    `Qwen3VLMoeForConditionalGeneration`.
    """
    transformers = importlib.import_module("transformers")

    if is_qwen3_vl_moe_model(model_id):
        moe_class = getattr(transformers, "Qwen3VLMoeForConditionalGeneration", None)
        if moe_class is not None:
            return moe_class
        raise RuntimeError(
            "Model requires Qwen3VLMoeForConditionalGeneration, but it is not "
            "available in your installed transformers version. "
            "Please upgrade transformers."
        )

    default_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if default_class is None:
        raise RuntimeError(
            "Qwen3VLForConditionalGeneration is not available in your installed "
            "transformers version."
        )
    return default_class

