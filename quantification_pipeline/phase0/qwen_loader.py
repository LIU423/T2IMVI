"""
Shared Qwen loading utilities for Phase0.

This module centralizes model-loading decisions so all Phase0 entry points
use consistent and safe settings for Qwen models.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN3_30B_A3B_INSTRUCT_2507 = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def is_qwen3_30b_a3b_instruct_2507(model_id: str) -> bool:
    """Return True if model_id points to Qwen3-30B-A3B-Instruct-2507."""
    normalized = model_id.strip().lower()
    return normalized in {
        "qwen/qwen3-30b-a3b-instruct-2507",
        "qwen3-30b-a3b-instruct-2507",
    }


def resolve_model_device_map(device: str) -> Optional[str]:
    """
    Convert user device string to Hugging Face `device_map`.

    - `auto` -> `"auto"` for sharded/offloaded loading.
    - `cuda` / `cpu` -> `None` (regular load + explicit `.to(device)`).
    """
    if device == "auto":
        return "auto"
    return None


def resolve_torch_dtype(
    model_id: str,
    requested_dtype: Union[torch.dtype, str, None],
    device: str,
) -> Union[torch.dtype, str]:
    """
    Resolve torch dtype for loading.

    For Qwen3-30B-A3B-Instruct-2507, follow official recommendation:
    `torch_dtype="auto"`.
    """
    if is_qwen3_30b_a3b_instruct_2507(model_id):
        return "auto"

    if requested_dtype is None:
        return torch.float32 if device == "cpu" else torch.float16

    if isinstance(requested_dtype, str):
        # Keep "auto" and other explicit strings untouched.
        return requested_dtype

    if device == "cpu" and requested_dtype == torch.float16:
        # float16 on CPU is often unsupported/slow.
        return torch.float32

    return requested_dtype


def verify_loaded_qwen_architecture(model_id: str, model) -> None:
    """
    Sanity-check loaded model architecture for Qwen3-30B-A3B-Instruct-2507.
    """
    if not is_qwen3_30b_a3b_instruct_2507(model_id):
        return

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "qwen3_moe":
        raise RuntimeError(
            "Unexpected architecture for Qwen/Qwen3-30B-A3B-Instruct-2507: "
            f"model_type={model_type!r} (expected 'qwen3_moe'). "
            "Please use transformers>=4.51.0."
        )


def load_qwen_model_and_tokenizer(
    model_id: str,
    device: str,
    requested_dtype: Union[torch.dtype, str, None],
):
    """
    Load tokenizer and model with safe defaults and architecture checks.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
    )

    device_map = resolve_model_device_map(device)
    torch_dtype = resolve_torch_dtype(model_id, requested_dtype, device)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
    except KeyError as exc:
        # Helpful error for older transformers that do not support qwen3_moe.
        if is_qwen3_30b_a3b_instruct_2507(model_id) and "qwen3_moe" in str(exc):
            raise RuntimeError(
                "Failed to load Qwen3-MoE architecture. "
                "Please upgrade transformers to >=4.51.0."
            ) from exc
        raise

    if device_map is None:
        model = model.to(device)

    verify_loaded_qwen_architecture(model_id, model)
    return model, tokenizer
