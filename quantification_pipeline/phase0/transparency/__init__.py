"""
Transparency evaluation pipeline for idiom semantic transparency.

This package evaluates whether English idioms are semantically transparent
(meaning easily inferable from literal words) using LLM logit extraction.

Usage:
    python -m quantification_pipeline.phase0.transparency.main --test
"""

from .base_model import BaseTransparencyModel, LogitResult
from .config import EvalConfig, get_default_config
from .data_handler import DataHandler, IdiomEntry, TransparencyResult
from .evaluator import TransparencyEvaluator

__all__ = [
    "BaseTransparencyModel",
    "LogitResult",
    "EvalConfig",
    "get_default_config",
    "DataHandler",
    "IdiomEntry",
    "TransparencyResult",
    "TransparencyEvaluator",
]
