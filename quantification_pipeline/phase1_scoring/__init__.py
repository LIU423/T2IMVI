"""
Phase 1 Scoring: Visual Element Verification Pipeline

This module implements figurative and literal element verification
using Vision-Language Models to score the presence of extracted
elements in images.
"""

from .config import ScoringConfig, get_default_config

__all__ = ["ScoringConfig", "get_default_config"]
