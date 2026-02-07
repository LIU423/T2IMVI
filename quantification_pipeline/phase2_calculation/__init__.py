"""
Phase 2 Calculation Pipeline: Abstract Element Alignment (AEA)

This module evaluates the alignment between images and abstract atmosphere
descriptions from figurative interpretation JSONs.
"""

from .config import AEAConfig, MODEL_REGISTRY

__all__ = ["AEAConfig", "MODEL_REGISTRY"]
