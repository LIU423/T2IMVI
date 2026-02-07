"""
Model implementations for visual element verification.

Provides Vision-Language Model wrappers with probability extraction
for yes/no classification tasks.
"""

from .base_model import BaseVerifierModel, LogitResult

__all__ = ["BaseVerifierModel", "LogitResult"]
