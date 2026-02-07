"""
Verifier implementations for figurative and literal elements.

Each verifier takes an image and element description, returning
a probability score for element presence.
"""

from .figurative_verifier import FigurativeVerifier
from .literal_verifier import LiteralVerifier

__all__ = ["FigurativeVerifier", "LiteralVerifier"]
