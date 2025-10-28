"""
MIGI Core Feature Store Package
==============================
Macierz Tożsamości (M) canonicalization and fingerprinting
"""

from .canonicalize import MCanonicalizer, CanonicalM, canonicalize_M

__version__ = "1.0.0"

__all__ = [
    "MCanonicalizer",
    "CanonicalM", 
    "canonicalize_M"
]