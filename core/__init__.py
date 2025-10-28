"""
MIGI Core Package
================
Main package for Modular AI ecosystem with J.S.K. governance
"""

from .jsk import JSKController, JSKConfig, quick_infer
from .feature_store import canonicalize_M

__version__ = "2.0.0"

__all__ = [
    "JSKController",
    "JSKConfig",
    "quick_infer", 
    "canonicalize_M"
]