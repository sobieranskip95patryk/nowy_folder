"""
MIGI Core - Jednolity Silnik Kalibracji (J.S.K.)
MŚWR v2.0 + GOK:AI Zero-Defect Inference Pipeline
Status: P=0.995 → P=1.0 (Protokół Wzrostu W)
"""

# J.S.K. Core imports
from .jsk.governance import JSKGovernance, JSKState, JSKTelemetry
from .jsk.config import load_jsk_config
from .jsk.engines import EngineFactory, SupraGenStub, ConvVerStub, DefSeekStub
from .feature_store.canonicalize import MCanonicalizer, CanonicalM, create_canonical_M

__version__ = "2.0.0"
__author__ = "MŚWR Core Team + GOK:AI"

# Status P-score progression tracking
CURRENT_P_SCORE = 0.995

__all__ = [
    # Governance Layer
    "JSKGovernance", 
    "JSKState", 
    "JSKTelemetry",
    
    # Configuration
    "load_jsk_config",
    
    # Engine Factory
    "EngineFactory", 
    "SupraGenStub", 
    "ConvVerStub", 
    "DefSeekStub",
    
    # Feature Store
    "MCanonicalizer", 
    "CanonicalM", 
    "create_canonical_M",
    
    # Meta
    "CURRENT_P_SCORE"
]