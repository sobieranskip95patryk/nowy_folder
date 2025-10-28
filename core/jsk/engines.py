"""
MIGI Core Engine Interfaces - U.G.P.O. Pipeline Components
==========================================================
Kontrakty dla Supra-Gen, Conv-Ver, i Def-Seek silników
"""

from typing import Protocol, Tuple, Dict, Any
import hashlib
import math
import random
import numpy as np

# ============================================================================
# ENGINE CONTRACTS (Protocols)
# ============================================================================

class GenerativeEngine(Protocol):
    """Contract for SUPRA-Gen: Ultra-Predictive Generative Engine"""
    
    def propose(self, M: Dict[str, Any]) -> Tuple[float, float]:
        """
        Generate proposal based on Macierz Tożsamości (M)
        
        Args:
            M: Canonical representation of input data
            
        Returns:
            Tuple[score, confidence]: Proposed score [0..1] and confidence [0..1]
        """
        ...

class VerificationEngine(Protocol):
    """Contract for CONV-Ver: Contextual Verification Engine"""
    
    def verify(self, M: Dict[str, Any], proposal: float) -> Tuple[float, float]:
        """
        Verify proposal against historical Macierz Tożsamości (M)
        
        Args:
            M: Canonical representation of input data
            proposal: Score proposed by GenerativeEngine
            
        Returns:
            Tuple[score, confidence]: Verified score [0..1] and confidence [0..1]
        """
        ...

class DefectEngine(Protocol):
    """Contract for DEF-Seek: Defect Detection Engine"""
    
    def detect(self, M: Dict[str, Any], g: float, v: float) -> Tuple[float, float]:
        """
        Detect potential defects in SUPRA-Gen vs CONV-Ver results
        
        Args:
            M: Canonical representation of input data
            g: Score from GenerativeEngine
            v: Score from VerificationEngine
            
        Returns:
            Tuple[defect_score, p_value]: Defect severity and statistical p-value
        """
        ...

# ============================================================================
# REFERENCE IMPLEMENTATIONS (Stubs for Testing)
# ============================================================================

class SupraGenStub:
    """Reference implementation of SUPRA-Gen engine"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.generation_count = 0
    
    def propose(self, M: Dict[str, Any]) -> Tuple[float, float]:
        """Generate proposal with controlled entropy"""
        self.generation_count += 1
        
        # Deterministic base score from M fingerprint
        fingerprint = M.get("fingerprint", "default")
        base_hash = int(hashlib.md5(fingerprint.encode()).hexdigest()[:8], 16)
        base_score = (base_hash % 1000) / 1000.0
        
        # Add controlled noise based on dimensionality
        dim = M.get("dim", 1)
        entropy_factor = math.log(max(dim, 1)) / 10.0
        noise = random.uniform(-entropy_factor, entropy_factor)
        
        score = max(0.0, min(1.0, base_score + noise))
        confidence = 0.92 - (entropy_factor * 0.1)  # Higher dim = lower confidence
        
        return (score, confidence)

class ConvVerStub:
    """Reference implementation of CONV-Ver engine"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed + 1)  # Different seed than SupraGen
        self.verification_count = 0
    
    def verify(self, M: Dict[str, Any], proposal: float) -> Tuple[float, float]:
        """Verify proposal with contextual analysis"""
        self.verification_count += 1
        
        # Contextual adjustment based on M characteristics
        dim = M.get("dim", 1)
        schema_version = M.get("schema_version", "1.0.0")
        
        # Verification tends to be more conservative
        verification_bias = -0.05
        
        # Adjust based on proposal confidence
        if proposal > 0.8:
            # High proposals get more scrutiny
            adjustment = random.uniform(-0.1, 0.05)
        elif proposal < 0.2:
            # Low proposals get benefit of doubt
            adjustment = random.uniform(-0.02, 0.08)
        else:
            # Medium proposals get standard treatment
            adjustment = random.uniform(-0.05, 0.05)
        
        verified_score = max(0.0, min(1.0, proposal + verification_bias + adjustment))
        confidence = 0.94 - (abs(proposal - 0.5) * 0.1)  # More confident near middle values
        
        return (verified_score, confidence)

class DefSeekStub:
    """Reference implementation of DEF-Seek engine"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed + 2)  # Different seed from other engines
        self.detection_count = 0
    
    def detect(self, M: Dict[str, Any], g: float, v: float) -> Tuple[float, float]:
        """Detect defects using statistical analysis"""
        self.detection_count += 1
        
        # Primary defect metric: difference between generators
        diff = abs(g - v)
        
        # Enhanced defect analysis
        dim = M.get("dim", 1)
        
        # Defect severity increases with:
        # 1. Large difference between g and v
        # 2. High dimensionality (more complex data)
        # 3. Extreme values (near 0 or 1)
        
        defect_score = diff
        
        # Dimension penalty
        if dim > 100:
            defect_score *= 1.5
        elif dim > 500:
            defect_score *= 2.0
        
        # Extreme value penalty
        if g < 0.1 or g > 0.9 or v < 0.1 or v > 0.9:
            defect_score *= 1.3
        
        # Statistical p-value using exponential decay
        # Lower p-value = more significant defect
        p_value = math.exp(-1000 * diff)
        
        # Add some noise to prevent perfect determinism
        noise = random.uniform(-0.001, 0.001)
        p_value = max(0.0, min(1.0, p_value + noise))
        
        return (defect_score, p_value)

# ============================================================================
# ENGINE FACTORY
# ============================================================================

class EngineFactory:
    """Factory for creating engine instances"""
    
    @staticmethod
    def create_supra_gen(engine_type: str = "stub", **kwargs) -> GenerativeEngine:
        """Create SUPRA-Gen engine instance"""
        if engine_type == "stub":
            return SupraGenStub(**kwargs)
        else:
            raise ValueError(f"Unknown SUPRA-Gen engine type: {engine_type}")
    
    @staticmethod
    def create_conv_ver(engine_type: str = "stub", **kwargs) -> VerificationEngine:
        """Create CONV-Ver engine instance"""
        if engine_type == "stub":
            return ConvVerStub(**kwargs)
        else:
            raise ValueError(f"Unknown CONV-Ver engine type: {engine_type}")
    
    @staticmethod
    def create_def_seek(engine_type: str = "stub", **kwargs) -> DefectEngine:
        """Create DEF-Seek engine instance"""
        if engine_type == "stub":
            return DefSeekStub(**kwargs)
        else:
            raise ValueError(f"Unknown DEF-Seek engine type: {engine_type}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_engine_output(score: float, confidence: float) -> bool:
    """Validate engine output ranges"""
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"Score must be in [0, 1], got {score}")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Confidence must be in [0, 1], got {confidence}")
    return True

def calculate_engine_agreement(g: float, v: float, threshold: float = 0.1) -> bool:
    """Check if engines agree within threshold"""
    return abs(g - v) <= threshold