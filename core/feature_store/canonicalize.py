"""
MIGI Core Feature Store - Macierz Tożsamości (M) Canonicalization
================================================================
Deterministic fingerprinting and canonicalization for Zero-Defect Inference
"""

import hashlib
import json
import time
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class CanonicalM:
    """Canonical representation of Macierz Tożsamości (M)"""
    digest: str
    dim: int
    schema_version: str
    source_map: Dict[str, Any]
    timestamp: float
    fingerprint: str

class MCanonicalizer:
    """
    Canonicalization engine for Macierz Tożsamości (M)
    Ensures deterministic, versioned, and auditable data representation
    """
    
    SCHEMA_VERSION = "1.0.0"
    
    def __init__(self):
        self.stats = {
            "canonical_calls": 0,
            "schema_drifts": 0,
            "validation_failures": 0
        }
    
    def canonicalize_M(self, inputs: Dict[str, Any]) -> CanonicalM:
        """
        Transform raw inputs into canonical Macierz Tożsamości (M)
        
        Args:
            inputs: Raw input data dictionary
            
        Returns:
            CanonicalM: Canonical representation with fingerprint
        """
        self.stats["canonical_calls"] += 1
        
        try:
            # 1. Deterministic serialization (sort_keys ensures consistency)
            blob = json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            
            # 2. Generate SHA256 digest
            digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()
            
            # 3. Calculate dimensionality
            dim = self._calculate_dimensions(inputs)
            
            # 4. Create source map for auditability
            source_map = self._create_source_map(inputs)
            
            # 5. Generate unique fingerprint
            fingerprint_data = {
                "digest": digest,
                "schema_version": self.SCHEMA_VERSION,
                "dim": dim
            }
            fingerprint_blob = json.dumps(fingerprint_data, sort_keys=True)
            fingerprint = hashlib.sha256(fingerprint_blob.encode()).hexdigest()[:16]
            
            # 6. Create canonical M
            canonical_m = CanonicalM(
                digest=digest,
                dim=dim,
                schema_version=self.SCHEMA_VERSION,
                source_map=source_map,
                timestamp=time.time(),
                fingerprint=fingerprint
            )
            
            # 7. Validate canonical M
            self._validate_canonical_m(canonical_m)
            
            return canonical_m
            
        except Exception as e:
            self.stats["validation_failures"] += 1
            raise ValueError(f"Failed to canonicalize M: {e}")
    
    def _calculate_dimensions(self, inputs: Dict[str, Any]) -> int:
        """Calculate dimensionality of input data"""
        total_dim = 0
        
        for key, value in inputs.items():
            if isinstance(value, str):
                total_dim += len(value)
            elif isinstance(value, (int, float)):
                total_dim += 1
            elif isinstance(value, (list, tuple)):
                total_dim += len(value)
            elif isinstance(value, dict):
                total_dim += len(value)
            else:
                total_dim += 1  # Default dimension for unknown types
        
        return total_dim
    
    def _create_source_map(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create source mapping for auditability"""
        source_map = {}
        
        for key, value in inputs.items():
            source_map[key] = {
                "type": type(value).__name__,
                "size": len(str(value)),
                "hash": hashlib.md5(str(value).encode()).hexdigest()[:8]
            }
        
        return source_map
    
    def _validate_canonical_m(self, canonical_m: CanonicalM) -> bool:
        """Validate canonical M structure and constraints"""
        if not canonical_m.digest:
            raise ValueError("Missing digest in canonical M")
        
        if canonical_m.dim <= 0:
            raise ValueError("Invalid dimensions in canonical M")
        
        if canonical_m.schema_version != self.SCHEMA_VERSION:
            self.stats["schema_drifts"] += 1
            raise ValueError(f"Schema version mismatch: expected {self.SCHEMA_VERSION}, got {canonical_m.schema_version}")
        
        if not canonical_m.fingerprint:
            raise ValueError("Missing fingerprint in canonical M")
        
        return True
    
    def verify_fingerprint(self, canonical_m: CanonicalM, expected_fingerprint: str) -> bool:
        """Verify fingerprint integrity"""
        return canonical_m.fingerprint == expected_fingerprint
    
    def get_stats(self) -> Dict[str, int]:
        """Get canonicalization statistics"""
        return self.stats.copy()

# Convenience function for backward compatibility
def canonicalize_M(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy function for canonicalizing M
    Returns dictionary format for compatibility
    """
    canonicalizer = MCanonicalizer()
    canonical_m = canonicalizer.canonicalize_M(inputs)
    
    return {
        "digest": canonical_m.digest,
        "dim": canonical_m.dim,
        "schema_version": canonical_m.schema_version,
        "fingerprint": canonical_m.fingerprint,
        "timestamp": canonical_m.timestamp
    }