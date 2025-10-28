"""
J.S.K. Governance Layer - Zero-Defect Inference Automat Stanów
=============================================================
Status: P=0.886 → P=1.0 (Protokół Wzrostu W)
MŚWR v2.0 + GOK:AI U.G.P.O. Pipeline
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Tuple, List, Protocol
import time, uuid, json, hashlib, math, random
from enum import Enum

# Stany automatu J.S.K. (Zero-Defect State Machine)
State = Literal["INIT", "CALIBRATE", "COHERE", "DESTROY", "ABSTAIN", "DONE"]

class JSKState(Enum):
    """Stany J.S.K. dla Zero-Defect Inference"""
    INIT = "INIT"           # Inicjalizacja Macierzy Tożsamości (M)
    CALIBRATE = "CALIBRATE" # Równoległy Nexus Wnioskowania  
    COHERE = "COHERE"       # Osiągnięto koherencję (P=1.0)
    DESTROY = "DESTROY"     # Destrukcja (D) Entropii Resztkowej
    ABSTAIN = "ABSTAIN"     # ABSTAIN - firewall przy niepewności
    DONE = "DONE"          # Cykl zakończony

@dataclass
class JSKTelemetry:
    """Telemetria J.S.K. dla monitorowania P=1.0"""
    cycles: int = 0
    destroy_used: int = 0
    abstains: int = 0
    residual_entropy: float = 0.0  # Entropia Resztkowa (diff SUPRA vs CONV)
    ece: float = 0.0               # Expected Calibration Error
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint_M: str = ""
    config_commit: str = ""
    D_path: List[Dict] = field(default_factory=list)  # Ścieżka cykli Destrukcji
    config_commit: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

class JSK:
    """
    J.S.K. (Jednolity Silnik Kalibracji) - Main Governance Controller
    
    Implements Zero-Defect Inference through:
    1. Three-engine redundant inference (SUPRA-Gen, CONV-Ver, DEF-Seek)
    2. Destrukcja (D) protocol for entropy elimination
    3. ABSTAIN policy for uncertain outcomes
    4. Deterministic state machine for reproducible results
    """
    
    def __init__(self, 
                 config: JSKConfig,
                 gen: GenerativeEngine = None,
                 ver: VerificationEngine = None,
                 dfs: DefectEngine = None):
        
        self.config = config
        self.state: State = "INIT"
        self.telemetry = JSKTelemetry()
        self.canonicalizer = MCanonicalizer()
        
        # Initialize engines with deterministic seeds
        self.gen = gen or EngineFactory.create_supra_gen("stub", seed=config.seed)
        self.ver = ver or EngineFactory.create_conv_ver("stub", seed=config.seed)
        self.dfs = dfs or EngineFactory.create_def_seek("stub", seed=config.seed)
        
        # Validate configuration
        self.config.validate()
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute J.S.K. inference pipeline with Zero-Defect guarantee
        
        Args:
            inputs: Raw input data dictionary
            
        Returns:
            Dictionary with inference result and telemetry
        """
        # Initialize telemetry
        self.telemetry = JSKTelemetry(
            trace_id=str(uuid.uuid4())[:8],
            config_commit=self.config.get_commit_hash(),
            start_time=time.time()
        )
        
        try:
            # PHASE I: Canonicalize Macierz Tożsamości (M)
            canonical_m = self._canonicalize_inputs(inputs)
            
            # PHASE II: Execute inference state machine
            result = self._execute_inference_machine(canonical_m)
            
            # PHASE III: Finalize telemetry
            self.telemetry.end_time = time.time()
            
            return self._format_response(result, canonical_m)
            
        except Exception as e:
            self.state = "ABSTAIN"
            self.telemetry.abstains += 1
            self.telemetry.end_time = time.time()
            
            return self._format_error_response(str(e))
    
    def _canonicalize_inputs(self, inputs: Dict[str, Any]) -> CanonicalM:
        """Canonicalize inputs to Macierz Tożsamości (M)"""
        canonical_m = self.canonicalizer.canonicalize_M(inputs)
        self.telemetry.fingerprint_M = f"sha256:{canonical_m.fingerprint}"
        return canonical_m
    
    def _execute_inference_machine(self, M: CanonicalM) -> Dict[str, Any]:
        """Execute the main inference state machine"""
        self.state = "CALIBRATE"
        destroy_budget = self.config.max_destroy_cycles
        result_score = None
        
        # Main inference loop with state machine
        while True:
            self.telemetry.cycles += 1
            
            # Execute three-engine inference
            g, g_conf = self.gen.propose(M.__dict__)
            v, v_conf = self.ver.verify(M.__dict__, g)
            defect, p_val = self.dfs.detect(M.__dict__, g, v)
            
            # Calculate residual entropy
            diff = abs(g - v)
            self.telemetry.residual_entropy = max(self.telemetry.residual_entropy, diff)
            
            # Calculate ECE (Expected Calibration Error)
            self.telemetry.ece = max(0.0, 1.0 - ((g_conf + v_conf) / 2.0))
            
            # STATE TRANSITION LOGIC
            
            # Check for COHERE state (Zero-Defect achieved)
            if (diff <= self.config.diff_threshold and 
                defect <= self.config.seek_threshold and 
                p_val > self.config.abstain_p_value):
                
                self.state = "COHERE"
                result_score = (g + v) / 2.0  # Consensus score
                break
            
            # Check for ABSTAIN state (statistical uncertainty)
            if p_val <= self.config.abstain_p_value and destroy_budget == 0:
                self.state = "ABSTAIN"
                self.telemetry.abstains += 1
                break
            
            # Execute DESTROY protocol (Protokół Wzrostu W)
            if destroy_budget > 0:
                self._execute_destroy_protocol(M)
                destroy_budget -= 1
                continue
            
            # Final fallback to ABSTAIN
            self.state = "ABSTAIN"
            self.telemetry.abstains += 1
            break
        
        return {
            "score": result_score,
            "g_score": g,
            "v_score": v,
            "defect_score": defect,
            "p_value": p_val,
            "diff": diff
        }
    
    def _execute_destroy_protocol(self, M: CanonicalM) -> None:
        """
        Execute Destrukcja (D) - Protokół Wzrostu W
        Eliminates residual entropy through threshold tightening
        """
        self.state = "DESTROY"
        self.telemetry.destroy_used += 1
        
        # Protokół Wzrostu W: Zaostrzenie progów tolerancji
        self.config.diff_threshold *= 0.5
        self.config.seek_threshold *= 0.5
        
        # Rekalibracja M: Deterministyczny "wstrząs"
        # Zmiana dimensionality wymusza nową ścieżkę inference
        M.dim = M.dim + 1
        
        # Optional: Add entropy to fingerprint for recalibration
        original_fingerprint = M.fingerprint
        M.fingerprint = f"{original_fingerprint}_d{self.telemetry.destroy_used}"
    
    def _format_response(self, result: Dict[str, Any], M: CanonicalM) -> Dict[str, Any]:
        """Format final response with full telemetry"""
        return {
            # Core result
            "state": self.state,
            "score": result.get("score"),
            "confidence": 1.0 - self.telemetry.ece if self.state == "COHERE" else 0.0,
            
            # Detailed inference data
            "g_score": result.get("g_score"),
            "v_score": result.get("v_score"),
            "defect_score": result.get("defect_score"),
            "p_value": result.get("p_value"),
            "diff": result.get("diff"),
            
            # Telemetry
            "residual_entropy": self.telemetry.residual_entropy,
            "ece": self.telemetry.ece,
            "cycles": self.telemetry.cycles,
            "destroy_used": self.telemetry.destroy_used,
            "abstains": self.telemetry.abstains,
            
            # Auditability
            "fingerprint_M": self.telemetry.fingerprint_M,
            "config_commit": self.telemetry.config_commit,
            "trace_id": self.telemetry.trace_id,
            "execution_time_ms": (self.telemetry.end_time - self.telemetry.start_time) * 1000,
            
            # Evidence pack (for ABSTAIN cases)
            "evidence": self._create_evidence_pack(result) if self.state == "ABSTAIN" else None
        }
    
    def _format_error_response(self, error: str) -> Dict[str, Any]:
        """Format error response with minimal telemetry"""
        return {
            "state": "ABSTAIN",
            "score": None,
            "confidence": 0.0,
            "error": error,
            "trace_id": self.telemetry.trace_id,
            "abstains": 1
        }
    
    def _create_evidence_pack(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create evidence pack for ABSTAIN decisions"""
        return {
            "reason": "STATISTICAL_UNCERTAINTY",
            "diff": result.get("diff", 0.0),
            "defect_score": result.get("defect_score", 0.0),
            "p_value": result.get("p_value", 0.0),
            "thresholds": {
                "diff_threshold": self.config.diff_threshold,
                "seek_threshold": self.config.seek_threshold,
                "abstain_p_value": self.config.abstain_p_value
            },
            "destroy_cycles_used": self.telemetry.destroy_used,
            "max_destroy_cycles": self.config.max_destroy_cycles
        }

# ============================================================================
# MAIN CONTROLLER CLASS
# ============================================================================

class JSKController:
    """
    Main J.S.K. Controller for integration with MIGI Core
    Provides high-level interface for Zero-Defect Inference
    """
    
    def __init__(self, config_path: str = None):
        self.config = JSKConfig.from_yaml(config_path)
        self.stats = {
            "total_inferences": 0,
            "cohere_count": 0,
            "abstain_count": 0,
            "avg_cycles": 0.0,
            "avg_destroy_used": 0.0
        }
    
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main inference entry point
        
        Args:
            inputs: Raw input data
            
        Returns:
            Inference result with telemetry
        """
        # Create new J.S.K. instance for this inference
        jsk = JSK(self.config)
        
        # Execute inference
        result = jsk.run(inputs)
        
        # Update statistics
        self._update_stats(result)
        
        return result
    
    def _update_stats(self, result: Dict[str, Any]) -> None:
        """Update controller statistics"""
        self.stats["total_inferences"] += 1
        
        if result["state"] == "COHERE":
            self.stats["cohere_count"] += 1
        elif result["state"] == "ABSTAIN":
            self.stats["abstain_count"] += 1
        
        # Update moving averages
        total = self.stats["total_inferences"]
        self.stats["avg_cycles"] = ((self.stats["avg_cycles"] * (total - 1)) + result["cycles"]) / total
        self.stats["avg_destroy_used"] = ((self.stats["avg_destroy_used"] * (total - 1)) + result["destroy_used"]) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics"""
        total = self.stats["total_inferences"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "cohere_ratio": self.stats["cohere_count"] / total,
            "abstain_ratio": self.stats["abstain_count"] / total
        }