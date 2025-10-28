"""
J.S.K. (Jednolity Silnik Kalibracji) - Configuration Management
MŚWR v2.0 + GOK:AI Zero-Defect Inference Pipeline
Status: P=0.886 → P=1.0 (Protokół Wzrostu W)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import yaml
import os
from pathlib import Path

@dataclass
class GovernanceConfig:
    """Parametry koherencji M-Operacyjne"""
    seed: int = 42
    max_destroy_cycles: int = 2
    schema_version: str = "1.0.0"

@dataclass  
class ThresholdConfig:
    """Progi kalibracyjne dla SLO P=1.0"""
    diff_threshold: float = 1e-3      # Koherencja SUPRA-Gen vs CONV-Ver
    seek_threshold: float = 5e-4      # Czułość DEF-Seek (detektor defektu)
    abstain_p_value: float = 0.05     # Próg konformalny dla ABSTAIN

@dataclass
class StrictModeConfig:
    """Tryb STRICT - aktywacja przy naruszeniu SLO"""
    enable: bool = True
    trigger_abstain_ratio: float = 0.10       # 10% ABSTAIN → STRICT
    trigger_entropy_p95: float = 3e-3         # Entropia > 3e-3 → STRICT
    trigger_duration_min: int = 15            # Czas naruszenia dla aktywacji
    multiplier_thresholds: float = 0.5        # Zaostrzenie progów x0.5
    disable_tool_calls: bool = True           # Wyłączenie narzędzi w STRICT

@dataclass
class SLOConfig:
    """Service Level Objectives - Kryteria P=1.0"""
    cohere_ratio_target: float = 0.999        # 99.9% COHERE w 24h
    max_cycles: int = 2                       # Średnie cykle ≤ 2
    residual_entropy_p95: float = 1e-3        # Entropia Resztkowa p95 ≤ 1e-3
    ece_p95: float = 0.05                     # Expected Calibration Error p95 ≤ 0.05
    abstain_ratio_max: float = 0.05           # ABSTAIN ≤ 5% w 24h

@dataclass
class TelemetryConfig:
    """Telemetry and monitoring configuration"""
    emit_interval_seconds: int = 30
    metrics_retention_hours: int = 168
    alert_thresholds: Dict[str, float] = None

@dataclass
class SecurityConfig:
    """Security policies for MIGI Core"""
    tool_calls_allowlist: List[str] = None
    fingerprint_M_required: bool = True
    config_commit_tracking: bool = True
    hard_stop_on_schema_drift: bool = True

@dataclass
class TestingConfig:
    """Testing and validation configuration"""
    metamorphic_tests_enabled: bool = True
    chaos_tests_enabled: bool = True
    metamorphic_count: int = 12
    chaos_count: int = 6
    test_traffic_ratio: float = 0.05

@dataclass
class JSKConfig:
    """
    Main J.S.K. Configuration for MIGI Core Zero-Defect Inference
    
    Attributes:
        seed: Deterministic seed for reproducibility
        max_destroy_cycles: Budget for Destrukcja (D) operations
        diff_threshold: Convergence threshold between SUPRA-Gen and CONV-Ver
        seek_threshold: DEF-Seek sensitivity threshold
        abstain_p_value: Conformal prediction threshold for ABSTAIN policy
        shadow_traffic_ratio: Percentage of traffic for shadow testing
    """
    seed: int = 42
    max_destroy_cycles: int = 2
    diff_threshold: float = 1e-3
    seek_threshold: float = 5e-4
    abstain_p_value: float = 0.05
    shadow_traffic_ratio: float = 0.10
    
    # Nested configurations
    slo: SLOConfig = None
    telemetry: TelemetryConfig = None
    security: SecurityConfig = None
    testing: TestingConfig = None
    
    def __post_init__(self):
        """Initialize nested configurations with defaults"""
        if self.slo is None:
            self.slo = SLOConfig()
        if self.telemetry is None:
            self.telemetry = TelemetryConfig(
                alert_thresholds={
                    "abstain_ratio_15min": 0.05,
                    "residual_entropy_p95": 1e-3,
                    "ece_p95": 0.05,
                    "avg_cycles": 2.0
                }
            )
        if self.security is None:
            self.security = SecurityConfig(
                tool_calls_allowlist=["file_ops", "compute", "verify"]
            )
        if self.testing is None:
            self.testing = TestingConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str = None) -> 'JSKConfig':
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'jsk.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Extract nested configurations
            slo_data = data.pop('slo', {})
            telemetry_data = data.pop('telemetry', {})
            security_data = data.pop('security', {})
            testing_data = data.pop('testing', {})
            
            # Create instance with main config
            config = cls(**data)
            
            # Set nested configurations
            config.slo = SLOConfig(**slo_data)
            config.telemetry = TelemetryConfig(**telemetry_data)
            config.security = SecurityConfig(**security_data)
            config.testing = TestingConfig(**testing_data)
            
            return config
            
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {config_path}. Using defaults.")
            return cls()
        except Exception as e:
            print(f"❌ Error loading config: {e}. Using defaults.")
            return cls()
    
    def validate(self) -> bool:
        """Validate configuration constraints"""
        if self.diff_threshold <= 0:
            raise ValueError("diff_threshold must be positive")
        if self.seek_threshold <= 0:
            raise ValueError("seek_threshold must be positive")
        if not (0 < self.abstain_p_value < 1):
            raise ValueError("abstain_p_value must be in (0, 1)")
        if self.max_destroy_cycles < 0:
            raise ValueError("max_destroy_cycles must be non-negative")
        if not (0 <= self.shadow_traffic_ratio <= 1):
            raise ValueError("shadow_traffic_ratio must be in [0, 1]")
        
        return True

    def get_commit_hash(self) -> str:
        """Get current configuration commit hash for reproducibility"""
        import hashlib
        import json
        
        # Create deterministic hash of configuration
        config_dict = {
            'seed': self.seed,
            'max_destroy_cycles': self.max_destroy_cycles,
            'diff_threshold': self.diff_threshold,
            'seek_threshold': self.seek_threshold,
            'abstain_p_value': self.abstain_p_value
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]