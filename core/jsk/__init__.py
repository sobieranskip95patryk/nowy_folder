"""
MIGI Core J.S.K. Package
=======================
Jednolity Silnik Kalibracji for Zero-Defect Inference

Main exports:
- JSKController: Main interface for inference
- JSKConfig: Configuration management
- JSK: Core governance state machine
- Telemetry: Metrics and monitoring
"""

from .config import JSKConfig
from .governance import JSK, JSKController
from .telemetry import Telemetry, metrics
from .policies import PolicyEngine, PolicyDecision
from .engines import EngineFactory

__version__ = "1.0.0"
__author__ = "MIGI Core Team"

# Main exports
__all__ = [
    "JSKController",
    "JSKConfig", 
    "JSK",
    "Telemetry",
    "PolicyEngine",
    "PolicyDecision",
    "EngineFactory",
    "metrics"
]

# Default configuration
DEFAULT_CONFIG_PATH = None

def create_controller(config_path: str = None) -> JSKController:
    """
    Create JSK Controller with optional config path
    
    Args:
        config_path: Path to JSK YAML configuration file
        
    Returns:
        JSKController instance ready for inference
    """
    return JSKController(config_path or DEFAULT_CONFIG_PATH)

def quick_infer(inputs: dict, config_path: str = None) -> dict:
    """
    Quick inference without creating persistent controller
    
    Args:
        inputs: Input data dictionary
        config_path: Optional config path
        
    Returns:
        Inference result dictionary
    """
    controller = create_controller(config_path)
    return controller.infer(inputs)