"""
MIGI Core J.S.K. Telemetry - Prometheus Metrics & SLO Monitoring
===============================================================
Real-time monitoring dla Zero-Defect Inference Pipeline
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class MetricWindow:
    """Sliding window for metric calculations"""
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add(self, value: float, timestamp: float = None):
        """Add value to window"""
        if timestamp is None:
            timestamp = time.time()
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_p95(self) -> float:
        """Get 95th percentile"""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(0.95 * len(sorted_values))
        return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def get_average(self) -> float:
        """Get average value"""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def get_count(self) -> int:
        """Get total count"""
        return len(self.values)

class JSKMetrics:
    """
    J.S.K. Metrics Collector and Aggregator
    Collects real-time metrics for SLO monitoring and alerting
    """
    
    def __init__(self):
        # Core metrics
        self.residual_entropy = MetricWindow()
        self.ece = MetricWindow()
        self.cycles = MetricWindow()
        self.destroy_used = MetricWindow()
        self.execution_time = MetricWindow()
        
        # Counters
        self.counters = defaultdict(int)
        
        # State tracking
        self.state_counts = defaultdict(int)
        self.last_emit_time = time.time()
        
        # SLO violations
        self.slo_violations = []
    
    def record_inference(self, result: Dict[str, Any]) -> None:
        """Record metrics from inference result"""
        timestamp = time.time()
        
        # Record core metrics
        self.residual_entropy.add(result.get("residual_entropy", 0.0), timestamp)
        self.ece.add(result.get("ece", 0.0), timestamp)
        self.cycles.add(result.get("cycles", 0), timestamp)
        self.destroy_used.add(result.get("destroy_used", 0), timestamp)
        self.execution_time.add(result.get("execution_time_ms", 0.0), timestamp)
        
        # Count by state
        state = result.get("state", "UNKNOWN")
        self.state_counts[state] += 1
        self.counters["total_inferences"] += 1
        
        # Count specific events
        if result.get("destroy_used", 0) > 0:
            self.counters["destroy_used_total"] += result["destroy_used"]
        
        if state == "ABSTAIN":
            self.counters["abstain_total"] += 1
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics"""
        timestamp = int(time.time() * 1000)
        
        metrics = []
        
        # Gauges
        metrics.append(f"jsk_residual_entropy {self.residual_entropy.get_average():.6f} {timestamp}")
        metrics.append(f"jsk_residual_entropy_p95 {self.residual_entropy.get_p95():.6f} {timestamp}")
        metrics.append(f"jsk_ece {self.ece.get_average():.6f} {timestamp}")
        metrics.append(f"jsk_ece_p95 {self.ece.get_p95():.6f} {timestamp}")
        metrics.append(f"jsk_avg_cycles {self.cycles.get_average():.2f} {timestamp}")
        metrics.append(f"jsk_avg_destroy_used {self.destroy_used.get_average():.2f} {timestamp}")
        metrics.append(f"jsk_execution_time_ms_avg {self.execution_time.get_average():.2f} {timestamp}")
        
        # Ratios
        total = self.counters["total_inferences"]
        if total > 0:
            cohere_ratio = self.state_counts["COHERE"] / total
            abstain_ratio = self.state_counts["ABSTAIN"] / total
            metrics.append(f"jsk_cohere_ratio {cohere_ratio:.4f} {timestamp}")
            metrics.append(f"jsk_abstain_ratio {abstain_ratio:.4f} {timestamp}")
        
        # Counters
        for name, value in self.counters.items():
            metrics.append(f"jsk_{name} {value} {timestamp}")
        
        return "\n".join(metrics)
    
    def check_slos(self, config) -> Dict[str, bool]:
        """Check SLO compliance"""
        violations = {}
        
        # Check residual entropy SLO
        if self.residual_entropy.get_p95() > config.slo.residual_entropy:
            violations["residual_entropy_p95"] = True
        
        # Check ECE SLO
        if self.ece.get_p95() > config.slo.ece:
            violations["ece_p95"] = True
        
        # Check cycles SLO
        if self.cycles.get_average() > config.slo.max_cycles:
            violations["avg_cycles"] = True
        
        # Check cohere ratio SLO
        total = self.counters["total_inferences"]
        if total > 10:  # Minimum sample size
            cohere_ratio = self.state_counts["COHERE"] / total
            if cohere_ratio < config.slo.cohere_ratio_min:
                violations["cohere_ratio"] = True
            
            abstain_ratio = self.state_counts["ABSTAIN"] / total
            if abstain_ratio > config.slo.abstain_ratio_max:
                violations["abstain_ratio"] = True
        
        # Record violations
        if violations:
            self.slo_violations.append({
                "timestamp": time.time(),
                "violations": violations.copy()
            })
        
        return violations
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        total = self.counters["total_inferences"]
        
        return {
            "summary": {
                "total_inferences": total,
                "cohere_count": self.state_counts["COHERE"],
                "abstain_count": self.state_counts["ABSTAIN"],
                "cohere_ratio": self.state_counts["COHERE"] / max(total, 1),
                "abstain_ratio": self.state_counts["ABSTAIN"] / max(total, 1)
            },
            "performance": {
                "avg_residual_entropy": self.residual_entropy.get_average(),
                "p95_residual_entropy": self.residual_entropy.get_p95(),
                "avg_ece": self.ece.get_average(),
                "p95_ece": self.ece.get_p95(),
                "avg_cycles": self.cycles.get_average(),
                "avg_destroy_used": self.destroy_used.get_average(),
                "avg_execution_time_ms": self.execution_time.get_average()
            },
            "slo_violations": self.slo_violations[-10:],  # Last 10 violations
            "recent_states": dict(self.state_counts)
        }
    
    def reset_counters(self) -> None:
        """Reset all counters (for testing)"""
        self.counters.clear()
        self.state_counts.clear()
        self.slo_violations.clear()
        
        # Clear metric windows
        self.residual_entropy = MetricWindow()
        self.ece = MetricWindow()
        self.cycles = MetricWindow()
        self.destroy_used = MetricWindow()
        self.execution_time = MetricWindow()

# Global metrics instance
metrics = JSKMetrics()

class Telemetry:
    """Static class for easy metric emission"""
    
    @staticmethod
    def emit_inference(result: Dict[str, Any]) -> None:
        """Emit inference metrics"""
        metrics.record_inference(result)
    
    @staticmethod
    def emit_all(result: Dict[str, Any], M: Dict[str, Any], config) -> None:
        """Emit all metrics with context"""
        metrics.record_inference(result)
        
        # Check SLOs
        violations = metrics.check_slos(config)
        if violations:
            print(f"⚠️  SLO Violations detected: {violations}")
    
    @staticmethod
    def get_prometheus() -> str:
        """Get Prometheus metrics"""
        return metrics.get_prometheus_metrics()
    
    @staticmethod
    def get_dashboard() -> Dict[str, Any]:
        """Get dashboard data"""
        return metrics.get_dashboard_data()
    
    @staticmethod
    def reset() -> None:
        """Reset all metrics"""
        metrics.reset_counters()

class AlertManager:
    """Simple alert manager for SLO violations"""
    
    def __init__(self):
        self.alert_history = deque(maxlen=100)
        self.suppression_window = 300  # 5 minutes
    
    def check_alerts(self, violations: Dict[str, bool]) -> None:
        """Check and emit alerts"""
        current_time = time.time()
        
        for violation_type, violated in violations.items():
            if violated:
                # Check if we recently alerted on this
                recent_alert = any(
                    alert["type"] == violation_type and 
                    current_time - alert["timestamp"] < self.suppression_window
                    for alert in self.alert_history
                )
                
                if not recent_alert:
                    self._emit_alert(violation_type, current_time)
    
    def _emit_alert(self, violation_type: str, timestamp: float) -> None:
        """Emit alert (placeholder - integrate with real alerting)"""
        alert = {
            "type": violation_type,
            "timestamp": timestamp,
            "message": f"SLO violation: {violation_type}"
        }
        
        self.alert_history.append(alert)
        print(f"🚨 ALERT: {alert['message']}")

# Global alert manager
alert_manager = AlertManager()