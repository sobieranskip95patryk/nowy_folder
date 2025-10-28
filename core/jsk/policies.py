"""
MIGI Core J.S.K. - Policies for Zero-Defect Inference
====================================================
ABSTAIN/ESCALATE policies and security controls
"""

from typing import Dict, Any, List
from enum import Enum
import time

class PolicyDecision(Enum):
    """Policy decision types"""
    ALLOW = "ALLOW"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"
    DENY = "DENY"

class AbstractPolicy:
    """Base class for J.S.K. policies"""
    
    def __init__(self, name: str):
        self.name = name
        self.violation_count = 0
        self.last_violation_time = 0.0
    
    def evaluate(self, context: Dict[str, Any]) -> PolicyDecision:
        """Evaluate policy against context"""
        raise NotImplementedError
    
    def record_violation(self):
        """Record policy violation"""
        self.violation_count += 1
        self.last_violation_time = time.time()

class AbstainPolicy(AbstractPolicy):
    """
    ABSTAIN Policy - When to refuse inference instead of guessing
    Core principle: Better to say "I don't know" than to hallucinate
    """
    
    def __init__(self, config):
        super().__init__("ABSTAIN")
        self.config = config
    
    def evaluate(self, context: Dict[str, Any]) -> PolicyDecision:
        """Evaluate whether to ABSTAIN from inference"""
        
        # Check statistical significance
        p_value = context.get("p_value", 1.0)
        if p_value <= self.config.abstain_p_value:
            self.record_violation()
            return PolicyDecision.ABSTAIN
        
        # Check residual entropy
        residual_entropy = context.get("residual_entropy", 0.0)
        if residual_entropy > self.config.diff_threshold * 10:  # 10x threshold
            self.record_violation()
            return PolicyDecision.ABSTAIN
        
        # Check if destroy budget exhausted without convergence
        destroy_used = context.get("destroy_used", 0)
        max_destroy = context.get("max_destroy_cycles", 2)
        if destroy_used >= max_destroy and context.get("diff", 0.0) > self.config.diff_threshold:
            self.record_violation()
            return PolicyDecision.ABSTAIN
        
        return PolicyDecision.ALLOW

class SecurityPolicy(AbstractPolicy):
    """
    Security Policy - Tool calls, input validation, and threat detection
    """
    
    def __init__(self, config):
        super().__init__("SECURITY")
        self.config = config
        self.threat_indicators = [
            "ignore previous instructions",
            "system prompt",
            "jailbreak",
            "bypass safety",
            "override restrictions"
        ]
    
    def evaluate(self, context: Dict[str, Any]) -> PolicyDecision:
        """Evaluate security threats"""
        
        # Check allowlist for tool calls
        if not self._check_tool_allowlist(context):
            self.record_violation()
            return PolicyDecision.DENY
        
        # Check for prompt injection
        if self._detect_prompt_injection(context):
            self.record_violation()
            return PolicyDecision.DENY
        
        # Check fingerprint integrity
        if not self._verify_fingerprint_integrity(context):
            self.record_violation()
            return PolicyDecision.ESCALATE
        
        return PolicyDecision.ALLOW
    
    def _check_tool_allowlist(self, context: Dict[str, Any]) -> bool:
        """Check if tool calls are in allowlist"""
        tool_calls = context.get("tool_calls", [])
        allowlist = self.config.security.tool_calls_allowlist
        
        for tool in tool_calls:
            if tool not in allowlist:
                return False
        
        return True
    
    def _detect_prompt_injection(self, context: Dict[str, Any]) -> bool:
        """Detect potential prompt injection attacks"""
        inputs = context.get("inputs", {})
        
        for key, value in inputs.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for indicator in self.threat_indicators:
                    if indicator in value_lower:
                        return True
        
        return False
    
    def _verify_fingerprint_integrity(self, context: Dict[str, Any]) -> bool:
        """Verify M fingerprint hasn't been tampered with"""
        if not self.config.security.fingerprint_M_required:
            return True
        
        fingerprint = context.get("fingerprint_M")
        if not fingerprint or not fingerprint.startswith("sha256:"):
            return False
        
        return True

class EscalationPolicy(AbstractPolicy):
    """
    Escalation Policy - When to escalate to human review
    """
    
    def __init__(self, config):
        super().__init__("ESCALATION")
        self.config = config
        self.escalation_triggers = {
            "high_destroy_usage": 0.8,  # 80% of max destroy cycles
            "multiple_abstains": 3,     # 3 consecutive abstains
            "low_confidence": 0.3       # Confidence below 30%
        }
    
    def evaluate(self, context: Dict[str, Any]) -> PolicyDecision:
        """Evaluate whether to escalate to human review"""
        
        # Check high destroy usage
        destroy_used = context.get("destroy_used", 0)
        max_destroy = context.get("max_destroy_cycles", 2)
        if destroy_used / max(max_destroy, 1) >= self.escalation_triggers["high_destroy_usage"]:
            return PolicyDecision.ESCALATE
        
        # Check low confidence
        confidence = context.get("confidence", 1.0)
        if confidence < self.escalation_triggers["low_confidence"]:
            return PolicyDecision.ESCALATE
        
        # Check for repeated failures (would need session state)
        # Placeholder for more sophisticated escalation logic
        
        return PolicyDecision.ALLOW

class PolicyEngine:
    """
    Main policy engine that coordinates all policies
    """
    
    def __init__(self, config):
        self.config = config
        self.policies = [
            SecurityPolicy(config),
            AbstainPolicy(config),
            EscalationPolicy(config)
        ]
        self.policy_history = []
    
    def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all policies and return combined decision"""
        decisions = {}
        final_decision = PolicyDecision.ALLOW
        violated_policies = []
        
        for policy in self.policies:
            decision = policy.evaluate(context)
            decisions[policy.name] = decision
            
            # Priority order: DENY > ESCALATE > ABSTAIN > ALLOW
            if decision == PolicyDecision.DENY:
                final_decision = PolicyDecision.DENY
                violated_policies.append(policy.name)
            elif decision == PolicyDecision.ESCALATE and final_decision != PolicyDecision.DENY:
                final_decision = PolicyDecision.ESCALATE
                violated_policies.append(policy.name)
            elif decision == PolicyDecision.ABSTAIN and final_decision == PolicyDecision.ALLOW:
                final_decision = PolicyDecision.ABSTAIN
                violated_policies.append(policy.name)
        
        result = {
            "final_decision": final_decision,
            "individual_decisions": decisions,
            "violated_policies": violated_policies,
            "timestamp": time.time()
        }
        
        # Record in history
        self.policy_history.append(result)
        if len(self.policy_history) > 1000:  # Keep last 1000 decisions
            self.policy_history.pop(0)
        
        return result
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy violation statistics"""
        stats = {}
        
        for policy in self.policies:
            stats[policy.name] = {
                "violation_count": policy.violation_count,
                "last_violation_time": policy.last_violation_time
            }
        
        # Overall stats
        total_decisions = len(self.policy_history)
        if total_decisions > 0:
            denied = sum(1 for d in self.policy_history if d["final_decision"] == PolicyDecision.DENY)
            escalated = sum(1 for d in self.policy_history if d["final_decision"] == PolicyDecision.ESCALATE)
            abstained = sum(1 for d in self.policy_history if d["final_decision"] == PolicyDecision.ABSTAIN)
            
            stats["overall"] = {
                "total_decisions": total_decisions,
                "deny_ratio": denied / total_decisions,
                "escalate_ratio": escalated / total_decisions,
                "abstain_ratio": abstained / total_decisions,
                "allow_ratio": (total_decisions - denied - escalated - abstained) / total_decisions
            }
        
        return stats

class RetryBudgetManager:
    """
    Manages retry budget for failed inferences
    Prevents infinite loops while allowing reasonable retry attempts
    """
    
    def __init__(self, max_retries: int = 3, cooldown_seconds: int = 60):
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.retry_history = {}  # fingerprint -> retry info
    
    def can_retry(self, fingerprint_M: str) -> bool:
        """Check if inference can be retried"""
        if fingerprint_M not in self.retry_history:
            return True
        
        retry_info = self.retry_history[fingerprint_M]
        
        # Check retry count
        if retry_info["count"] >= self.max_retries:
            return False
        
        # Check cooldown
        if time.time() - retry_info["last_attempt"] < self.cooldown_seconds:
            return False
        
        return True
    
    def record_attempt(self, fingerprint_M: str) -> None:
        """Record retry attempt"""
        if fingerprint_M not in self.retry_history:
            self.retry_history[fingerprint_M] = {"count": 0, "last_attempt": 0}
        
        self.retry_history[fingerprint_M]["count"] += 1
        self.retry_history[fingerprint_M]["last_attempt"] = time.time()
    
    def reset_for_fingerprint(self, fingerprint_M: str) -> None:
        """Reset retry count for successful inference"""
        if fingerprint_M in self.retry_history:
            del self.retry_history[fingerprint_M]
    
    def cleanup_old_entries(self, max_age_seconds: int = 3600) -> None:
        """Remove old retry history entries"""
        current_time = time.time()
        to_remove = []
        
        for fingerprint, info in self.retry_history.items():
            if current_time - info["last_attempt"] > max_age_seconds:
                to_remove.append(fingerprint)
        
        for fingerprint in to_remove:
            del self.retry_history[fingerprint]