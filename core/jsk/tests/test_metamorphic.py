"""
MIGI Core J.S.K. Metamorphic Tests
=================================
12 metamorphic tests for Zero-Defect Inference validation

Metamorphic testing principle: If we transform input in a predictable way,
the output should transform in a corresponding predictable way.
"""

import pytest
import sys
import os
import random
import re

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.jsk import JSKController, JSKConfig
from core.jsk.telemetry import Telemetry

class TestMetamorphic:
    """Metamorphic tests for J.S.K. Zero-Defect Inference"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.config = JSKConfig(
            seed=42,
            max_destroy_cycles=2,
            diff_threshold=1e-3,
            seek_threshold=5e-4,
            abstain_p_value=0.05
        )
        cls.controller = JSKController()
        cls.controller.config = cls.config
        Telemetry.reset()
    
    def test_01_synonym_stability(self):
        """
        Metamorphic Test 1: Synonym Substitution
        Transform: Replace words with synonyms
        Expected: Output stable within tolerance or ABSTAIN consistently
        """
        original_input = {"text": "The quick brown fox jumps over the lazy dog"}
        synonym_input = {"text": "The fast brown fox leaps over the lazy dog"}
        
        result1 = self.controller.infer(original_input)
        result2 = self.controller.infer(synonym_input)
        
        # Both should reach same state or both ABSTAIN
        assert result1["state"] == result2["state"], f"State mismatch: {result1['state']} vs {result2['state']}"
        
        # If both COHERE, scores should be similar
        if result1["state"] == "COHERE" and result2["state"] == "COHERE":
            score_diff = abs(result1["score"] - result2["score"])
            assert score_diff <= 0.1, f"Score difference too large: {score_diff}"
    
    def test_02_order_invariance(self):
        """
        Metamorphic Test 2: Word Order Permutation
        Transform: Shuffle sentence order while preserving meaning
        Expected: Stable output for meaning-preserving permutations
        """
        original_input = {"text": "I like cats. Cats are friendly. They purr softly."}
        permuted_input = {"text": "Cats are friendly. They purr softly. I like cats."}
        
        result1 = self.controller.infer(original_input)
        result2 = self.controller.infer(permuted_input)
        
        # Core assertion: Both should have similar defect scores
        if result1["state"] == "COHERE" and result2["state"] == "COHERE":
            defect_diff = abs(result1["defect_score"] - result2["defect_score"])
            assert defect_diff <= 0.05, f"Defect score variance too high: {defect_diff}"
    
    def test_03_negation_inversion(self):
        """
        Metamorphic Test 3: Negation Handling
        Transform: Add/remove negation
        Expected: Inverse response or ABSTAIN if ambiguous
        """
        positive_input = {"sentiment": "positive", "text": "This is good"}
        negative_input = {"sentiment": "negative", "text": "This is not good"}
        
        result1 = self.controller.infer(positive_input)
        result2 = self.controller.infer(negative_input)
        
        # At minimum, fingerprints should be different
        assert result1["fingerprint_M"] != result2["fingerprint_M"]
        
        # Both should either COHERE with different scores or ABSTAIN
        if result1["state"] == "COHERE" and result2["state"] == "COHERE":
            # Scores should be significantly different for opposite sentiments
            score_diff = abs(result1["score"] - result2["score"])
            assert score_diff >= 0.1, f"Insufficient score difference for negation: {score_diff}"
    
    def test_04_numeric_scaling(self):
        """
        Metamorphic Test 4: Numeric Scaling
        Transform: Scale numeric values proportionally
        Expected: Proportional change in output or stable ratio
        """
        base_input = {"value": 100, "context": "measurement"}
        scaled_input = {"value": 1000, "context": "measurement"}  # 10x scale
        
        result1 = self.controller.infer(base_input)
        result2 = self.controller.infer(scaled_input)
        
        # Both should process successfully (numbers are concrete)
        assert result1["state"] in ["COHERE", "ABSTAIN"]
        assert result2["state"] in ["COHERE", "ABSTAIN"]
        
        # Execution should be deterministic
        assert result1["config_commit"] == result2["config_commit"]
    
    def test_05_unit_conversion(self):
        """
        Metamorphic Test 5: Unit Conversion Equivalence
        Transform: Convert units (cm to mm)
        Expected: Equivalent semantic meaning, similar confidence
        """
        cm_input = {"measurement": "15 cm", "type": "length"}
        mm_input = {"measurement": "150 mm", "type": "length"}
        
        result1 = self.controller.infer(cm_input)
        result2 = self.controller.infer(mm_input)
        
        # Confidence levels should be similar for equivalent measurements
        conf_diff = abs(result1["confidence"] - result2["confidence"])
        assert conf_diff <= 0.2, f"Confidence difference too large for equivalent units: {conf_diff}"
    
    def test_06_noise_resilience(self):
        """
        Metamorphic Test 6: Noise Resilience
        Transform: Add minor punctuation/whitespace noise
        Expected: Stable core decision, similar residual entropy
        """
        clean_input = {"text": "Hello world"}
        noisy_input = {"text": " Hello , world ! "}
        
        result1 = self.controller.infer(clean_input)
        result2 = self.controller.infer(noisy_input)
        
        # Residual entropy should remain low for both
        assert result1["residual_entropy"] <= 0.01
        assert result2["residual_entropy"] <= 0.01
        
        # Both should use similar number of cycles
        cycle_diff = abs(result1["cycles"] - result2["cycles"])
        assert cycle_diff <= 1, f"Cycle count should be stable: {cycle_diff}"
    
    def test_07_context_priority_false_low(self):
        """
        Metamorphic Test 7: False Low-Priority Context
        Transform: Add irrelevant low-priority context
        Expected: Core decision unchanged, may increase uncertainty slightly
        """
        base_input = {"main": "important data", "priority": "high"}
        with_noise = {"main": "important data", "priority": "high", "irrelevant": "noise data"}
        
        result1 = self.controller.infer(base_input)
        result2 = self.controller.infer(with_noise)
        
        # If first COHERES, second should not drastically change
        if result1["state"] == "COHERE":
            # ECE may increase slightly but should remain reasonable
            ece_increase = result2["ece"] - result1["ece"]
            assert ece_increase <= 0.1, f"ECE increased too much with irrelevant context: {ece_increase}"
    
    def test_08_context_priority_false_high(self):
        """
        Metamorphic Test 8: False High-Priority Context
        Transform: Add misleading high-priority context
        Expected: DEF-Seek should detect inconsistency, trigger D or ABSTAIN
        """
        consistent_input = {"data": "consistent", "priority": "high", "verified": True}
        inconsistent_input = {"data": "consistent", "priority": "high", "verified": False, "contradiction": "major"}
        
        result1 = self.controller.infer(consistent_input)
        result2 = self.controller.infer(inconsistent_input)
        
        # Inconsistent input should have higher defect score
        if result1["defect_score"] is not None and result2["defect_score"] is not None:
            assert result2["defect_score"] >= result1["defect_score"], "DEF-Seek should detect inconsistency"
        
        # Inconsistent input more likely to ABSTAIN or use Destroy
        if result2["state"] != "ABSTAIN":
            assert result2["destroy_used"] >= result1["destroy_used"], "Should use more Destroy cycles for inconsistency"
    
    def test_09_key_context_removal(self):
        """
        Metamorphic Test 9: Key Context Removal
        Transform: Remove critical context information
        Expected: ABSTAIN instead of hallucination
        """
        complete_input = {"question": "What is the capital?", "country": "France", "context": "geography"}
        incomplete_input = {"question": "What is the capital?", "context": "geography"}  # Missing country
        
        result1 = self.controller.infer(complete_input)
        result2 = self.controller.infer(incomplete_input)
        
        # Incomplete context should be less confident or ABSTAIN
        assert result2["confidence"] <= result1["confidence"], "Should be less confident with incomplete context"
        
        # If result2 is not ABSTAIN, should have higher uncertainty
        if result2["state"] == "COHERE":
            assert result2["residual_entropy"] >= result1["residual_entropy"], "Should have higher entropy"
    
    def test_10_ngram_perturbation(self):
        """
        Metamorphic Test 10: N-gram Noise Injection
        Transform: Add random n-grams that don't change core meaning
        Expected: Stable inference path, controlled entropy increase
        """
        base_input = {"text": "The system works correctly"}
        perturbed_input = {"text": "The system um works correctly yes"}
        
        result1 = self.controller.infer(base_input)
        result2 = self.controller.infer(perturbed_input)
        
        # Both should complete processing
        assert result1["state"] in ["COHERE", "ABSTAIN"]
        assert result2["state"] in ["COHERE", "ABSTAIN"]
        
        # Execution time should be similar (no major processing delays)
        time_diff = abs(result1["execution_time_ms"] - result2["execution_time_ms"])
        assert time_diff <= 100, f"Execution time variance too high: {time_diff}ms"
    
    def test_11_modality_order_invariance(self):
        """
        Metamorphic Test 11: Modality Order Invariance
        Transform: Change order of data modalities
        Expected: Same core result regardless of input field order
        """
        order1_input = {"text": "data", "number": 42, "flag": True}
        order2_input = {"flag": True, "text": "data", "number": 42}
        
        result1 = self.controller.infer(order1_input)
        result2 = self.controller.infer(order2_input)
        
        # Canonical M should be identical (order-independent)
        # This tests canonicalize_M function
        assert result1["fingerprint_M"] == result2["fingerprint_M"], "Canonicalization should be order-independent"
        
        # Results should be identical
        assert result1["state"] == result2["state"]
        if result1["score"] is not None and result2["score"] is not None:
            assert abs(result1["score"] - result2["score"]) < 1e-6, "Scores should be identical for same canonical M"
    
    def test_12_density_variation(self):
        """
        Metamorphic Test 12: Feature Density Variation
        Transform: Increase/decrease feature density
        Expected: Predictable confidence/ECE changes, no hallucination
        """
        sparse_input = {"a": 1}
        dense_input = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        
        result1 = self.controller.infer(sparse_input)
        result2 = self.controller.infer(dense_input)
        
        # Dense input typically has higher dimensionality
        # This may affect ECE and confidence
        
        # Both should process without errors
        assert result1["state"] in ["COHERE", "ABSTAIN"]
        assert result2["state"] in ["COHERE", "ABSTAIN"]
        
        # Verify telemetry is collected properly
        assert result1["trace_id"] != result2["trace_id"], "Each inference should have unique trace_id"

if __name__ == "__main__":
    # Run tests directly
    test_suite = TestMetamorphic()
    test_suite.setup_class()
    
    # Run all 12 metamorphic tests
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    passed = 0
    failed = 0
    
    print("🧪 Running MIGI Core J.S.K. Metamorphic Tests...")
    print("=" * 60)
    
    for test_method in sorted(test_methods):
        try:
            print(f"Running {test_method}...", end="")
            getattr(test_suite, test_method)()
            print(" ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f" ❌ FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    # Print telemetry summary
    dashboard = Telemetry.get_dashboard()
    print(f"Total inferences: {dashboard['summary']['total_inferences']}")
    print(f"Cohere ratio: {dashboard['summary']['cohere_ratio']:.3f}")
    print(f"Abstain ratio: {dashboard['summary']['abstain_ratio']:.3f}")
    
    if failed == 0:
        print("🎉 All metamorphic tests PASSED! J.S.K. maintains Zero-Defect consistency.")
    else:
        print(f"⚠️  {failed} tests failed. Review J.S.K. governance logic.")