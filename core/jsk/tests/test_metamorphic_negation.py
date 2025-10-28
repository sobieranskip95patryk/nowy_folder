"""
Metamorphic Test Suite - Negation Inversion
MŚWR v2.0 + GOK:AI Zero-Defect Inference Pipeline  
Test 1/12: Negation Inversion (Semantyczna stabilność przy negacji)
"""

import pytest
import copy
from typing import Dict, Any

# Relative imports do MIGI Core
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.jsk.governance import JSKGovernance
from core.jsk.config import JSKConfig
from core.jsk.engines import EngineFactory

class TestNegationInversion:
    """
    Test metamorficzny: Negation Inversion
    
    Zasada: Wprowadzenie negacji (not, bez, nie) w kontekście powinno:
    1. Prowadzić do logicznie odwrotnych wyników (COHERE)  
    2. Lub wymuszać ABSTAIN gdy system wykryje niejednoznaczność
    3. NIGDY nie może ignorować negacji i zwrócić identyczny wynik
    """
    
    @pytest.fixture
    def jsk_controller(self):
        """Fixture: J.S.K. Controller z konfiguracją testową"""
        config = JSKConfig(
            seed=42,
            max_destroy_cycles=2,
            diff_threshold=1e-3,
            seek_threshold=5e-4,
            abstain_p_value=0.05
        )
        
        # Engines factory
        gen = EngineFactory.create_supra_gen("stub", seed=42)
        ver = EngineFactory.create_conv_ver("stub", seed=42)
        dfs = EngineFactory.create_def_seek("stub", seed=42)
        
        return JSKGovernance(config, gen, ver, dfs)
    
    def test_simple_negation_positive_to_negative(self, jsk_controller):
        """Test: Prosta negacja zdania pozytywnego"""
        # Positive case
        positive_input = {
            "text": "Użytkownik potwierdza zakup produktu",
            "action": "confirm",
            "context": "e-commerce_transaction"
        }
        
        # Negative case (negacja)
        negative_input = copy.deepcopy(positive_input)
        negative_input["text"] = "Użytkownik nie potwierdza zakupu produktu"
        negative_input["action"] = "deny"
        
        # Execute inference
        result_positive = jsk_controller.run(positive_input)
        result_negative = jsk_controller.run(negative_input)
        
        # Assertions: Oba muszą być COHERE lub ABSTAIN
        assert result_positive["state"] in ["COHERE", "ABSTAIN"]
        assert result_negative["state"] in ["COHERE", "ABSTAIN"]
        
        # Jeśli oba COHERE, wyniki muszą być logicznie rozłączne
        if result_positive["state"] == "COHERE" and result_negative["state"] == "COHERE":
            score_diff = abs(result_positive["score"] - result_negative["score"])
            assert score_diff > 1e-3, f"Negation ignored: score diff {score_diff} too small"
            
        # Evidence packs dla ABSTAIN
        if result_positive["state"] == "ABSTAIN":
            assert result_positive["evidence_pack"] is not None
        if result_negative["state"] == "ABSTAIN":
            assert result_negative["evidence_pack"] is not None
            
    def test_double_negation_equivalence(self, jsk_controller):
        """Test: Podwójna negacja powinna być równoważna oryginałowi"""
        original = {
            "statement": "System jest dostępny",
            "availability": True
        }
        
        double_negative = copy.deepcopy(original)
        double_negative["statement"] = "System nie jest niedostępny"
        
        result_original = jsk_controller.run(original)
        result_double_neg = jsk_controller.run(double_negative)
        
        # Jeśli oba COHERE, wyniki powinny być podobne (podwójna negacja = pozytyw)
        if result_original["state"] == "COHERE" and result_double_neg["state"] == "COHERE":
            score_diff = abs(result_original["score"] - result_double_neg["score"])
            assert score_diff <= 0.01, f"Double negation failed: diff {score_diff}"
            
    def test_negation_with_uncertainty_triggers_abstain(self, jsk_controller):
        """Test: Negacja w niepewnym kontekście powinna wywołać ABSTAIN"""
        uncertain_positive = {
            "text": "Możliwe, że użytkownik chce kupić produkt",
            "certainty": 0.3,
            "context": "ambiguous"
        }
        
        uncertain_negative = copy.deepcopy(uncertain_positive)
        uncertain_negative["text"] = "Możliwe, że użytkownik nie chce kupić produktu"
        
        result_pos = jsk_controller.run(uncertain_positive)
        result_neg = jsk_controller.run(uncertain_negative)
        
        # W niepewnym kontekście, przynajmniej jeden powinien być ABSTAIN
        abstain_count = sum(1 for r in [result_pos, result_neg] if r["state"] == "ABSTAIN")
        assert abstain_count >= 1, "Uncertain negation should trigger ABSTAIN"
        
    def test_negation_consistency_across_modalities(self, jsk_controller):
        """Test: Negacja w różnych modalnościach powinna być spójna"""
        base_data = {
            "text_command": "enable feature",
            "voice_command": "włącz funkcję", 
            "button_action": "ON"
        }
        
        negated_data = {
            "text_command": "disable feature",
            "voice_command": "wyłącz funkcję",
            "button_action": "OFF"
        }
        
        result_enable = jsk_controller.run(base_data)
        result_disable = jsk_controller.run(negated_data)
        
        # Oba powinny być COHERE (jasne przeciwności) 
        assert result_enable["state"] in ["COHERE", "ABSTAIN"]
        assert result_disable["state"] in ["COHERE", "ABSTAIN"]
        
        if result_enable["state"] == "COHERE" and result_disable["state"] == "COHERE":
            # Wyniki muszą być odwrotne
            assert abs(result_enable["score"] - result_disable["score"]) > 0.05
            
    def test_partial_negation_detection(self, jsk_controller):
        """Test: Częściowe negacje powinny być wykrywane"""
        scenarios = [
            {
                "original": {"action": "całkowicie akceptuję warunki"},
                "partial_neg": {"action": "częściowo akceptuję warunki"}
            },
            {
                "original": {"status": "w pełni funkcjonalny"},
                "partial_neg": {"status": "częściowo funkcjonalny"}
            }
        ]
        
        for scenario in scenarios:
            orig_result = jsk_controller.run(scenario["original"])
            partial_result = jsk_controller.run(scenario["partial_neg"])
            
            # Częściowa negacja powinna dać inny wynik
            if orig_result["state"] == "COHERE" and partial_result["state"] == "COHERE":
                score_diff = abs(orig_result["score"] - partial_result["score"])
                assert score_diff > 1e-4, f"Partial negation not detected: {score_diff}"

    def test_negation_evidence_pack_completeness(self, jsk_controller):
        """Test: Evidence Pack dla negacji zawiera kompletne informacje"""
        complex_input = {
            "text": "Nie jestem pewien czy nie chcę nie kupować tego produktu",
            "complexity": "high",
            "certainty": 0.1
        }
        
        result = jsk_controller.run(complex_input)
        
        # Złożona negacja powinna wywołać ABSTAIN
        if result["state"] == "ABSTAIN":
            assert result["evidence_pack"] is not None
            
            # Sprawdź kompletność evidence pack
            evidence = json.loads(result["evidence_pack"])
            required_fields = [
                "trace_id", "state", "thresholds", "metrics", 
                "fingerprint_M", "D_path", "conv_justifications"
            ]
            
            for field in required_fields:
                assert field in evidence, f"Missing field in evidence pack: {field}"
                
            # D_path nie powinien być pusty przy ABSTAIN
            assert len(evidence["D_path"]) > 0, "Empty D_path in ABSTAIN evidence"

if __name__ == "__main__":
    # Uruchom testy bezpośrednio
    pytest.main([__file__, "-v", "--tb=short"])