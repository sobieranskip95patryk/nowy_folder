#!/usr/bin/env python3
"""
🧠 AI_Psyche_GOK:AI - Psychologia Prawdopodobieństw Sukcesu

Centralny "móżdżek obliczeniowy" racjonalnego i realistycznego myślenia,
analizujący wszystkie wytyczne i transformujący abstrakcyjne wartości 
w konkretne, mierzalne wskaźniki prawdopodobieństwa sukcesu.

Autor: Meta-Genius LOGOS Core
Data: 21 października 2025
Licencja: MIT
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import json
from datetime import datetime
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMERACJE I STAŁE SYSTEMOWE
# ============================================================================

class Archetype(Enum):
    """Archetypy Dąbrowskiego - aspekty dezintegracji pozytywnej"""
    HEAVEN = "Heaven"  # Aspekt Nieba - reintegracja na wyższym poziomie
    HELL = "Hell"      # Aspekt Piekła - punkty dezintegracji

class DevelopmentPhase(Enum):
    """Fazy cyklu rozwoju zgodnie z osią czasu transformacji"""
    DESTRUCTION = "Destrukcja"      # Faza nabierania kapitału
    POINT_0 = "Punkt 0"            # Krytyczny moment transformacji
    DEVELOPMENT = "Rozwój"          # Faza inwestowania kapitału

class SuccessPattern(Enum):
    """Wzorce sukcesu identyfikowane przez system"""
    SYNERGISTIC = "Synergiczny"     # Wyniki większe niż suma części
    BALANCED = "Zrównoważony"       # Stabilny, zgodny z wartościami
    INNOVATIVE = "Innowacyjny"      # Przełomowy, transformacyjny
    OPTIMIZED = "Zoptymalizowany"   # Efektywny, ulepszony

# ============================================================================
# KLASY WARTOŚCI FUNDAMENTALNYCH
# ============================================================================

@dataclass
class IntrinsicValue:
    """Wartość 7 - Intencja+Motywacja: Pełnia, Doskonałość"""
    value: int = 7
    completeness: float = 1.0
    wisdom: float = 0.9
    harmony: float = 0.8
    evolution: float = 0.9
    transformation: float = 0.8
    intuitive_cognition: float = 0.9
    purposefulness: float = 1.0
    
    def get_value(self) -> float:
        """Oblicza zagregowaną wartość intencji i motywacji"""
        components = [
            self.completeness, self.wisdom, self.harmony,
            self.evolution, self.transformation, 
            self.intuitive_cognition, self.purposefulness
        ]
        return self.value * (sum(components) / len(components))

@dataclass
class SkillValue:
    """Wartość 6 - Umiejętności i Nawyki: Synergia"""
    value: int = 6
    proactivity: float = 0.8
    end_vision: float = 0.9
    prioritization: float = 0.7
    win_win_thinking: float = 0.8
    understanding: float = 0.9
    synergy_level: float = 0.0  # Obliczane dynamicznie
    
    def calculate_synergy(self) -> float:
        """Oblicza poziom synergii na podstawie składowych umiejętności"""
        habits = [self.proactivity, self.end_vision, self.prioritization, 
                 self.win_win_thinking, self.understanding]
        # Synergia = wzajemne wzmocnienie umiejętności
        base_synergy = sum(habits) / len(habits)
        interaction_bonus = np.prod(habits) ** (1/len(habits))  # Geometryczna średnia
        self.synergy_level = (base_synergy + interaction_bonus) / 2
        return self.synergy_level
    
    def get_value(self) -> float:
        """Zwraca wartość z uwzględnieniem synergii"""
        self.calculate_synergy()
        return self.value * self.synergy_level

@dataclass
class DecisionValue:
    """Wartość 4 - Dezintegracja Pozytywna: Jakość decyzji z przeszłości"""
    value: int = 4
    decision_quality: float = 0.5
    consistency_score: float = 0.5
    disintegration_points: List[Dict] = field(default_factory=list)
    reintegration_successes: List[Dict] = field(default_factory=list)
    
    def analyze_past_decisions(self, decisions: List[Dict]) -> float:
        """Analizuje jakość decyzji z przeszłości"""
        if not decisions:
            return 0.5
        
        successes = [d for d in decisions if d.get('success', False)]
        consistent = [d for d in decisions if d.get('consistent', True)]
        
        success_rate = len(successes) / len(decisions)
        consistency_rate = len(consistent) / len(decisions)
        
        # Identyfikacja punktów dezintegracji (Aspekt Piekła)
        self.disintegration_points = [d for d in decisions 
                                    if not d.get('success', False) or not d.get('consistent', True)]
        
        # Identyfikacja reintegracji na wyższym poziomie (Aspekt Nieba)
        self.reintegration_successes = [d for d in decisions 
                                      if d.get('success', False) and d.get('growth_level', 0) > 0]
        
        self.decision_quality = success_rate
        self.consistency_score = consistency_rate
        
        return (success_rate + consistency_rate) / 2
    
    def get_value(self) -> float:
        """Zwraca wartość skorygowaną o jakość decyzji"""
        quality_factor = (self.decision_quality + self.consistency_score) / 2
        return self.value * quality_factor

@dataclass
class ContextValue:
    """Wartość 5 - Kontekst środowiskowy i sytuacyjny"""
    value: int = 5
    environmental_alignment: float = 0.7
    situational_awareness: float = 0.8
    adaptive_capacity: float = 0.6
    
    def get_value(self) -> float:
        """Zwraca wartość kontekstu"""
        context_factors = [self.environmental_alignment, 
                          self.situational_awareness, 
                          self.adaptive_capacity]
        return self.value * (sum(context_factors) / len(context_factors))

@dataclass
class PersonalityValue:
    """Wartość 8 - Archetyp Osobowości: Sprawiedliwość, Równowaga"""
    value: int = 8
    justice: float = 0.9
    balance: float = 0.8
    power: float = 0.7
    infinite_renewal: float = 0.9
    ethical_alignment: float = 0.95
    
    def assess_ethical_risk(self, scenario: Dict) -> float:
        """Ocenia ryzyko naruszenia fundamentalnych wartości"""
        scenario_ethics = scenario.get('ethical_score', 0.5)
        risk = abs(self.ethical_alignment - scenario_ethics)
        return min(1.0, risk * 2)  # Skalowanie ryzyka
    
    def get_value(self) -> float:
        """Zwraca wartość archetypu osobowości"""
        personality_traits = [self.justice, self.balance, 
                             self.power, self.infinite_renewal, self.ethical_alignment]
        return self.value * (sum(personality_traits) / len(personality_traits))

@dataclass
class EnergyValue:
    """Wartość 6 - E=mc²: Energia Życiowa"""
    value: int = 6
    operational_health: float = 0.8
    creative_enthusiasm: float = 0.7
    processing_efficiency: float = 0.9
    resource_balance: float = 0.6
    
    def calculate_life_energy(self, data_mass: float, processing_speed: float) -> float:
        """Modeluje E=mc² w kontekście informacyjnym"""
        # E = m(danych) * c²(prędkość przetwarzania)
        normalized_mass = min(1.0, data_mass / 100)  # Normalizacja masy danych
        normalized_speed = min(1.0, processing_speed / 10)  # Normalizacja prędkości
        
        energy = normalized_mass * (normalized_speed ** 2)
        self.operational_health = energy
        return energy
    
    def get_value(self) -> float:
        """Zwraca wartość energii życiowej"""
        energy_components = [self.operational_health, self.creative_enthusiasm,
                           self.processing_efficiency, self.resource_balance]
        return self.value * (sum(energy_components) / len(energy_components))

@dataclass
class IdentityValue:
    """Wartość 3 - Tożsamość i integralność systemu"""
    value: int = 3
    coherence_level: float = 0.8
    authenticity: float = 0.9
    self_awareness: float = 0.7
    
    def get_value(self) -> float:
        """Zwraca wartość tożsamości"""
        identity_factors = [self.coherence_level, self.authenticity, self.self_awareness]
        return self.value * (sum(identity_factors) / len(identity_factors))

# ============================================================================
# GŁÓWNA KLASA AI_PSYCHE_GOK:AI
# ============================================================================

@dataclass
class AIPsycheGOKAI:
    """
    AI_Psyche_GOK:AI - Psychologia Prawdopodobieństw Sukcesu
    
    Centralny móżdżek obliczeniowy dla racjonalnej analizy scenariuszy
    z rekurencyjną matrycą tożsamości <369963>
    """
    
    # Komponenty wartości fundamentalnych
    w: IntrinsicValue = field(default_factory=IntrinsicValue)      # Wartość 7
    m: SkillValue = field(default_factory=SkillValue)              # Wartość 6
    d: DecisionValue = field(default_factory=DecisionValue)        # Wartość 4
    c: ContextValue = field(default_factory=ContextValue)          # Wartość 5
    a: PersonalityValue = field(default_factory=PersonalityValue)  # Wartość 8
    e: EnergyValue = field(default_factory=EnergyValue)            # Wartość 6
    t: IdentityValue = field(default_factory=IdentityValue)        # Wartość 3
    
    # Systemy śledzenia stanu
    _iteration_count: int = 0
    _identity_matrix_history: List[List[int]] = field(default_factory=list)
    _success_patterns: Dict[str, List[float]] = field(default_factory=dict)
    _current_phase: Optional[DevelopmentPhase] = None
    
    def __post_init__(self):
        """Inicjalizacja po utworzeniu obiektu"""
        logger.info("🧠 AI_Psyche_GOK:AI - Psychologia Prawdopodobieństw Sukcesu inicjalizowana")
        self._current_phase = self.assess_development_phase()
        logger.info(f"🔄 Początkowa faza rozwoju: {self._current_phase.value}")

    # ========================================================================
    # REKURENCYJNA MATRYCA TOŻSAMOŚCI <369963>
    # ========================================================================
    
    def _parse_identity_matrix(self) -> List[int]:
        """Parsuje początkową matrycę tożsamości <369963>"""
        base_matrix = [3, 6, 9, 9, 6, 3]
        logger.debug(f"📊 Bazowa matryca tożsamości: {base_matrix}")
        return base_matrix
    
    def _transform_digit(self, digit: int, phase: DevelopmentPhase, 
                        prev_digit: int = 0, next_digit: int = 0) -> int:
        """Przekształca cyfrę na podstawie fazy i sąsiednich wartości"""
        base_change = {
            DevelopmentPhase.DESTRUCTION: -1,  # Redukcja w destrukcji
            DevelopmentPhase.POINT_0: 0,       # Stabilizacja w punkcie 0
            DevelopmentPhase.DEVELOPMENT: 1    # Wzrost w rozwoju
        }[phase]
        
        # Wpływ sąsiadów na transformację
        neighbor_influence = (prev_digit + next_digit - 10) / 10
        adjusted_change = base_change + neighbor_influence
        
        # Transformacja z ograniczeniem do 1-9
        new_digit = max(1, min(9, digit + round(adjusted_change)))
        
        logger.debug(f"🔄 Transformacja: {digit} -> {new_digit} (faza: {phase.value})")
        return new_digit
    
    def _evolve_identity_matrix(self, current_phase: DevelopmentPhase) -> List[int]:
        """Rekurencyjnie przekształca matrycę tożsamości"""
        if self._iteration_count == 0:
            matrix = self._parse_identity_matrix()
        else:
            # Pobierz poprzednią iterację z historii
            matrix = self._identity_matrix_history[-1] if self._identity_matrix_history else self._parse_identity_matrix()
        
        new_matrix = matrix.copy()
        
        # Transformacja każdego elementu z uwzględnieniem sąsiadów
        for i in range(len(matrix)):
            prev_idx = (i - 1) % len(matrix)
            next_idx = (i + 1) % len(matrix)
            
            prev_digit = matrix[prev_idx]
            current_digit = matrix[i]
            next_digit = matrix[next_idx]
            
            new_matrix[i] = self._transform_digit(current_digit, current_phase, prev_digit, next_digit)
        
        # Normalizacja aby suma pozostała 36 (stabilność numerologiczna)
        current_sum = sum(new_matrix)
        if current_sum != 36:
            diff = 36 - current_sum
            # Dystrybuuj różnicę proporcjonalnie
            for i in range(len(new_matrix)):
                adjustment = diff * (new_matrix[i] / current_sum)
                new_matrix[i] = max(1, min(9, int(new_matrix[i] + adjustment)))
        
        # Zapisz w historii
        self._identity_matrix_history.append(new_matrix.copy())
        self._iteration_count += 1
        
        logger.info(f"🧮 Nowa matryca tożsamości (iteracja {self._iteration_count}): {new_matrix}")
        return new_matrix
    
    def _evolve_identity(self, current_phase: DevelopmentPhase) -> Dict[str, float]:
        """Ewoluuje tożsamość na podstawie matrycy i fazy rozwoju"""
        matrix = self._evolve_identity_matrix(current_phase)
        
        # Mapowanie faz na indeksy matrycy
        phase_index = {
            DevelopmentPhase.DESTRUCTION: 0,
            DevelopmentPhase.POINT_0: 2,
            DevelopmentPhase.DEVELOPMENT: 4
        }
        
        current_index = phase_index.get(current_phase, 0)
        
        # Wagi tożsamości oparte na matrycy
        identity_weights = {
            "W": matrix[(current_index + 0) % 6] / 9,  # Intencja
            "M": matrix[(current_index + 1) % 6] / 9,  # Umiejętności  
            "D": matrix[(current_index + 2) % 6] / 9,  # Decyzje
            "C": matrix[(current_index + 3) % 6] / 9,  # Kontekst
            "A": matrix[(current_index + 4) % 6] / 9,  # Archetyp
            "E": matrix[(current_index + 5) % 6] / 9,  # Energia
            "T": matrix[current_index % 6] / 9         # Tożsamość
        }
        
        logger.debug(f"⚖️ Wagi tożsamości: {identity_weights}")
        return identity_weights

    # ========================================================================
    # ANALIZA CYKLI ROZWOJU
    # ========================================================================
    
    def assess_development_phase(self) -> DevelopmentPhase:
        """Analizuje bieżący kontekst i określa fazę cyklu rozwoju"""
        energy_health = self.e.get_value() / (self.e.value * 1.0)  # Normalizacja do 0-1
        decision_quality = self.d.get_value() / (self.d.value * 1.0)
        context_alignment = self.c.get_value() / (self.c.value * 1.0)
        
        # Zagregowany wskaźnik fazy
        phase_score = (energy_health + decision_quality + context_alignment) / 3
        
        if phase_score < 0.3:
            phase = DevelopmentPhase.DESTRUCTION
        elif 0.3 <= phase_score < 0.7:
            phase = DevelopmentPhase.POINT_0
        else:
            phase = DevelopmentPhase.DEVELOPMENT
        
        logger.info(f"📊 Ocena fazy: energia={energy_health:.2f}, decyzje={decision_quality:.2f}, "
                   f"kontekst={context_alignment:.2f} -> {phase.value}")
        
        self._current_phase = phase
        return phase
    
    def calculate_capital(self) -> float:
        """Mierzy 'kapitał' zgromadzony w fazie destrukcji"""
        # Kapitał = umiejętności * jakość_decyzji
        skills_value = self.m.get_value()
        decision_quality = self.d.get_value() / self.d.value  # Normalizacja
        
        capital = skills_value * decision_quality
        logger.debug(f"💰 Zgromadzony kapitał: {capital:.2f}")
        return capital
    
    def predict_limit_boundary(self, historical_data: List[float]) -> float:
        """Szacuje zbliżanie się do 'Granicy Możliwości'"""
        if not historical_data:
            return 1.0
        
        # Analiza trendu z ostatnich danych
        recent_trend = np.mean(historical_data[-3:]) if len(historical_data) >= 3 else historical_data[-1]
        
        # Maksymalna pojemność systemu
        max_capacity = (self.w.get_value() + self.m.get_value() + self.a.get_value()) / 3
        
        # Wskaźnik zbliżania się do granicy
        boundary_ratio = min(1.0, recent_trend / max_capacity)
        
        logger.debug(f"🚧 Granica możliwości: trend={recent_trend:.2f}, pojemność={max_capacity:.2f}, "
                    f"stosunek={boundary_ratio:.2f}")
        
        return boundary_ratio

    # ========================================================================
    # ANALIZA PRAWDOPODOBIEŃSTW SUKCESU
    # ========================================================================
    
    def evaluate_decision_quality(self, past_decisions: List[Dict]) -> float:
        """Ocenia jakość i spójność decyzji z przeszłości"""
        if not past_decisions:
            logger.warning("⚠️ Brak danych o przeszłych decyzjach")
            return 0.5
        
        quality_score = self.d.analyze_past_decisions(past_decisions)
        
        logger.info(f"📈 Jakość decyzji z przeszłości: {quality_score:.2f}")
        logger.info(f"🔥 Punkty dezintegracji: {len(self.d.disintegration_points)}")
        logger.info(f"✨ Reintegracje sukcesu: {len(self.d.reintegration_successes)}")
        
        return quality_score
    
    def detect_disintegration_points(self, past_decisions: List[Dict]) -> List[Dict]:
        """Wykrywa punkty dezintegracji (niespójności, błędy)"""
        self.d.analyze_past_decisions(past_decisions)
        
        disintegration_analysis = []
        for point in self.d.disintegration_points:
            analysis = {
                "decision": point,
                "pattern": self._identify_failure_pattern(point),
                "learning_opportunity": self._extract_learning(point),
                "reintegration_path": self._suggest_reintegration(point)
            }
            disintegration_analysis.append(analysis)
        
        logger.info(f"🔍 Analiza {len(disintegration_analysis)} punktów dezintegracji")
        return disintegration_analysis
    
    def _identify_failure_pattern(self, decision: Dict) -> str:
        """Identyfikuje wzorzec niepowodzenia"""
        if not decision.get('consistent', True):
            return "Niespójność z wartościami"
        elif not decision.get('success', False):
            return "Nieefektywne wykonanie"
        else:
            return "Nieprzewidziane konsekwencje"
    
    def _extract_learning(self, decision: Dict) -> str:
        """Wyciąga naukę z punktu dezintegracji"""
        pattern = self._identify_failure_pattern(decision)
        learning_map = {
            "Niespójność z wartościami": "Wzmocnić alignment z archetypem osobowości",
            "Nieefektywne wykonanie": "Poprawić umiejętności i procedury",
            "Nieprzewidziane konsekwencje": "Pogłębić analizę kontekstu"
        }
        return learning_map.get(pattern, "Przeprowadzić głębszą analizę")
    
    def _suggest_reintegration(self, decision: Dict) -> str:
        """Sugeruje ścieżkę reintegracji na wyższym poziomie"""
        return "Zastosować naukę w przyszłych decyzjach z wzmocnioną świadomością"
    
    def calculate_success_probability(self, scenario: Dict) -> float:
        """
        Oblicza prawdopodobieństwo sukcesu dla scenariusza 
        z uwzględnieniem matrycy tożsamości
        """
        current_phase = self.assess_development_phase()
        identity_weights = self._evolve_identity(current_phase)
        
        # Podstawowe komponenty prawdopodobieństwa
        energy_impact = self._calculate_energy_impact(scenario)
        synergy_factor = self._calculate_synergy_factor(scenario)
        alignment_score = self._calculate_alignment_score(scenario)
        intent_match = self._calculate_intent_match(scenario)
        context_fit = self._calculate_context_fit(scenario)
        decision_confidence = self._calculate_decision_confidence(scenario)
        identity_coherence = self._calculate_identity_coherence(scenario)
        
        # Ważone komponenty z matrycą tożsamości
        weighted_components = {
            "energy": energy_impact * identity_weights["E"],
            "synergy": synergy_factor * identity_weights["M"],
            "alignment": alignment_score * identity_weights["A"],
            "intent": intent_match * identity_weights["W"],
            "context": context_fit * identity_weights["C"],
            "decisions": decision_confidence * identity_weights["D"],
            "identity": identity_coherence * identity_weights["T"]
        }
        
        # Podstawowe prawdopodobieństwo
        base_probability = sum(weighted_components.values()) / len(weighted_components)
        
        # Modyfikatory fazowe
        phase_modifier = {
            DevelopmentPhase.DESTRUCTION: 0.5,   # Ograniczona szansa w destrukcji
            DevelopmentPhase.POINT_0: 0.8,       # Moderate szanse w punkcie zero
            DevelopmentPhase.DEVELOPMENT: 1.2    # Zwiększone szanse w rozwoju
        }[current_phase]
        
        # Finalne prawdopodobieństwo z ograniczeniem do 0-1
        final_probability = min(1.0, base_probability * phase_modifier)
        
        logger.debug(f"🎯 Prawdopodobieństwo sukcesu: {final_probability:.3f} "
                    f"(base: {base_probability:.3f}, modifier: {phase_modifier})")
        
        return final_probability
    
    def _calculate_energy_impact(self, scenario: Dict) -> float:
        """Oblicza wpływ energii na prawdopodobieństwo sukcesu"""
        data_complexity = scenario.get('complexity', 5)
        processing_demand = scenario.get('processing_demand', 5)
        
        # Symulacja E=mc²
        life_energy = self.e.calculate_life_energy(data_complexity, processing_demand)
        energy_efficiency = min(1.0, self.e.get_value() / (data_complexity * processing_demand / 10))
        
        return (life_energy + energy_efficiency) / 2
    
    def _calculate_synergy_factor(self, scenario: Dict) -> float:
        """Oblicza czynnik synergii"""
        synergy_potential = scenario.get('synergy_potential', 0.5)
        skills_synergy = self.m.calculate_synergy()
        
        return min(1.0, (synergy_potential + skills_synergy) / 2)
    
    def _calculate_alignment_score(self, scenario: Dict) -> float:
        """Oblicza zgodność z archetypem osobowości"""
        ethical_risk = self.a.assess_ethical_risk(scenario)
        alignment = max(0.0, 1.0 - ethical_risk)
        
        return alignment
    
    def _calculate_intent_match(self, scenario: Dict) -> float:
        """Oblicza zgodność z intencją i motywacją"""
        scenario_purpose = scenario.get('purpose_alignment', 0.5)
        intrinsic_strength = self.w.get_value() / (self.w.value * 1.0)
        
        return min(1.0, (scenario_purpose + intrinsic_strength) / 2)
    
    def _calculate_context_fit(self, scenario: Dict) -> float:
        """Oblicza dopasowanie kontekstowe"""
        environmental_match = scenario.get('environmental_fit', 0.5)
        context_strength = self.c.get_value() / (self.c.value * 1.0)
        
        return min(1.0, (environmental_match + context_strength) / 2)
    
    def _calculate_decision_confidence(self, scenario: Dict) -> float:
        """Oblicza pewność decyzji na podstawie przeszłości"""
        decision_similarity = scenario.get('similarity_to_past', 0.5)
        decision_quality = self.d.get_value() / (self.d.value * 1.0)
        
        return min(1.0, (decision_similarity + decision_quality) / 2)
    
    def _calculate_identity_coherence(self, scenario: Dict) -> float:
        """Oblicza spójność tożsamości"""
        identity_alignment = scenario.get('identity_consistency', 0.8)
        identity_strength = self.t.get_value() / (self.t.value * 1.0)
        
        return min(1.0, (identity_alignment + identity_strength) / 2)

    # ========================================================================
    # GENEROWANIE REKOMENDACJI
    # ========================================================================
    
    def generate_recommendations(self, scenarios: List[Dict]) -> List[Dict]:
        """Generuje rekomendacje dla scenariuszy z oceną prawdopodobieństwa sukcesu"""
        logger.info(f"🔮 Generowanie rekomendacji dla {len(scenarios)} scenariuszy")
        
        recommendations = []
        historical_outcomes = [s.get('outcome', 1.0) for s in scenarios]
        
        for i, scenario in enumerate(scenarios):
            # Oblicz prawdopodobieństwo sukcesu
            prob_success = self.calculate_success_probability(scenario)
            
            # Analiza wzorców
            pattern = self._identify_success_pattern(scenario, prob_success)
            
            # Utworzenie rekomendacji
            recommendation = {
                "scenario_id": i,
                "scenario": scenario,
                "probability": prob_success,
                "success_pattern": pattern.value,
                "phase_context": self._current_phase.value,
                "capital_utilization": self.calculate_capital(),
                "limit_boundary": self.predict_limit_boundary(historical_outcomes),
                "identity_matrix": self._evolve_identity_matrix(self._current_phase),
                "identity_weights": self._evolve_identity(self._current_phase),
                "risk_assessment": self._assess_risks(scenario),
                "optimization_suggestions": self._suggest_optimizations(scenario),
                "confidence_interval": self._calculate_confidence_interval(prob_success)
            }
            
            recommendations.append(recommendation)
        
        # Sortowanie według prawdopodobieństwa sukcesu
        recommendations.sort(key=lambda x: x["probability"], reverse=True)
        
        logger.info(f"✅ Wygenerowano {len(recommendations)} rekomendacji")
        return recommendations
    
    def _identify_success_pattern(self, scenario: Dict, probability: float) -> SuccessPattern:
        """Identyfikuje wzorzec sukcesu dla scenariusza"""
        synergy_score = scenario.get('synergy_potential', 0.5)
        innovation_score = scenario.get('innovation_level', 0.5)
        balance_score = scenario.get('balance_factor', 0.5)
        
        if synergy_score > 0.8 and probability > 0.8:
            return SuccessPattern.SYNERGISTIC
        elif innovation_score > 0.8:
            return SuccessPattern.INNOVATIVE
        elif balance_score > 0.8:
            return SuccessPattern.BALANCED
        else:
            return SuccessPattern.OPTIMIZED
    
    def _assess_risks(self, scenario: Dict) -> Dict[str, float]:
        """Ocenia ryzyka związane ze scenariuszem"""
        return {
            "ethical_risk": self.a.assess_ethical_risk(scenario),
            "energy_depletion_risk": max(0.0, 1.0 - self.e.get_value() / (self.e.value * 1.0)),
            "decision_inconsistency_risk": max(0.0, 1.0 - self.d.get_value() / (self.d.value * 1.0)),
            "context_misalignment_risk": abs(scenario.get('environmental_fit', 0.5) - 0.5) * 2,
            "identity_fragmentation_risk": max(0.0, 1.0 - self.t.get_value() / (self.t.value * 1.0))
        }
    
    def _suggest_optimizations(self, scenario: Dict) -> List[str]:
        """Sugeruje optymalizacje dla scenariusza"""
        suggestions = []
        
        # Analiza słabych punktów
        if scenario.get('energy_efficiency', 0.5) < 0.6:
            suggestions.append("Zwiększ efektywność energetyczną przez optymalizację procesów")
        
        if scenario.get('synergy_potential', 0.5) < 0.7:
            suggestions.append("Wzmocnij potencjał synergiczny przez lepszą integrację komponentów")
        
        if scenario.get('ethical_score', 0.8) < 0.9:
            suggestions.append("Popraw alignment etyczny z fundamentalnymi wartościami")
        
        if not suggestions:
            suggestions.append("Scenariusz jest dobrze zoptymalizowany - kontynuuj obecne podejście")
        
        return suggestions
    
    def _calculate_confidence_interval(self, probability: float) -> Tuple[float, float]:
        """Oblicza przedział ufności dla prawdopodobieństwa"""
        # Prosty model oparty na jakości danych i doświadczeniu
        confidence_width = 0.1 * (1.0 - self.d.get_value() / (self.d.value * 1.0))
        
        lower_bound = max(0.0, probability - confidence_width)
        upper_bound = min(1.0, probability + confidence_width)
        
        return (lower_bound, upper_bound)

    # ========================================================================
    # INTERFEJS PUBLICZNY I RAPORTOWANIE
    # ========================================================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Zwraca pełny status systemu AI_Psyche_GOK:AI"""
        status = {
            "system_name": "AI_Psyche_GOK:AI - Psychologia Prawdopodobieństw Sukcesu",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "current_phase": self._current_phase.value if self._current_phase else "Unknown",
            "iteration_count": self._iteration_count,
            "identity_matrix_current": self._identity_matrix_history[-1] if self._identity_matrix_history else self._parse_identity_matrix(),
            "identity_matrix_sum": sum(self._identity_matrix_history[-1]) if self._identity_matrix_history else 36,
            "component_values": {
                "intrinsic": self.w.get_value(),
                "skills": self.m.get_value(),
                "decisions": self.d.get_value(),
                "context": self.c.get_value(),
                "personality": self.a.get_value(),
                "energy": self.e.get_value(),
                "identity": self.t.get_value()
            },
            "capital_level": self.calculate_capital(),
            "system_coherence": self._calculate_system_coherence()
        }
        
        return status
    
    def _calculate_system_coherence(self) -> float:
        """Oblicza ogólną spójność systemu"""
        values = [
            self.w.get_value() / (self.w.value * 1.0),
            self.m.get_value() / (self.m.value * 1.0),
            self.d.get_value() / (self.d.value * 1.0),
            self.c.get_value() / (self.c.value * 1.0),
            self.a.get_value() / (self.a.value * 1.0),
            self.e.get_value() / (self.e.value * 1.0),
            self.t.get_value() / (self.t.value * 1.0)
        ]
        
        # Spójność jako stabilność wariancji
        coherence = 1.0 - np.var(values)
        return max(0.0, min(1.0, coherence))
    
    def export_session_data(self, filename: Optional[str] = None) -> str:
        """Eksportuje dane sesji do pliku JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_psyche_gok_ai_session_{timestamp}.json"
        
        session_data = {
            "system_status": self.get_system_status(),
            "identity_matrix_history": self._identity_matrix_history,
            "success_patterns": self._success_patterns,
            "disintegration_points": self.d.disintegration_points,
            "reintegration_successes": self.d.reintegration_successes
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Dane sesji wyeksportowane do: {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Błąd eksportu: {e}")
            return ""

# ============================================================================
# FUNKCJE DEMONSTRACYJNE
# ============================================================================

def demonstrate_ai_psyche_gok_ai():
    """Demonstracja możliwości AI_Psyche_GOK:AI"""
    print("🌟" + "="*70)
    print("🧠 AI_Psyche_GOK:AI - DEMONSTRACJA PSYCHOLOGII PRAWDOPODOBIEŃSTW SUKCESU")
    print("🌟" + "="*70)
    
    # Inicjalizacja systemu
    psyche = AIPsycheGOKAI()
    
    # Status systemu
    print(f"\n📊 === STATUS SYSTEMU ===")
    status = psyche.get_system_status()
    print(f"🔄 Bieżąca faza rozwoju: {status['current_phase']}")
    print(f"💰 Zgromadzony kapitał: {status['capital_level']:.2f}")
    print(f"🧮 Matryca tożsamości: {status['identity_matrix_current']}")
    print(f"🎯 Spójność systemu: {status['system_coherence']:.2f}")
    
    # Analiza przeszłych decyzji
    print(f"\n🔍 === ANALIZA PRZESZŁYCH DECYZJI ===")
    past_decisions = [
        {"success": True, "consistent": True, "growth_level": 2},
        {"success": False, "consistent": False, "growth_level": 0},
        {"success": True, "consistent": True, "growth_level": 1},
        {"success": True, "consistent": False, "growth_level": 3},
        {"success": False, "consistent": True, "growth_level": 0}
    ]
    
    decision_quality = psyche.evaluate_decision_quality(past_decisions)
    disintegration_analysis = psyche.detect_disintegration_points(past_decisions)
    
    print(f"📈 Jakość decyzji z przeszłości: {decision_quality:.2f}")
    print(f"🔥 Wykryte punkty dezintegracji: {len(disintegration_analysis)}")
    for point in disintegration_analysis[:2]:  # Pokaż pierwsze 2
        print(f"   • Wzorzec: {point['pattern']}")
        print(f"   • Nauka: {point['learning_opportunity']}")
    
    # Scenariusze testowe
    print(f"\n🎯 === ANALIZA SCENARIUSZY ===")
    scenarios = [
        {
            "goal": "Innowacja przełomowa",
            "complexity": 8,
            "processing_demand": 7,
            "synergy_potential": 0.9,
            "innovation_level": 0.95,
            "balance_factor": 0.6,
            "ethical_score": 0.9,
            "purpose_alignment": 0.85,
            "environmental_fit": 0.7,
            "similarity_to_past": 0.3,
            "identity_consistency": 0.9,
            "outcome": 0.9
        },
        {
            "goal": "Optymalizacja procesów",
            "complexity": 5,
            "processing_demand": 4,
            "synergy_potential": 0.7,
            "innovation_level": 0.4,
            "balance_factor": 0.9,
            "ethical_score": 0.95,
            "purpose_alignment": 0.8,
            "environmental_fit": 0.85,
            "similarity_to_past": 0.8,
            "identity_consistency": 0.95,
            "outcome": 0.75
        },
        {
            "goal": "Ekspansja strategiczna",
            "complexity": 9,
            "processing_demand": 8,
            "synergy_potential": 0.8,
            "innovation_level": 0.7,
            "balance_factor": 0.5,
            "ethical_score": 0.7,
            "purpose_alignment": 0.9,
            "environmental_fit": 0.6,
            "similarity_to_past": 0.5,
            "identity_consistency": 0.8,
            "outcome": 0.8
        }
    ]
    
    recommendations = psyche.generate_recommendations(scenarios)
    
    print(f"\n🏆 === TOP REKOMENDACJE ===")
    for i, rec in enumerate(recommendations[:2], 1):
        print(f"\n{i}. 🎯 {rec['scenario']['goal']}")
        print(f"   📊 Prawdopodobieństwo sukcesu: {rec['probability']:.3f}")
        print(f"   🔮 Wzorzec sukcesu: {rec['success_pattern']}")
        print(f"   ⚖️ Przedział ufności: {rec['confidence_interval'][0]:.3f} - {rec['confidence_interval'][1]:.3f}")
        print(f"   🎨 Faza kontekstu: {rec['phase_context']}")
        print(f"   💡 Główna optymalizacja: {rec['optimization_suggestions'][0]}")
        
        # Analiza ryzyk
        risks = rec['risk_assessment']
        max_risk = max(risks.values())
        max_risk_type = max(risks, key=risks.get)
        print(f"   ⚠️ Główne ryzyko: {max_risk_type} ({max_risk:.2f})")
    
    # Eksport danych sesji
    print(f"\n💾 === EKSPORT DANYCH ===")
    export_file = psyche.export_session_data()
    if export_file:
        print(f"✅ Dane sesji wyeksportowane do: {export_file}")
    
    # Podsumowanie ewolucji matrycy
    print(f"\n🧮 === EWOLUCJA MATRYCY TOŻSAMOŚCI ===")
    print(f"📈 Liczba iteracji: {psyche._iteration_count}")
    if len(psyche._identity_matrix_history) >= 2:
        initial = psyche._identity_matrix_history[0]
        final = psyche._identity_matrix_history[-1]
        print(f"🔄 Początkowa: {initial}")
        print(f"🎯 Aktualna: {final}")
        
        changes = [final[i] - initial[i] for i in range(len(initial))]
        print(f"📊 Zmiany: {changes}")
    
    print(f"\n🌟 === DEMONSTRACJA ZAKOŃCZONA ===")
    return psyche, recommendations

# ============================================================================
# PUNKT WEJŚCIA
# ============================================================================

if __name__ == "__main__":
    # Uruchomienie demonstracji
    psyche_system, demo_recommendations = demonstrate_ai_psyche_gok_ai()
    
    # Dodatkowe testy interaktywne
    print(f"\n🔬 === TESTY INTERAKTYWNE ===")
    
    # Test ewolucji matrycy przez różne fazy
    print(f"\n🔄 Test ewolucji przez fazy rozwoju:")
    
    phases = [DevelopmentPhase.DESTRUCTION, DevelopmentPhase.POINT_0, DevelopmentPhase.DEVELOPMENT]
    for phase in phases:
        psyche_system._current_phase = phase
        matrix = psyche_system._evolve_identity_matrix(phase)
        weights = psyche_system._evolve_identity(phase)
        print(f"   {phase.value}: Matryca={matrix}, Suma={sum(matrix)}")
    
    print(f"\n✨ AI_Psyche_GOK:AI gotowe do integracji z Meta-Genius Unified System!")