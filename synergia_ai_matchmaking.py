#!/usr/bin/env python3
"""
PinkPlay: AI Matchmaking System "Synergia" - Eksperymentalny Prototyp
Implementacja algorytmów hybrydowych dla dopasowania recyprokalnego

Ostrzeżenie: To jest eksperymentalny kod do celów badawczych
Produkcyjna implementacja wymaga znacznie więcej zabezpieczeń i walidacji
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

class EnergeticCycle(Enum):
    """Eksperymentalne cykle energetyczne - wymaga weryfikacji naukowej"""
    MORNING_PEAK = "morning_peak"
    AFTERNOON_STEADY = "afternoon_steady"
    EVENING_CREATIVE = "evening_creative"
    NIGHT_REFLECTIVE = "night_reflective"
    LUNAR_ALIGNED = "lunar_aligned"
    SEASONAL_FLOW = "seasonal_flow"

class SpiritualOrientation(Enum):
    """Orientacje duchowe użytkowników"""
    TANTRA = "tantra"
    MINDFULNESS = "mindfulness" 
    ENERGY_WORK = "energy_work"
    SACRED_SEXUALITY = "sacred_sexuality"
    NATURE_BASED = "nature_based"
    ECLECTIC = "eclectic"
    SCIENTIFIC_SPIRITUAL = "scientific_spiritual"

@dataclass
class UserProfile:
    """Profil użytkownika do matchmakingu"""
    user_id: str
    age: int
    location: Tuple[float, float]  # (lat, lon)
    
    # Podstawowe preferencje
    seeking_genders: List[str]
    relationship_goals: List[str]
    
    # Wymiary seksualności (0-1 scale)
    openness_to_experimentation: float
    communication_style: float  # 0=shy, 1=very_open
    emotional_intimacy_need: float
    physical_touch_preference: float
    
    # Wymiary duchowe
    spiritual_orientations: List[SpiritualOrientation]
    meditation_experience: float  # 0-1 scale
    consciousness_exploration: float
    
    # Nauka i wellness
    health_consciousness: float
    learning_orientation: float
    science_appreciation: float
    
    # Eksperymentalne: "Cykl energetyczny" 
    current_energetic_cycle: EnergeticCycle
    cycle_intensity: float  # 0-1
    
    # Dane behawioralne
    interaction_history: List[str] = field(default_factory=list)
    content_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Prywatność i zgoda
    privacy_level: float = 0.7  # 0=max_privacy, 1=fully_open
    ai_matching_consent: bool = True
    data_sharing_consent: bool = False

@dataclass 
class MatchingFactors:
    """Czynniki wpływające na dopasowanie"""
    compatibility_score: float
    energy_sync: float
    spiritual_alignment: float
    communication_potential: float
    growth_synergy: float
    geographic_proximity: float
    temporal_compatibility: float  # czas aktywności

class SynergiaAI:
    """Główny system AI Matchmaking"""
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self.interaction_matrix = {}  # Dla collaborative filtering
        self.content_features = {}  # Dla content-based filtering
        self.ethical_constraints = {
            'max_age_gap': 15,
            'require_mutual_consent': True,
            'bias_monitoring': True,
            'fairness_enforcement': True
        }
    
    def add_user(self, profile: UserProfile) -> bool:
        """Dodaj użytkownika do systemu"""
        if not profile.ai_matching_consent:
            print(f"❌ Użytkownik {profile.user_id} nie wyraził zgody na AI matching")
            return False
        
        self.users[profile.user_id] = profile
        print(f"✅ Dodano użytkownika {profile.user_id} do systemu Synergia")
        return True
    
    def calculate_compatibility(self, user1: UserProfile, user2: UserProfile) -> MatchingFactors:
        """Oblicz czynniki kompatybilności między użytkownikami"""
        
        # 1. Podstawowa kompatybilność preferencji
        age_compatibility = self._calculate_age_compatibility(user1.age, user2.age)
        gender_compatibility = self._check_gender_compatibility(user1, user2)
        goal_compatibility = self._calculate_goal_overlap(user1.relationship_goals, user2.relationship_goals)
        
        base_compatibility = (age_compatibility + gender_compatibility + goal_compatibility) / 3
        
        # 2. Synchronizacja energetyczna (eksperymentalne)
        energy_sync = self._calculate_energy_sync(user1, user2)
        
        # 3. Alignment duchowy
        spiritual_alignment = self._calculate_spiritual_alignment(user1, user2)
        
        # 4. Potencjał komunikacyjny
        communication_potential = self._calculate_communication_potential(user1, user2)
        
        # 5. Synergia wzrostu
        growth_synergy = self._calculate_growth_synergy(user1, user2)
        
        # 6. Bliskość geograficzna
        geographic_proximity = self._calculate_geographic_proximity(user1, user2)
        
        # 7. Kompatybilność czasowa
        temporal_compatibility = self._calculate_temporal_compatibility(user1, user2)
        
        return MatchingFactors(
            compatibility_score=base_compatibility,
            energy_sync=energy_sync,
            spiritual_alignment=spiritual_alignment,
            communication_potential=communication_potential,
            growth_synergy=growth_synergy,
            geographic_proximity=geographic_proximity,
            temporal_compatibility=temporal_compatibility
        )
    
    def _calculate_energy_sync(self, user1: UserProfile, user2: UserProfile) -> float:
        """
        Eksperymentalne: Oblicz synchronizację cykli energetycznych
        UWAGA: Wymaga weryfikacji naukowej i źródła danych MetaGeniusz OS
        """
        # Mapowanie cykli na wartości liczbowe dla prostoty
        cycle_values = {
            EnergeticCycle.MORNING_PEAK: 0.2,
            EnergeticCycle.AFTERNOON_STEADY: 0.4,
            EnergeticCycle.EVENING_CREATIVE: 0.6,
            EnergeticCycle.NIGHT_REFLECTIVE: 0.8,
            EnergeticCycle.LUNAR_ALIGNED: 0.3,
            EnergeticCycle.SEASONAL_FLOW: 0.5
        }
        
        val1 = cycle_values.get(user1.current_energetic_cycle, 0.5)
        val2 = cycle_values.get(user2.current_energetic_cycle, 0.5)
        
        # Oblicz synchronizację
        cycle_distance = abs(val1 - val2)
        intensity_sync = 1 - abs(user1.cycle_intensity - user2.cycle_intensity)
        
        # Niektóre cykle są komplementarne, inne synergiczne
        complementary_pairs = [
            (EnergeticCycle.MORNING_PEAK, EnergeticCycle.EVENING_CREATIVE),
            (EnergeticCycle.AFTERNOON_STEADY, EnergeticCycle.NIGHT_REFLECTIVE)
        ]
        
        cycle_pair = (user1.current_energetic_cycle, user2.current_energetic_cycle)
        if cycle_pair in complementary_pairs or tuple(reversed(cycle_pair)) in complementary_pairs:
            complementary_bonus = 0.3
        else:
            complementary_bonus = 0
        
        energy_sync = (1 - cycle_distance) * intensity_sync + complementary_bonus
        return min(1.0, energy_sync)
    
    def _calculate_spiritual_alignment(self, user1: UserProfile, user2: UserProfile) -> float:
        """Oblicz alignment duchowy"""
        # Znajdź wspólne orientacje duchowe
        common_orientations = set(user1.spiritual_orientations) & set(user2.spiritual_orientations)
        orientation_overlap = len(common_orientations) / max(len(user1.spiritual_orientations), len(user2.spiritual_orientations))
        
        # Porównaj poziomy doświadczenia
        meditation_compatibility = 1 - abs(user1.meditation_experience - user2.meditation_experience)
        consciousness_compatibility = 1 - abs(user1.consciousness_exploration - user2.consciousness_exploration)
        
        spiritual_alignment = (orientation_overlap + meditation_compatibility + consciousness_compatibility) / 3
        return spiritual_alignment
    
    def _calculate_communication_potential(self, user1: UserProfile, user2: UserProfile) -> float:
        """Oblicz potencjał komunikacyjny"""
        # Style komunikacji - podobne vs komplementarne
        style_diff = abs(user1.communication_style - user2.communication_style)
        
        # Potrzeby intymności emocjonalnej
        intimacy_compatibility = 1 - abs(user1.emotional_intimacy_need - user2.emotional_intimacy_need)
        
        # Otwartość na eksperymenty
        experimentation_sync = 1 - abs(user1.openness_to_experimentation - user2.openness_to_experimentation)
        
        # Balans między podobieństwem a komplementarnością
        style_compatibility = 1 - (style_diff * 0.7)  # Preferujemy podobne style
        
        communication_potential = (style_compatibility + intimacy_compatibility + experimentation_sync) / 3
        return communication_potential
    
    def _calculate_growth_synergy(self, user1: UserProfile, user2: UserProfile) -> float:
        """Oblicz potencjał wzrostu i transformacji"""
        learning_compatibility = 1 - abs(user1.learning_orientation - user2.learning_orientation)
        science_compatibility = 1 - abs(user1.science_appreciation - user2.science_appreciation)
        health_compatibility = 1 - abs(user1.health_consciousness - user2.health_consciousness)
        
        growth_synergy = (learning_compatibility + science_compatibility + health_compatibility) / 3
        return growth_synergy
    
    def _calculate_geographic_proximity(self, user1: UserProfile, user2: UserProfile) -> float:
        """Oblicz bliskość geograficzną"""
        lat1, lon1 = user1.location
        lat2, lon2 = user2.location
        
        # Wzór haversine dla odległości
        R = 6371  # promień Ziemi w km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        # Konwersja na kompatybilność (im bliżej, tym lepiej)
        max_reasonable_distance = 100  # km
        proximity = max(0, 1 - (distance / max_reasonable_distance))
        return proximity
    
    def _calculate_temporal_compatibility(self, user1: UserProfile, user2: UserProfile) -> float:
        """Oblicz kompatybilność czasową aktywności"""
        # Symulacja na podstawie cykli energetycznych
        # W rzeczywistości wymagałoby to danych o aktywności
        
        cycle_timing = {
            EnergeticCycle.MORNING_PEAK: 0.2,
            EnergeticCycle.AFTERNOON_STEADY: 0.5,
            EnergeticCycle.EVENING_CREATIVE: 0.7,
            EnergeticCycle.NIGHT_REFLECTIVE: 0.9,
            EnergeticCycle.LUNAR_ALIGNED: 0.6,
            EnergeticCycle.SEASONAL_FLOW: 0.4
        }
        
        time1 = cycle_timing.get(user1.current_energetic_cycle, 0.5)
        time2 = cycle_timing.get(user2.current_energetic_cycle, 0.5)
        
        temporal_compatibility = 1 - abs(time1 - time2)
        return temporal_compatibility
    
    def _calculate_age_compatibility(self, age1: int, age2: int) -> float:
        """Oblicz kompatybilność wieku"""
        age_diff = abs(age1 - age2)
        max_diff = self.ethical_constraints['max_age_gap']
        
        if age_diff > max_diff:
            return 0.0
        
        return 1 - (age_diff / max_diff)
    
    def _check_gender_compatibility(self, user1: UserProfile, user2: UserProfile) -> float:
        """Sprawdź kompatybilność płci/orientacji"""
        # Symplifikacja - w rzeczywistości bardziej złożone
        # Zakładamy, że seeking_genders zawiera preferowane płci
        
        # Dla uproszczenia zwracamy 1.0 - w rzeczywistości wymagałoby to
        # bardziej sofistykowanej logiki orientacji seksualnej
        return 1.0
    
    def _calculate_goal_overlap(self, goals1: List[str], goals2: List[str]) -> float:
        """Oblicz pokrywanie się celów relacyjnych"""
        if not goals1 or not goals2:
            return 0.0
        
        common_goals = set(goals1) & set(goals2)
        total_unique_goals = set(goals1) | set(goals2)
        
        overlap = len(common_goals) / len(total_unique_goals)
        return overlap
    
    def find_matches(self, user_id: str, limit: int = 10) -> List[Tuple[str, float, MatchingFactors]]:
        """Znajdź dopasowania dla użytkownika"""
        if user_id not in self.users:
            print(f"❌ Użytkownik {user_id} nie znaleziony")
            return []
        
        target_user = self.users[user_id]
        matches = []
        
        for candidate_id, candidate in self.users.items():
            if candidate_id == user_id:
                continue
            
            # Sprawdź zgodę na matching
            if not candidate.ai_matching_consent:
                continue
            
            # Oblicz czynniki kompatybilności
            factors = self.calculate_compatibility(target_user, candidate)
            
            # Oblicz zagregowany wynik
            overall_score = self._calculate_overall_score(factors)
            
            matches.append((candidate_id, overall_score, factors))
        
        # Sortuj według wyniku
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Zastosuj filtry etyczne i sprawiedliwości
        matches = self._apply_ethical_filters(matches)
        
        return matches[:limit]
    
    def _calculate_overall_score(self, factors: MatchingFactors) -> float:
        """Oblicz zagregowany wynik kompatybilności"""
        # Wagi dla różnych czynników - do optymalizacji przez badania
        weights = {
            'compatibility': 0.25,
            'energy_sync': 0.15,  # Eksperymentalne
            'spiritual': 0.20,
            'communication': 0.20,
            'growth': 0.15,
            'geographic': 0.03,
            'temporal': 0.02
        }
        
        score = (
            factors.compatibility_score * weights['compatibility'] +
            factors.energy_sync * weights['energy_sync'] +
            factors.spiritual_alignment * weights['spiritual'] +
            factors.communication_potential * weights['communication'] +
            factors.growth_synergy * weights['growth'] +
            factors.geographic_proximity * weights['geographic'] +
            factors.temporal_compatibility * weights['temporal']
        )
        
        return score
    
    def _apply_ethical_filters(self, matches: List[Tuple[str, float, MatchingFactors]]) -> List[Tuple[str, float, MatchingFactors]]:
        """Zastosuj filtry etyczne i sprawiedliwości"""
        # Tu mogłyby być implementowane mechanizmy:
        # - Zapobieganie bias (np. popularity bias)
        # - Sprawiedliwa dystrybucja rekomendacji
        # - Filtry bezpieczeństwa
        
        if not self.ethical_constraints['bias_monitoring']:
            return matches
        
        # Symulacja prostego filtra różnorodności
        # W rzeczywistości wymagałoby to znacznie bardziej sofistykowanych algorytmów
        return matches
    
    def generate_explanation(self, user_id: str, match_id: str, factors: MatchingFactors) -> str:
        """Wygeneruj wyjaśnienie dla dopasowania (transparentność)"""
        explanation = f"""
🌸 Dlaczego {match_id} to dobre dopasowanie dla Ciebie:

💫 Kompatybilność ogólna: {factors.compatibility_score:.1%}
⚡ Synchronizacja energetyczna: {factors.energy_sync:.1%}
🕉️ Alignment duchowy: {factors.spiritual_alignment:.1%} 
💬 Potencjał komunikacji: {factors.communication_potential:.1%}
📈 Synergia wzrostu: {factors.growth_synergy:.1%}
📍 Bliskość geograficzna: {factors.geographic_proximity:.1%}
🕒 Kompatybilność czasowa: {factors.temporal_compatibility:.1%}

Ta rekomendacja została wygenerowana przez AI Synergia z uwzględnieniem Twoich preferencji, 
wartości duchowych i unikalnego profilu energetycznego.
"""
        return explanation

def create_sample_users() -> List[UserProfile]:
    """Stwórz przykładowych użytkowników do testów"""
    
    # Użytkownik 1: Doświadczony praktyk tantry
    user1 = UserProfile(
        user_id="tantra_master_42",
        age=35,
        location=(52.2297, 21.0122),  # Warszawa
        seeking_genders=["female", "non-binary"],
        relationship_goals=["deep_connection", "spiritual_growth", "conscious_relationship"],
        openness_to_experimentation=0.9,
        communication_style=0.8,
        emotional_intimacy_need=0.9,
        physical_touch_preference=0.7,
        spiritual_orientations=[SpiritualOrientation.TANTRA, SpiritualOrientation.ENERGY_WORK],
        meditation_experience=0.9,
        consciousness_exploration=0.8,
        health_consciousness=0.8,
        learning_orientation=0.9,
        science_appreciation=0.7,
        current_energetic_cycle=EnergeticCycle.EVENING_CREATIVE,
        cycle_intensity=0.8
    )
    
    # Użytkownik 2: Naukowo zorientowana osoba poszukująca duchowości
    user2 = UserProfile(
        user_id="science_seeker_23",
        age=28,
        location=(52.4064, 16.9252),  # Poznań
        seeking_genders=["male", "non-binary"],
        relationship_goals=["learning_together", "conscious_exploration", "authentic_intimacy"],
        openness_to_experimentation=0.7,
        communication_style=0.6,
        emotional_intimacy_need=0.8,
        physical_touch_preference=0.6,
        spiritual_orientations=[SpiritualOrientation.MINDFULNESS, SpiritualOrientation.SCIENTIFIC_SPIRITUAL],
        meditation_experience=0.5,
        consciousness_exploration=0.7,
        health_consciousness=0.9,
        learning_orientation=0.9,
        science_appreciation=0.9,
        current_energetic_cycle=EnergeticCycle.MORNING_PEAK,
        cycle_intensity=0.6
    )
    
    # Użytkownik 3: Eklektyczny poszukiwacz
    user3 = UserProfile(
        user_id="eclectic_explorer_31",
        age=31,
        location=(50.0647, 19.9450),  # Kraków
        seeking_genders=["all"],
        relationship_goals=["adventure", "growth", "authentic_connection"],
        openness_to_experimentation=0.8,
        communication_style=0.7,
        emotional_intimacy_need=0.7,
        physical_touch_preference=0.8,
        spiritual_orientations=[SpiritualOrientation.ECLECTIC, SpiritualOrientation.NATURE_BASED],
        meditation_experience=0.6,
        consciousness_exploration=0.8,
        health_consciousness=0.7,
        learning_orientation=0.8,
        science_appreciation=0.6,
        current_energetic_cycle=EnergeticCycle.LUNAR_ALIGNED,
        cycle_intensity=0.7
    )
    
    return [user1, user2, user3]

if __name__ == "__main__":
    print("🌸 PinkPlay: SEX 7.0 - System AI Matchmaking 'Synergia'")
    print("=" * 60)
    print("⚠️  EKSPERYMENTALNY PROTOTYP - DO CELÓW BADAWCZYCH")
    print("   Rzeczywista implementacja wymaga:")
    print("   • Weryfikacji koncepcji 'cyklu energetycznego'")
    print("   • Zaawansowanych zabezpieczeń prywatności")
    print("   • Audytu etycznego AI")
    print("   • Zgodności z RODO/GDPR")
    print("=" * 60)
    
    # Inicjalizuj system
    synergia = SynergiaAI()
    
    # Dodaj przykładowych użytkowników
    sample_users = create_sample_users()
    for user in sample_users:
        synergia.add_user(user)
    
    print(f"\n✅ Dodano {len(sample_users)} użytkowników do systemu\n")
    
    # Znajdź dopasowania dla pierwszego użytkownika
    target_user_id = sample_users[0].user_id
    print(f"🔍 Szukam dopasowań dla: {target_user_id}")
    
    matches = synergia.find_matches(target_user_id, limit=5)
    
    print(f"\n🎯 Znaleziono {len(matches)} potencjalnych dopasowań:\n")
    
    for i, (match_id, score, factors) in enumerate(matches, 1):
        print(f"{i}. {match_id} (wynik: {score:.1%})")
        
        # Wygeneruj wyjaśnienie
        explanation = synergia.generate_explanation(target_user_id, match_id, factors)
        print(explanation)
        print("-" * 50)
    
    print("\n🔬 Analiza czynników kompatybilności:")
    if matches:
        _, _, factors = matches[0]
        print(f"• Kompatybilność podstawowa: {factors.compatibility_score:.1%}")
        print(f"• Sync energetyczny: {factors.energy_sync:.1%}")
        print(f"• Alignment duchowy: {factors.spiritual_alignment:.1%}")
        print(f"• Potencjał komunikacji: {factors.communication_potential:.1%}")
        print(f"• Synergia wzrostu: {factors.growth_synergy:.1%}")
        print(f"• Bliskość geograficzna: {factors.geographic_proximity:.1%}")
        print(f"• Kompatybilność czasowa: {factors.temporal_compatibility:.1%}")
    
    print("\n⚠️  UWAGI TECHNICZNE:")
    print("1. 'Cykl energetyczny' wymaga weryfikacji naukowej")
    print("2. Integracja z MetaGeniusz OS niewyjaśniona") 
    print("3. Potrzebne zaawansowane zabezpieczenia prywatności")
    print("4. Wymagany audyt etyczny algorytmów")
    print("5. Implementacja zgodności z RODO/GDPR")
    
    print("\n💡 Eksperyment zakończony pomyślnie!")
    print("   Kod może służyć jako punkt wyjścia do dalszych badań")
    print("   nad algorytmami matchmakingu dla PinkPlay")