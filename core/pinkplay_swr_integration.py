# 🧠 SWR Integration Module for PinkPlayEvo
# Moduł Świadomego Wnioskowania Resztkowego - Integracja z PinkPlayEvo

"""
SWR (Świadome Wnioskowanie Resztkowe) Integration for PinkPlayEvo
Conscious Residual Inference Module for Emotional AI Video Generation

Integruje się z pipeline PinkPlayEvo jako middleware między promptEngine.js a productionGenerator.js
Analizuje fabuły użytkowników pod kątem resztek emocjonalnych i logicznych,
mutuje prompty dla lepszej spójności narracyjnej.

Autor: Meta-Geniusz® System
Data: 26 października 2025
Wersja: SWR-PinkPlayEvo Integration v1.0
"""

import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Import MŚWR core
from .mswr_v2_clean import ConsciousResidualInferenceModule, create_mswr_system

class PinkPlaySWR:
    """
    SWR Integration dla PinkPlayEvo
    Analizuje fabuły i mutuje prompty dla lepszej spójności emocjonalnej
    """
    
    def __init__(self):
        self.mswr = create_mswr_system()
        self.story_history = []
        self.mutation_patterns = {
            "emotional_boost": [
                "z wzmocnionym dramatyzmem i kontrastem emocjonalnym",
                "z głębszym rezonansem emocjonalnym",
                "z intensywniejszą ekspresją uczuć"
            ],
            "narrative_coherence": [
                "z lepszą spójnością narracyjną",
                "z jaśniejszym przekazem fabularnym",
                "z wzmocnioną dramaturgią"
            ],
            "visual_enhancement": [
                "z bogatszą paletą wizualną",
                "z wzmocnionym kontrastem wizualnym",
                "z dynamiczniejszą kompozycją"
            ]
        }
        
    def analyze_story_sentiment(self, story: str) -> Dict[str, Any]:
        """Analizuje sentyment fabuły"""
        # Prosta analiza sentymentu (można rozbudować o zewnętrzne API)
        emotional_words = {
            "positive": ["miłość", "radość", "szczęście", "nadzieja", "sukces", "zwycięstwo"],
            "negative": ["smutek", "ból", "strach", "lęk", "porażka", "śmierć"],
            "intense": ["pasja", "gniew", "ekstaza", "desperacja", "obsesja", "szaleństwo"],
            "neutral": ["praca", "dom", "droga", "książka", "komputer", "jedzenie"]
        }
        
        story_lower = story.lower()
        sentiment_scores = {}
        
        for category, words in emotional_words.items():
            score = sum(1 for word in words if word in story_lower)
            sentiment_scores[category] = score
        
        # Oblicz dominujący sentyment
        dominant = max(sentiment_scores, key=sentiment_scores.get)
        total_words = len(story.split())
        emotional_density = sum(sentiment_scores.values()) / max(total_words, 1)
        
        return {
            "dominant_sentiment": dominant,
            "sentiment_scores": sentiment_scores,
            "emotional_density": emotional_density,
            "story_length": total_words
        }
    
    def detect_narrative_residuals(self, story: str, sentiment_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Wykrywa resztki narracyjne w fabule"""
        residuals = []
        
        # Resztka 1: Niska gęstość emocjonalna
        if sentiment_analysis["emotional_density"] < 0.1:
            residuals.append({
                "type": "low_emotional_density",
                "severity": 0.7,
                "description": f"Niska gęstość emocjonalna: {sentiment_analysis['emotional_density']:.3f}",
                "suggestion": "Dodaj więcej ekspresyjnych słów lub kontrastów emocjonalnych"
            })
        
        # Resztka 2: Brak akcji/czasowników
        action_words = ["biega", "walczy", "tańczy", "śpiewa", "płacze", "śmieje", "krzyczy"]
        action_count = sum(1 for word in action_words if word in story.lower())
        if action_count == 0:
            residuals.append({
                "type": "lack_of_action",
                "severity": 0.5,
                "description": "Brak słów akcji - może wpłynąć na dynamikę wizualną",
                "suggestion": "Dodaj czasowniki ruchu lub akcji"
            })
        
        # Resztka 3: Zbyt długa fabuła (problem z prompt tokenami)
        if len(story.split()) > 100:
            residuals.append({
                "type": "story_too_long",
                "severity": 0.6,
                "description": f"Fabuła zbyt długa: {len(story.split())} słów",
                "suggestion": "Skróć do maksymalnie 50-80 słów dla lepszej konwersji na prompty"
            })
        
        # Resztka 4: Niespójność sentymentu (mieszane emocje bez kontekstu)
        sentiment_scores = sentiment_analysis["sentiment_scores"]
        conflicting_emotions = sum(1 for score in sentiment_scores.values() if score > 0)
        if conflicting_emotions > 2 and sentiment_analysis["emotional_density"] > 0.3:
            residuals.append({
                "type": "sentiment_conflict",
                "severity": 0.4,
                "description": "Konflikt sentymentów może wpłynąć na spójność wizualną",
                "suggestion": "Wybierz dominującą emocję lub dodaj kontekst przejścia"
            })
        
        return residuals
    
    def mutate_story_prompt(self, story: str, residuals: List[Dict[str, Any]]) -> str:
        """Mutuje prompt na podstawie wykrytych resztek"""
        if not residuals:
            return story
        
        mutated_story = story
        applied_mutations = []
        
        for residual in residuals:
            residual_type = residual["type"]
            
            if residual_type == "low_emotional_density":
                # Dodaj wzmocnienie emocjonalne
                boost = self.mutation_patterns["emotional_boost"][0]
                mutated_story = f"{mutated_story} {boost}"
                applied_mutations.append("emotional_boost")
                
            elif residual_type == "lack_of_action":
                # Dodaj sugestię ruchu
                if "tańczy" not in mutated_story.lower():
                    mutated_story = mutated_story.replace(".", ", poruszając się dynamicznie.")
                    applied_mutations.append("action_enhancement")
                    
            elif residual_type == "story_too_long":
                # Skróć do kluczowych elementów (prosta heurystyka)
                words = mutated_story.split()
                if len(words) > 80:
                    mutated_story = " ".join(words[:80]) + "..."
                    applied_mutations.append("length_reduction")
                    
            elif residual_type == "sentiment_conflict":
                # Dodaj kontekst spójności
                coherence = self.mutation_patterns["narrative_coherence"][0]
                mutated_story = f"{mutated_story} {coherence}"
                applied_mutations.append("coherence_boost")
        
        return mutated_story
    
    def process_story_for_pinkplay(self, story: str, user_id: str = None) -> Dict[str, Any]:
        """
        Główna funkcja przetwarzania fabuły dla PinkPlayEvo
        
        Args:
            story: Fabuła użytkownika
            user_id: ID użytkownika (opcjonalne)
            
        Returns:
            Dict z wynikami analizy SWR i zmutowanym promptem
        """
        processing_start = time.time()
        
        # Analiza sentymentu
        sentiment_analysis = self.analyze_story_sentiment(story)
        
        # Wykrycie resztek narracyjnych
        residuals = self.detect_narrative_residuals(story, sentiment_analysis)
        
        # Analiza MŚWR (głębsza analiza poznawcza)
        mswr_result = self.mswr.zero_time_inference(
            story, 
            {
                "creative": True,
                "pinkplay_context": True,
                "sentiment": sentiment_analysis
            }
        )
        
        # Mutacja promptu
        mutated_story = story
        if residuals:
            mutated_story = self.mutate_story_prompt(story, residuals)
        
        # Kalkulacja jakości
        quality_score = self._calculate_story_quality(story, sentiment_analysis, residuals)
        
        # Log historii
        session_record = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "original_story": story,
            "mutated_story": mutated_story,
            "sentiment_analysis": sentiment_analysis,
            "residuals_found": len(residuals),
            "residuals": residuals,
            "mswr_analysis": {
                "probability_score": mswr_result["probability_score"],
                "residual_entropy": mswr_result["residual_entropy"],
                "zero_time_achieved": mswr_result["zero_time_achieved"]
            },
            "quality_score": quality_score,
            "processing_time_ms": (time.time() - processing_start) * 1000
        }
        
        self.story_history.append(session_record)
        
        return {
            "original_story": story,
            "enhanced_story": mutated_story,
            "sentiment_analysis": sentiment_analysis,
            "residuals": residuals,
            "quality_score": quality_score,
            "mswr_insights": {
                "probability": mswr_result["probability_score"],
                "entropy": mswr_result["residual_entropy"],
                "cognitive_coherence": mswr_result.get("narrative_reframing", {}).get("narrative_improvement_score", 0)
            },
            "recommendations": self._generate_recommendations(residuals, sentiment_analysis),
            "ready_for_generation": quality_score > 0.7,
            "processing_metadata": {
                "processing_time_ms": session_record["processing_time_ms"],
                "session_id": mswr_result.get("session_id", "unknown")
            }
        }
    
    def _calculate_story_quality(self, story: str, sentiment_analysis: Dict[str, Any], residuals: List[Dict[str, Any]]) -> float:
        """Oblicza jakość fabuły (0.0 - 1.0)"""
        base_quality = 0.8
        
        # Penalty za resztki
        residual_penalty = len(residuals) * 0.1
        
        # Bonus za dobrą gęstość emocjonalną
        emotional_bonus = min(0.2, sentiment_analysis["emotional_density"] * 0.5)
        
        # Penalty za zbyt krótkie/długie fabuły
        word_count = sentiment_analysis["story_length"]
        length_penalty = 0
        if word_count < 10:
            length_penalty = 0.3
        elif word_count > 100:
            length_penalty = 0.2
        
        final_quality = base_quality - residual_penalty + emotional_bonus - length_penalty
        return max(0.0, min(1.0, final_quality))
    
    def _generate_recommendations(self, residuals: List[Dict[str, Any]], sentiment_analysis: Dict[str, Any]) -> List[str]:
        """Generuje rekomendacje dla użytkownika"""
        recommendations = []
        
        if sentiment_analysis["emotional_density"] < 0.1:
            recommendations.append("💡 Dodaj więcej emocjonalnych słów dla lepszego wpływu wizualnego")
        
        if sentiment_analysis["story_length"] < 10:
            recommendations.append("📝 Rozszerz fabułę - dodaj szczegóły dla bogatszego wideo")
        elif sentiment_analysis["story_length"] > 100:
            recommendations.append("✂️ Skróć fabułę - zbyt długie opisy mogą ograniczyć jakość generacji")
        
        for residual in residuals:
            if residual["severity"] > 0.6:
                recommendations.append(f"⚠️ {residual['suggestion']}")
        
        if not recommendations:
            recommendations.append("✅ Fabuła gotowa do generacji! Dobra struktura narracyjna.")
        
        return recommendations
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Zwraca podsumowanie analityki SWR"""
        if not self.story_history:
            return {"message": "Brak danych analitycznych"}
        
        total_stories = len(self.story_history)
        avg_quality = sum(record["quality_score"] for record in self.story_history) / total_stories
        total_residuals = sum(record["residuals_found"] for record in self.story_history)
        avg_processing_time = sum(record["processing_time_ms"] for record in self.story_history) / total_stories
        
        # Analiza typów resztek
        residual_types = {}
        for record in self.story_history:
            for residual in record["residuals"]:
                rtype = residual["type"]
                residual_types[rtype] = residual_types.get(rtype, 0) + 1
        
        return {
            "total_stories_processed": total_stories,
            "average_quality_score": avg_quality,
            "total_residuals_found": total_residuals,
            "average_residuals_per_story": total_residuals / total_stories,
            "average_processing_time_ms": avg_processing_time,
            "most_common_residuals": sorted(residual_types.items(), key=lambda x: x[1], reverse=True)[:5],
            "high_quality_stories_ratio": len([r for r in self.story_history if r["quality_score"] > 0.8]) / total_stories
        }
    
    def export_analytics(self, filepath: str = None) -> str:
        """Eksportuje analitykę do pliku JSON"""
        if not filepath:
            filepath = f"pinkplay_swr_analytics_{int(time.time())}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "swr_version": "PinkPlayEvo Integration v1.0",
            "summary": self.get_analytics_summary(),
            "detailed_history": self.story_history
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            return ""


# Factory function
def create_pinkplay_swr() -> PinkPlaySWR:
    """Factory function dla PinkPlayEvo SWR"""
    return PinkPlaySWR()


# Testowanie
if __name__ == "__main__":
    print("🎭 =================================================================")
    print("🎭     SWR INTEGRATION FOR PINKPLAYEVO - TESTING")
    print("🎭 =================================================================")
    
    # Inicjalizacja
    swr = create_pinkplay_swr()
    
    # Test Cases
    test_stories = [
        "Młoda kobieta tańczy w deszczu, czując wolność i radość życia.",
        "Bohater walczy z demonami wewnętrznymi w ciemnym labiryncie swojego umysłu, szukając światła nadziei.",
        "Kot śpi na parapecie. Jest ładna pogoda. Słońce świeci.",
        "Bardzo długa historia o bohaterze który podróżuje przez wiele krajów i spotyka różnych ludzi i przeżywa różne przygody i uczucia i emocje i ma różne doświadczenia życiowe które go kształtują jako człowieka i pomagają mu zrozumieć sens życia i odnaleźć swoje miejsce w świecie który jest pełen wyzwań i możliwości rozwoju osobistego."
    ]
    
    print("\n🧪 Testowanie różnych typów fabul...")
    
    for i, story in enumerate(test_stories, 1):
        print(f"\n{'='*60}")
        print(f"🔬 TEST {i}: {story[:50]}...")
        print('='*60)
        
        result = swr.process_story_for_pinkplay(story, f"test_user_{i}")
        
        print(f"📊 Jakość: {result['quality_score']:.3f}")
        print(f"🔍 Resztki: {len(result['residuals'])}")
        print(f"💭 Sentyment: {result['sentiment_analysis']['dominant_sentiment']}")
        print(f"⚡ Gotowy do generacji: {result['ready_for_generation']}")
        print(f"📝 Zmutowany prompt: {result['enhanced_story'][:100]}...")
        
        if result['recommendations']:
            print("💡 Rekomendacje:")
            for rec in result['recommendations']:
                print(f"   {rec}")
    
    # Analityka
    print(f"\n{'='*60}")
    print("📈 ANALITYKA SWR")
    print('='*60)
    
    analytics = swr.get_analytics_summary()
    for key, value in analytics.items():
        if isinstance(value, float):
            print(f"📊 {key}: {value:.3f}")
        else:
            print(f"📊 {key}: {value}")
    
    # Eksport
    export_path = swr.export_analytics()
    if export_path:
        print(f"\n✅ Analityka wyeksportowana: {export_path}")
    
    print(f"\n🎯 SWR Integration dla PinkPlayEvo gotowa!")
    print(f"🚀 Moduł może być zintegrowany z Node.js pipeline")