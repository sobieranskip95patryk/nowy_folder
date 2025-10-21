#!/usr/bin/env python3
"""
PinkPlay: Multimedialna Oś Czasu 4D - Eksperymentalny Prototyp
Interaktywna wizualizacja transformacji osobistej użytkownika

Wymiary Osi Czasu 4D:
1. Czas (chronologiczny)
2. Intensywność przeżyć/emocji 
3. Typ doświadczenia (seksualność, duchowość, nauka, transformacja)
4. Poziom świadomości/głębokości

UWAGA: Eksperymentalny kod do celów badawczych
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import math

class ExperienceType(Enum):
    """Typy doświadczeń na osi czasu"""
    SEXUALITY = "sexuality"
    SPIRITUALITY = "spirituality" 
    SCIENCE_LEARNING = "science_learning"
    TRANSFORMATION = "transformation"
    RELATIONSHIP = "relationship"
    WELLNESS = "wellness"
    MEDITATION = "meditation"
    CREATIVITY = "creativity"

class MediaType(Enum):
    """Typy mediów w osi czasu"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE_NOTE = "voice_note"
    ARTWORK = "artwork"
    RITUAL_RECORD = "ritual_record"

class PrivacyLevel(Enum):
    """Poziomy prywatności dla wpisów"""
    PRIVATE = "private"  # Tylko ja
    CONNECTIONS = "connections"  # Połączenia/matches
    COMMUNITY = "community"  # Społeczność PinkPlay
    PUBLIC = "public"  # Publiczne

@dataclass
class TimelineEntry:
    """Pojedynczy wpis na osi czasu 4D"""
    entry_id: str
    user_id: str
    timestamp: datetime
    
    # Wymiar 2: Intensywność (0.0 - 1.0)
    emotional_intensity: float
    physical_intensity: float
    spiritual_intensity: float
    
    # Wymiar 3: Typ doświadczenia
    experience_type: ExperienceType
    
    # Wymiar 4: Poziom świadomości/głębokości (0.0 - 1.0)
    consciousness_level: float
    transformation_depth: float
    
    # Treść multimedialna
    title: str
    description: str
    
    # Opcjonalne pola z domyślnymi wartościami
    experience_tags: List[str] = field(default_factory=list)
    media_items: List[Dict] = field(default_factory=list)  # {"type": MediaType, "url": str, "metadata": dict}
    related_entries: List[str] = field(default_factory=list)
    energy_cycle_tag: Optional[str] = None
    moon_phase: Optional[str] = None
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    content_warnings: List[str] = field(default_factory=list)
    personal_insights: str = ""
    growth_indicators: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

@dataclass
class TimelineFilter:
    """Filtry do przeglądania osi czasu"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    experience_types: List[ExperienceType] = field(default_factory=list)
    min_intensity: float = 0.0
    max_intensity: float = 1.0
    min_consciousness: float = 0.0
    max_consciousness: float = 1.0
    tags: List[str] = field(default_factory=list)
    media_types: List[MediaType] = field(default_factory=list)

@dataclass
class Timeline4DVisualization:
    """Konfiguracja wizualizacji osi czasu 4D"""
    view_mode: str = "spiral"  # spiral, linear, circular, wave, mandala
    time_granularity: str = "day"  # hour, day, week, month
    color_scheme: str = "chakra"  # chakra, emotion, intensity, custom
    show_connections: bool = True
    show_patterns: bool = True
    animation_speed: float = 1.0

class Timeline4DSystem:
    """System Multimedialnej Osi Czasu 4D"""
    
    def __init__(self):
        self.entries: Dict[str, TimelineEntry] = {}
        self.user_timelines: Dict[str, List[str]] = {}  # user_id -> [entry_ids]
        self.privacy_settings: Dict[str, Dict] = {}
        
    def add_entry(self, entry: TimelineEntry) -> bool:
        """Dodaj nowy wpis do osi czasu"""
        
        # Walidacja prywatności - Privacy by Design
        if not self._validate_privacy_consent(entry):
            print(f"❌ Nie można dodać wpisu - brak zgody na przetwarzanie danych")
            return False
        
        # Walidacja treści
        if not self._validate_content(entry):
            print(f"❌ Nie można dodać wpisu - treść nie przeszła walidacji")
            return False
        
        self.entries[entry.entry_id] = entry
        
        # Dodaj do timeline użytkownika
        if entry.user_id not in self.user_timelines:
            self.user_timelines[entry.user_id] = []
        self.user_timelines[entry.user_id].append(entry.entry_id)
        
        print(f"✅ Dodano wpis {entry.entry_id} do osi czasu 4D")
        return True
    
    def _validate_privacy_consent(self, entry: TimelineEntry) -> bool:
        """Waliduj zgodę na przetwarzanie danych"""
        # W rzeczywistości sprawdzałoby zgody RODO
        # Tu uproszczenie dla eksperymentu
        return True
    
    def _validate_content(self, entry: TimelineEntry) -> bool:
        """Waliduj bezpieczeństwo treści"""
        # Podstawowa walidacja - w rzeczywistości znacznie bardziej zaawansowana
        if len(entry.description) > 10000:  # Limit znaków
            return False
        
        # Sprawdź ostrzeżenia o treści
        sensitive_keywords = ["harm", "illegal", "non-consent"]
        description_lower = entry.description.lower()
        
        for keyword in sensitive_keywords:
            if keyword in description_lower and keyword not in [cw.lower() for cw in entry.content_warnings]:
                print(f"⚠️ Wykryto potencjalnie wrażliwą treść - wymagane ostrzeżenie")
                return False
        
        return True
    
    def get_timeline(self, user_id: str, filter_config: Optional[TimelineFilter] = None, 
                    requesting_user_id: Optional[str] = None) -> List[TimelineEntry]:
        """Pobierz oś czasu użytkownika z filtrami"""
        
        if user_id not in self.user_timelines:
            return []
        
        entries = []
        for entry_id in self.user_timelines[user_id]:
            entry = self.entries[entry_id]
            
            # Sprawdź uprawnienia dostępu (Privacy by Design)
            if not self._check_access_permissions(entry, requesting_user_id):
                continue
            
            # Zastosuj filtry
            if filter_config and not self._apply_filter(entry, filter_config):
                continue
            
            entries.append(entry)
        
        # Sortuj chronologicznie
        entries.sort(key=lambda x: x.timestamp)
        return entries
    
    def _check_access_permissions(self, entry: TimelineEntry, requesting_user_id: Optional[str]) -> bool:
        """Sprawdź uprawnienia dostępu do wpisu"""
        if requesting_user_id == entry.user_id:
            return True  # Własne wpisy zawsze dostępne
        
        if entry.privacy_level == PrivacyLevel.PRIVATE:
            return False
        
        if entry.privacy_level == PrivacyLevel.PUBLIC:
            return True
        
        # CONNECTIONS i COMMUNITY wymagałyby sprawdzenia relacji
        # Tu uproszczenie dla eksperymentu
        return entry.privacy_level in [PrivacyLevel.COMMUNITY, PrivacyLevel.PUBLIC]
    
    def _apply_filter(self, entry: TimelineEntry, filter_config: TimelineFilter) -> bool:
        """Zastosuj filtry do wpisu"""
        
        # Filtr czasowy
        if filter_config.start_date and entry.timestamp < filter_config.start_date:
            return False
        if filter_config.end_date and entry.timestamp > filter_config.end_date:
            return False
        
        # Filtr typu doświadczenia
        if filter_config.experience_types and entry.experience_type not in filter_config.experience_types:
            return False
        
        # Filtr intensywności (używamy maksymalnej intensywności z trzech wymiarów)
        max_intensity = max(entry.emotional_intensity, entry.physical_intensity, entry.spiritual_intensity)
        if max_intensity < filter_config.min_intensity or max_intensity > filter_config.max_intensity:
            return False
        
        # Filtr świadomości
        if (entry.consciousness_level < filter_config.min_consciousness or 
            entry.consciousness_level > filter_config.max_consciousness):
            return False
        
        # Filtr tagów
        if filter_config.tags:
            if not any(tag in entry.experience_tags for tag in filter_config.tags):
                return False
        
        return True
    
    def analyze_patterns(self, user_id: str) -> Dict:
        """Analizuj wzorce w osi czasu użytkownika"""
        entries = self.get_timeline(user_id, requesting_user_id=user_id)
        
        if not entries:
            return {"error": "Brak danych do analizy"}
        
        # Analiza wzorców intensywności
        intensity_patterns = self._analyze_intensity_patterns(entries)
        
        # Analiza wzorców doświadczeń
        experience_patterns = self._analyze_experience_patterns(entries)
        
        # Analiza wzorców świadomości
        consciousness_patterns = self._analyze_consciousness_patterns(entries)
        
        # Analiza cykliczności
        cyclic_patterns = self._analyze_cyclic_patterns(entries)
        
        return {
            "total_entries": len(entries),
            "time_span_days": (entries[-1].timestamp - entries[0].timestamp).days,
            "intensity_patterns": intensity_patterns,
            "experience_patterns": experience_patterns,
            "consciousness_patterns": consciousness_patterns,
            "cyclic_patterns": cyclic_patterns
        }
    
    def _analyze_intensity_patterns(self, entries: List[TimelineEntry]) -> Dict:
        """Analizuj wzorce intensywności"""
        emotional_values = [e.emotional_intensity for e in entries]
        physical_values = [e.physical_intensity for e in entries]
        spiritual_values = [e.spiritual_intensity for e in entries]
        
        return {
            "emotional": {
                "average": sum(emotional_values) / len(emotional_values),
                "max": max(emotional_values),
                "trend": "increasing" if emotional_values[-1] > emotional_values[0] else "decreasing"
            },
            "physical": {
                "average": sum(physical_values) / len(physical_values),
                "max": max(physical_values),
                "trend": "increasing" if physical_values[-1] > physical_values[0] else "decreasing"
            },
            "spiritual": {
                "average": sum(spiritual_values) / len(spiritual_values),
                "max": max(spiritual_values),
                "trend": "increasing" if spiritual_values[-1] > spiritual_values[0] else "decreasing"
            }
        }
    
    def _analyze_experience_patterns(self, entries: List[TimelineEntry]) -> Dict:
        """Analizuj wzorce typów doświadczeń"""
        type_counts = {}
        for entry in entries:
            exp_type = entry.experience_type
            type_counts[exp_type.value] = type_counts.get(exp_type.value, 0) + 1
        
        # Znajdź dominujący typ
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else None
        
        return {
            "type_distribution": type_counts,
            "dominant_type": dominant_type,
            "diversity_score": len(type_counts) / len(ExperienceType) if type_counts else 0
        }
    
    def _analyze_consciousness_patterns(self, entries: List[TimelineEntry]) -> Dict:
        """Analizuj wzorce poziomu świadomości"""
        consciousness_values = [e.consciousness_level for e in entries]
        transformation_values = [e.transformation_depth for e in entries]
        
        # Trend rozwoju świadomości
        if len(consciousness_values) > 1:
            consciousness_trend = (consciousness_values[-1] - consciousness_values[0]) / len(consciousness_values)
            transformation_trend = (transformation_values[-1] - transformation_values[0]) / len(transformation_values)
        else:
            consciousness_trend = 0
            transformation_trend = 0
        
        return {
            "consciousness": {
                "average": sum(consciousness_values) / len(consciousness_values),
                "current": consciousness_values[-1],
                "growth_trend": consciousness_trend
            },
            "transformation": {
                "average": sum(transformation_values) / len(transformation_values),
                "current": transformation_values[-1],
                "growth_trend": transformation_trend
            }
        }
    
    def _analyze_cyclic_patterns(self, entries: List[TimelineEntry]) -> Dict:
        """Analizuj wzorce cykliczne"""
        # Analiza aktywności według dni tygodnia
        weekday_activity = {}
        for entry in entries:
            weekday = entry.timestamp.strftime("%A")
            weekday_activity[weekday] = weekday_activity.get(weekday, 0) + 1
        
        # Analiza faz księżyca (jeśli dostępne)
        moon_phases = {}
        for entry in entries:
            if entry.moon_phase:
                moon_phases[entry.moon_phase] = moon_phases.get(entry.moon_phase, 0) + 1
        
        return {
            "weekday_activity": weekday_activity,
            "moon_phase_activity": moon_phases,
            "most_active_day": max(weekday_activity, key=weekday_activity.get) if weekday_activity else None
        }
    
    def generate_4d_coordinates(self, entry: TimelineEntry, baseline_time: datetime) -> Tuple[float, float, float, float]:
        """Wygeneruj współrzędne 4D dla wizualizacji"""
        
        # Wymiar 1: Czas (w dniach od baseline)
        time_coord = (entry.timestamp - baseline_time).total_seconds() / (24 * 3600)
        
        # Wymiar 2: Maksymalna intensywność
        intensity_coord = max(entry.emotional_intensity, entry.physical_intensity, entry.spiritual_intensity)
        
        # Wymiar 3: Typ doświadczenia (mapowany na liczbę)
        type_mapping = {t: i for i, t in enumerate(ExperienceType)}
        type_coord = type_mapping.get(entry.experience_type, 0) / len(ExperienceType)
        
        # Wymiar 4: Poziom świadomości
        consciousness_coord = entry.consciousness_level
        
        return (time_coord, intensity_coord, type_coord, consciousness_coord)
    
    def export_for_visualization(self, user_id: str, viz_config: Timeline4DVisualization) -> Dict:
        """Eksportuj dane do wizualizacji 4D"""
        entries = self.get_timeline(user_id, requesting_user_id=user_id)
        
        if not entries:
            return {"error": "Brak danych do wizualizacji"}
        
        baseline_time = entries[0].timestamp
        visualization_data = []
        
        for entry in entries:
            coords_4d = self.generate_4d_coordinates(entry, baseline_time)
            
            viz_point = {
                "id": entry.entry_id,
                "coordinates": coords_4d,
                "title": entry.title,
                "type": entry.experience_type.value,
                "intensity": {
                    "emotional": entry.emotional_intensity,
                    "physical": entry.physical_intensity,
                    "spiritual": entry.spiritual_intensity
                },
                "consciousness": entry.consciousness_level,
                "transformation": entry.transformation_depth,
                "media_count": len(entry.media_items),
                "tags": entry.experience_tags,
                "timestamp": entry.timestamp.isoformat()
            }
            visualization_data.append(viz_point)
        
        return {
            "visualization_config": viz_config,
            "data_points": visualization_data,
            "metadata": {
                "total_points": len(visualization_data),
                "time_span": (entries[-1].timestamp - entries[0].timestamp).days,
                "user_id": user_id
            }
        }

def create_sample_timeline_entries() -> List[TimelineEntry]:
    """Stwórz przykładowe wpisy osi czasu"""
    
    base_time = datetime.now() - timedelta(days=30)
    entries = []
    
    # Wpis 1: Rozpoczęcie podróży tantry
    entry1 = TimelineEntry(
        entry_id="timeline_001",
        user_id="user_123",
        timestamp=base_time,
        emotional_intensity=0.8,
        physical_intensity=0.6,
        spiritual_intensity=0.9,
        experience_type=ExperienceType.SPIRITUALITY,
        experience_tags=["tantra", "beginning", "ceremony"],
        consciousness_level=0.7,
        transformation_depth=0.6,
        title="🕉️ Pierwsza ceremonia tantryczna",
        description="Dziś uczestniczyłem w mojej pierwszej ceremonii tantrycznej. Było to niesamowite doświadczenie obecności i połączenia z własną energią seksualną w świętym kontekście. Poczułem głębokie przebudzenie...",
        media_items=[
            {"type": MediaType.IMAGE.value, "url": "/media/ceremony1.jpg", "metadata": {"taken_by": "self"}},
            {"type": MediaType.VOICE_NOTE.value, "url": "/media/reflection1.wav", "metadata": {"duration": 180}}
        ],
        privacy_level=PrivacyLevel.PRIVATE,
        personal_insights="Odkryłem, że moja energia seksualna może być źródłem głębokiej transformacji duchowej.",
        growth_indicators=["increased_awareness", "energy_sensitivity"],
        energy_cycle_tag="evening_creative",
        moon_phase="new_moon"
    )
    
    # Wpis 2: Połączenie z partnerem
    entry2 = TimelineEntry(
        entry_id="timeline_002", 
        user_id="user_123",
        timestamp=base_time + timedelta(days=7),
        emotional_intensity=0.9,
        physical_intensity=0.8,
        spiritual_intensity=0.8,
        experience_type=ExperienceType.SEXUALITY,
        experience_tags=["connection", "partner", "sacred_sexuality"],
        consciousness_level=0.8,
        transformation_depth=0.7,
        title="💕 Święte połączenie z ukochaną",
        description="Pierwsza praktyka tantryczna z partnerką. Wykorzystaliśmy techniki oddechowe i spojrzenia w oczy. Nigdy wcześniej nie czułem takiej głębi intymności...",
        media_items=[
            {"type": MediaType.ARTWORK.value, "url": "/media/connection_mandala.png", "metadata": {"created_together": True}}
        ],
        privacy_level=PrivacyLevel.CONNECTIONS,
        personal_insights="Intymność może być bramą do transcendencji.",
        growth_indicators=["deeper_intimacy", "partner_connection"],
        energy_cycle_tag="evening_creative"
    )
    
    # Wpis 3: Naukowe odkrycie
    entry3 = TimelineEntry(
        entry_id="timeline_003",
        user_id="user_123", 
        timestamp=base_time + timedelta(days=14),
        emotional_intensity=0.6,
        physical_intensity=0.3,
        spiritual_intensity=0.5,
        experience_type=ExperienceType.SCIENCE_LEARNING,
        experience_tags=["neuroscience", "oxytocin", "research"],
        consciousness_level=0.6,
        transformation_depth=0.5,
        title="🧠 Neurobiologia miłości i związków",
        description="Przeczytałem fascynujący artykuł o neurobiologii związków i roli oksytocyny w budowaniu więzi. To pomaga mi zrozumieć naukowe podstawy tego, czego doświadczam w praktykach...",
        media_items=[
            {"type": MediaType.TEXT.value, "url": "/media/research_notes.md", "metadata": {"word_count": 1500}}
        ],
        privacy_level=PrivacyLevel.COMMUNITY,
        personal_insights="Nauka i duchowość mogą się wzajemnie wzbogacać.",
        growth_indicators=["scientific_understanding"],
        energy_cycle_tag="morning_peak"
    )
    
    entries.extend([entry1, entry2, entry3])
    return entries

if __name__ == "__main__":
    print("🌸 PinkPlay: Multimedialna Oś Czasu 4D")
    print("=" * 50)
    print("🚀 Eksperymentalny prototyp interaktywnej wizualizacji transformacji")
    print("=" * 50)
    
    # Inicjalizuj system
    timeline_system = Timeline4DSystem()
    
    # Dodaj przykładowe wpisy
    sample_entries = create_sample_timeline_entries()
    
    for entry in sample_entries:
        timeline_system.add_entry(entry)
    
    user_id = "user_123"
    print(f"\n✅ Dodano {len(sample_entries)} wpisów do osi czasu użytkownika {user_id}")
    
    # Pobierz pełną oś czasu
    print(f"\n📅 Pełna oś czasu użytkownika:")
    timeline = timeline_system.get_timeline(user_id, requesting_user_id=user_id)
    
    for entry in timeline:
        print(f"\n{entry.timestamp.strftime('%Y-%m-%d')} | {entry.experience_type.value.upper()}")
        print(f"🎯 {entry.title}")
        print(f"💫 Intensywność: E:{entry.emotional_intensity:.1f} F:{entry.physical_intensity:.1f} S:{entry.spiritual_intensity:.1f}")
        print(f"🧠 Świadomość: {entry.consciousness_level:.1f} | Transformacja: {entry.transformation_depth:.1f}")
        print(f"🏷️ Tagi: {', '.join(entry.experience_tags)}")
        print(f"🔒 Prywatność: {entry.privacy_level.value}")
        if entry.media_items:
            print(f"📎 Media: {len(entry.media_items)} plików")
    
    # Analizuj wzorce
    print(f"\n🔍 Analiza wzorców transformacji:")
    patterns = timeline_system.analyze_patterns(user_id)
    
    print(f"📊 Statystyki podstawowe:")
    print(f"   • Całkowita liczba wpisów: {patterns['total_entries']}")
    print(f"   • Okres: {patterns['time_span_days']} dni")
    
    print(f"\n⚡ Wzorce intensywności:")
    intensity = patterns['intensity_patterns']
    print(f"   • Emocjonalna: śr.{intensity['emotional']['average']:.2f} max:{intensity['emotional']['max']:.2f} trend:{intensity['emotional']['trend']}")
    print(f"   • Fizyczna: śr.{intensity['physical']['average']:.2f} max:{intensity['physical']['max']:.2f} trend:{intensity['physical']['trend']}")
    print(f"   • Duchowa: śr.{intensity['spiritual']['average']:.2f} max:{intensity['spiritual']['max']:.2f} trend:{intensity['spiritual']['trend']}")
    
    print(f"\n🎭 Wzorce doświadczeń:")
    experience = patterns['experience_patterns']
    print(f"   • Dominujący typ: {experience['dominant_type']}")
    print(f"   • Różnorodność: {experience['diversity_score']:.2f}")
    print(f"   • Rozkład: {experience['type_distribution']}")
    
    print(f"\n🧠 Wzorce świadomości:")
    consciousness = patterns['consciousness_patterns']
    print(f"   • Średni poziom świadomości: {consciousness['consciousness']['average']:.2f}")
    print(f"   • Trend rozwoju świadomości: {consciousness['consciousness']['growth_trend']:.3f}")
    print(f"   • Trend transformacji: {consciousness['transformation']['growth_trend']:.3f}")
    
    # Eksport do wizualizacji
    print(f"\n🎨 Generowanie danych wizualizacji 4D...")
    viz_config = Timeline4DVisualization(
        view_mode="spiral",
        color_scheme="chakra", 
        show_connections=True,
        show_patterns=True
    )
    
    viz_data = timeline_system.export_for_visualization(user_id, viz_config)
    print(f"✅ Wygenerowano {len(viz_data['data_points'])} punktów danych do wizualizacji")
    
    print(f"\n📐 Przykładowe współrzędne 4D:")
    for i, point in enumerate(viz_data['data_points'][:3]):
        coords = point['coordinates']
        print(f"   {i+1}. {point['title'][:30]}...")
        print(f"      Czas:{coords[0]:.1f} Intensywność:{coords[1]:.2f} Typ:{coords[2]:.2f} Świadomość:{coords[3]:.2f}")
    
    print(f"\n💡 Eksperyment zakończony pomyślnie!")
    print(f"   Oś Czasu 4D oferuje bogaty wgląd w transformację użytkownika")
    print(f"   Dane są gotowe do wizualizacji w interfejsie 3D/VR")
    
    print(f"\n⚠️  Uwagi implementacyjne:")
    print(f"   • Wymagana integracja z systemami mediów (S3, Cloudinary)")
    print(f"   • Potrzebne zaawansowane zabezpieczenia prywatności")
    print(f"   • Konieczna moderacja treści wrażliwych")
    print(f"   • Implementacja zgodności z RODO dla danych osobowych")