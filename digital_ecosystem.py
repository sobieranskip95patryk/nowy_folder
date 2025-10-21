#!/usr/bin/env python3
"""
Digital Ecosystem - Symulacja życia cyfrowego
Eksperyment z ewolucją, AI i emergencją

Autor: GitHub Copilot
Data: 21 października 2025
"""

import random
import time
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class EntityType(Enum):
    GATHERER = "🟢"
    HUNTER = "🔴" 
    REPRODUCER = "🟡"
    HYBRID = "🟣"

@dataclass
class Entity:
    """Byt w cyfrowym ekosystemie"""
    x: float
    y: float
    energy: float
    age: int
    entity_type: EntityType
    intelligence: float
    speed: float
    size: float
    mutation_rate: float = 0.1
    
    def __post_init__(self):
        self.id = random.randint(1000, 9999)
        self.children = 0
        self.kills = 0
        self.resources_gathered = 0
    
    def move(self, world_size: int):
        """Ruch bytu w świecie"""
        direction = random.uniform(0, 2 * math.pi)
        distance = self.speed * (1 + self.intelligence * 0.1)
        
        self.x += math.cos(direction) * distance
        self.y += math.sin(direction) * distance
        
        # Odbicie od granic świata
        self.x = max(0, min(world_size, self.x))
        self.y = max(0, min(world_size, self.y))
        
        # Koszt energetyczny ruchu
        self.energy -= 0.1 * self.speed
    
    def can_reproduce(self) -> bool:
        """Czy byt może się rozmnażać"""
        return self.energy > 50 and self.age > 10
    
    def reproduce(self) -> 'Entity':
        """Reprodukcja z mutacjami"""
        if not self.can_reproduce():
            return None
            
        self.energy -= 30
        self.children += 1
        
        # Mutacje genów
        new_intelligence = self.mutate_trait(self.intelligence, 0.1, 2.0)
        new_speed = self.mutate_trait(self.speed, 0.5, 3.0)
        new_size = self.mutate_trait(self.size, 0.5, 2.0)
        new_mutation_rate = self.mutate_trait(self.mutation_rate, 0.01, 0.5)
        
        # Ewolucja typu
        new_type = self.evolve_type()
        
        child = Entity(
            x=self.x + random.uniform(-5, 5),
            y=self.y + random.uniform(-5, 5),
            energy=25,
            age=0,
            entity_type=new_type,
            intelligence=new_intelligence,
            speed=new_speed,
            size=new_size,
            mutation_rate=new_mutation_rate
        )
        
        return child
    
    def mutate_trait(self, value: float, min_val: float, max_val: float) -> float:
        """Mutacja pojedynczej cechy"""
        if random.random() < self.mutation_rate:
            mutation = random.uniform(-0.2, 0.2) * value
            return max(min_val, min(max_val, value + mutation))
        return value
    
    def evolve_type(self) -> EntityType:
        """Ewolucja typu bytu"""
        if self.intelligence > 1.5 and self.speed > 2.0:
            return EntityType.HYBRID
        elif self.intelligence > 1.2:
            return EntityType.REPRODUCER
        elif self.speed > 2.0:
            return EntityType.HUNTER
        else:
            return EntityType.GATHERER
    
    def hunt(self, target: 'Entity') -> bool:
        """Polowanie na inny byt"""
        if self.entity_type in [EntityType.HUNTER, EntityType.HYBRID]:
            distance = math.sqrt((self.x - target.x)**2 + (self.y - target.y)**2)
            if distance < 3 and self.size > target.size * 0.8:
                self.energy += target.energy * 0.3
                self.kills += 1
                return True
        return False
    
    def gather_resources(self, resources: List[Tuple[float, float]]) -> int:
        """Zbieranie zasobów"""
        gathered = 0
        if self.entity_type in [EntityType.GATHERER, EntityType.HYBRID]:
            for i, (rx, ry) in enumerate(resources):
                distance = math.sqrt((self.x - rx)**2 + (self.y - ry)**2)
                if distance < 2:
                    self.energy += 5
                    self.resources_gathered += 1
                    gathered += 1
                    resources.pop(i)
                    break
        return gathered
    
    def age_one_cycle(self):
        """Starzenie się bytu"""
        self.age += 1
        self.energy -= 1 + self.size * 0.1  # Koszt metabolizmu
        
        # Naturalna śmierć
        if self.age > 100 or self.energy <= 0:
            return False
        return True

class DigitalEcosystem:
    """Główna klasa symulacji ekosystemu"""
    
    def __init__(self, world_size: int = 100, initial_population: int = 20):
        self.world_size = world_size
        self.entities: List[Entity] = []
        self.resources: List[Tuple[float, float]] = []
        self.generation = 0
        self.stats = {
            'total_born': 0,
            'total_died': 0,
            'max_population': 0,
            'avg_intelligence': 0,
            'avg_speed': 0
        }
        
        self.initialize_population(initial_population)
        self.spawn_resources(50)
    
    def initialize_population(self, count: int):
        """Tworzenie początkowej populacji"""
        for _ in range(count):
            entity = Entity(
                x=random.uniform(0, self.world_size),
                y=random.uniform(0, self.world_size),
                energy=random.uniform(30, 50),
                age=random.randint(0, 20),
                entity_type=random.choice(list(EntityType)),
                intelligence=random.uniform(0.5, 1.5),
                speed=random.uniform(0.8, 2.0),
                size=random.uniform(0.8, 1.5)
            )
            self.entities.append(entity)
        
        self.stats['total_born'] = count
    
    def spawn_resources(self, count: int):
        """Dodawanie zasobów do świata"""
        for _ in range(count):
            self.resources.append((
                random.uniform(0, self.world_size),
                random.uniform(0, self.world_size)
            ))
    
    def simulate_cycle(self):
        """Jeden cykl symulacji"""
        if not self.entities:
            return False
        
        # Aktualizacja statystyk
        self.update_stats()
        
        # Ruch i działania bytów
        new_entities = []
        entities_to_remove = []
        
        for entity in self.entities:
            # Ruch
            entity.move(self.world_size)
            
            # Zbieranie zasobów
            entity.gather_resources(self.resources)
            
            # Polowanie
            for target in self.entities:
                if target != entity and entity.hunt(target):
                    if target in self.entities:
                        entities_to_remove.append(target)
            
            # Reprodukcja
            if entity.can_reproduce() and len(self.entities) < 100:
                child = entity.reproduce()
                if child:
                    new_entities.append(child)
                    self.stats['total_born'] += 1
            
            # Starzenie
            if not entity.age_one_cycle():
                entities_to_remove.append(entity)
        
        # Usuwanie zmarłych bytów
        for entity in entities_to_remove:
            if entity in self.entities:
                self.entities.remove(entity)
                self.stats['total_died'] += 1
        
        # Dodawanie nowych bytów
        self.entities.extend(new_entities)
        
        # Spawning zasobów
        if len(self.resources) < 30:
            self.spawn_resources(10)
        
        # Aktualizacja generacji
        if len(self.entities) == 0:
            return False
        
        return True
    
    def update_stats(self):
        """Aktualizacja statystyk"""
        if not self.entities:
            return
            
        pop_size = len(self.entities)
        self.stats['max_population'] = max(self.stats['max_population'], pop_size)
        
        total_intelligence = sum(e.intelligence for e in self.entities)
        total_speed = sum(e.speed for e in self.entities)
        
        self.stats['avg_intelligence'] = total_intelligence / pop_size
        self.stats['avg_speed'] = total_speed / pop_size
    
    def get_status(self) -> str:
        """Status ekosystemu"""
        if not self.entities:
            return "🔴 Ekosystem wymarł!"
        
        type_counts = {}
        for entity_type in EntityType:
            count = sum(1 for e in self.entities if e.entity_type == entity_type)
            type_counts[entity_type] = count
        
        status = f"""
🌍 Digital Ecosystem - Generacja {self.generation}
📊 Populacja: {len(self.entities)} bytów
🍃 Zasoby: {len(self.resources)}

🧬 Rozkład typów:
{EntityType.GATHERER.value} Zbieracze: {type_counts.get(EntityType.GATHERER, 0)}
{EntityType.HUNTER.value} Łowcy: {type_counts.get(EntityType.HUNTER, 0)}
{EntityType.REPRODUCER.value} Reproduktory: {type_counts.get(EntityType.REPRODUCER, 0)}
{EntityType.HYBRID.value} Hybrydy: {type_counts.get(EntityType.HYBRID, 0)}

📈 Statystyki:
• Średnia inteligencja: {self.stats['avg_intelligence']:.2f}
• Średnia prędkość: {self.stats['avg_speed']:.2f}
• Maksymalna populacja: {self.stats['max_population']}
• Całkowicie urodzone: {self.stats['total_born']}
• Całkowicie zmarłe: {self.stats['total_died']}
"""
        return status

if __name__ == "__main__":
    print("🚀 Uruchamianie Digital Ecosystem...")
    print("Eksperyment z ewolucją cyfrową rozpoczęty!")
    print("=" * 50)
    
    ecosystem = DigitalEcosystem(world_size=80, initial_population=15)
    
    try:
        while True:
            # Symulacja jednego cyklu
            if not ecosystem.simulate_cycle():
                print("💀 Wszystkie byty wymarły! Symulacja zakończona.")
                break
            
            ecosystem.generation += 1
            
            # Wyświetlanie statusu co 10 generacji
            if ecosystem.generation % 10 == 0:
                print(ecosystem.get_status())
                print("-" * 50)
            
            # Krótka przerwa dla czytelności
            time.sleep(0.1)
            
            # Ograniczenie do 1000 generacji
            if ecosystem.generation >= 1000:
                print("🏁 Osiągnięto limit 1000 generacji!")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Symulacja przerwana przez użytkownika")
        print(ecosystem.get_status())