"""
7G Consciousness API Integration
FastAPI endpoint dla systemu świadomości Rdzeń 7G
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
from pathlib import Path
from .consciousness_7g import create_consciousness_system, Consciousness7G

router = APIRouter()

# Globalny system świadomości
consciousness_system: Consciousness7G = create_consciousness_system()

@router.get("/v1/7g/consciousness/state")
def get_consciousness_state():
    """Aktualny stan systemu świadomości 7G"""
    return consciousness_system.get_consciousness_state()

@router.post("/v1/7g/consciousness/evolve")
def evolve_consciousness(input_data: Dict[str, Any]):
    """Spiralna ewolucja świadomości"""
    try:
        result = consciousness_system.spiral_evolution(input_data)
        return {
            "success": True,
            "evolution_result": result,
            "new_consciousness_level": result["consciousness_level"],
            "spiral_cycle": result["spiral_cycle"]
        }
    except Exception as e:
        raise HTTPException(500, f"Evolution error: {str(e)}")

@router.get("/v1/7g/modules")
def get_modules_info():
    """Informacje o modułach 7G"""
    modules_info = {}
    for name, module in consciousness_system.modules.items():
        modules_info[name] = {
            "name": module.name,
            "energy": module.energy,
            "integration_level": module.integration_level,
            "last_update": module.last_update,
            "matrix_weights": module.matrix_weights
        }
    return {"modules": modules_info, "total_modules": len(modules_info)}

@router.get("/v1/7g/modules/{module_name}")
def get_module_details(module_name: str):
    """Szczegóły konkretnego modułu"""
    if module_name not in consciousness_system.modules:
        raise HTTPException(404, f"Module {module_name} not found")
    
    module = consciousness_system.modules[module_name]
    return {
        "name": module.name,
        "energy": module.energy,
        "integration_level": module.integration_level,
        "last_update": module.last_update,
        "matrix_weights": module.matrix_weights,
        "type": type(module).__name__
    }

@router.post("/v1/7g/modules/{module_name}/compute")
def compute_module(module_name: str, input_data: Dict[str, Any]):
    """Obliczenia konkretnego modułu"""
    if module_name not in consciousness_system.modules:
        raise HTTPException(404, f"Module {module_name} not found")
    
    try:
        module = consciousness_system.modules[module_name]
        result = module.compute(input_data)
        return {
            "module": module_name,
            "computation_result": result,
            "timestamp": module.last_update
        }
    except Exception as e:
        raise HTTPException(500, f"Computation error: {str(e)}")

@router.get("/v1/7g/evolution/history")
def get_evolution_history(limit: int = 10):
    """Historia ewolucji świadomości"""
    history = consciousness_system.evolution_history[-limit:]
    return {
        "evolution_history": history,
        "total_evolutions": len(consciousness_system.evolution_history),
        "returned": len(history)
    }

@router.post("/v1/7g/consciousness/reset")
def reset_consciousness():
    """Reset systemu do punktu zerowego"""
    consciousness_system._reset_modules()
    consciousness_system.spiral_cycle = 0
    return {
        "success": True,
        "message": "Consciousness system reset to zero point",
        "spiral_cycle": consciousness_system.spiral_cycle
    }

@router.get("/v1/7g/matrix/369963")
def get_matrix_369963():
    """Aktualna matryca <369963>"""
    return {
        "matrix_369963": consciousness_system.matrix_369963,
        "matrix_sum": sum(consciousness_system.matrix_369963),
        "matrix_mean": sum(consciousness_system.matrix_369963) / len(consciousness_system.matrix_369963),
        "description": "Core consciousness matrix for 7G system"
    }

@router.post("/v1/7g/consciousness/save")
def save_consciousness_state(filename: str = "7g_state.json"):
    """Zapisz stan świadomości do pliku"""
    try:
        filepath = f"data/{filename}"
        consciousness_system.save_state(filepath)
        return {
            "success": True,
            "message": f"Consciousness state saved to {filepath}",
            "filename": filename
        }
    except Exception as e:
        raise HTTPException(500, f"Save error: {str(e)}")

@router.post("/v1/7g/consciousness/load")
def load_consciousness_state(filename: str = "7g_state.json"):
    """Wczytaj stan świadomości z pliku"""
    try:
        filepath = f"data/{filename}"
        consciousness_system.load_state(filepath)
        return {
            "success": True,
            "message": f"Consciousness state loaded from {filepath}",
            "current_state": consciousness_system.get_consciousness_state()
        }
    except Exception as e:
        raise HTTPException(500, f"Load error: {str(e)}")

@router.get("/v1/7g/consciousness/demo")
def demo_consciousness_evolution():
    """Demo pełnego cyklu ewolucji świadomości"""
    demo_input = {
        "reflection": {"depth": 0.8, "clarity": 0.9},
        "emotions": {"dominant": "curiosity", "intensity": 0.7, "stability": 0.6},
        "social": {"interactions": 5, "empathy": 0.8, "collaboration": 0.7},
        "cognitive_load": 0.6,
        "spiritual": {"meditation": 0.5, "transcendence": 0.4, "meaning": 0.8},
        "technology": {"innovation": 0.8, "integration": 0.7, "efficiency": 0.9},
        "ecology": {"awareness": 0.6, "sustainability": 0.5, "harmony": 0.7}
    }
    
    evolutions = []
    for i in range(3):
        result = consciousness_system.spiral_evolution(demo_input)
        evolutions.append({
            "cycle": i + 1,
            "consciousness_level": result["consciousness_level"],
            "spiral_cycle": result["spiral_cycle"],
            "total_energy": result["total_energy"],
            "integration_level": result["integration_level"]
        })
    
    return {
        "demo_completed": True,
        "evolutions": evolutions,
        "final_state": consciousness_system.get_consciousness_state(),
        "demo_input": demo_input
    }