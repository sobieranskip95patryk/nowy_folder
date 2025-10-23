"""
MIGI Core Service
Implementuje system MIGI (CORE/NOVA/SOMA/AETHER/HARMONIA)
"""

from fastapi import FastAPI, HTTPException, Query
from datetime import datetime
import json
from typing import Dict, List
from pathlib import Path
import sys
import os

# Dodajemy core do ścieżki
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.consciousness_api import router as consciousness_router

app = FastAPI(title="MIGI Core Service", version="1.0.0")

# Domyślne wagi MIGI
DEFAULT_MIGI_WEIGHTS = {
    "core": 6.14,      # obliczenia, symulacje, bezpieczeństwo
    "nova": 2.47,      # regeneracja ekosystemów
    "soma": 9.0,       # biosensory, neuro-AI, LifeNet
    "aether": 6.14,    # kontakt kosmiczny
    "harmonia": 2.47   # prawo ewolucyjne, etyka
}

MIGI_DESCRIPTIONS = {
    "core": "Obliczenia kwantowe, symulacje, bezpieczeństwo danych",
    "nova": "Regeneracja ekosystemów, terraformowanie",
    "soma": "Biosensory, neuro-AI, interfejs z biosferą",
    "aether": "Kontakt z cywilizacjami, wymiana danych kosmicznych", 
    "harmonia": "Prawo ewolucyjne, algorytmy współodczuwania i etyki"
}

@app.get("/health")
def health():
    """Service health check"""
    return {
        "status": "ok",
        "service": "migi_core",
        "timestamp": datetime.now().isoformat(),
        "modules": list(DEFAULT_MIGI_WEIGHTS.keys()),
        "version": "1.0.0"
    }

@app.get("/v1/migi/state")
def migi_state():
    """Current MIGI system state with module weights"""
    total_weight = sum(DEFAULT_MIGI_WEIGHTS.values())
    
    state = {}
    for module, weight in DEFAULT_MIGI_WEIGHTS.items():
        state[module] = {
            "weight": weight,
            "normalized": round(weight / 9.0, 3),
            "percentage": round((weight / total_weight) * 100, 1),
            "description": MIGI_DESCRIPTIONS[module],
            "status": "active" if weight > 0 else "inactive"
        }
    
    return {
        "modules": state,
        "total_weight": total_weight,
        "average_weight": round(total_weight / len(DEFAULT_MIGI_WEIGHTS), 2),
        "matrix_369963": [3, 6, 9, 9, 6, 3],
        "timestamp": datetime.now().isoformat(),
        "version": "v1.0.0"
    }

@app.get("/v1/migi/overview")
def migi_overview():
    """MIGI system overview and statistics"""
    active_modules = [k for k, v in DEFAULT_MIGI_WEIGHTS.items() if v > 0]
    dominant_module = max(DEFAULT_MIGI_WEIGHTS.items(), key=lambda x: x[1])
    
    return {
        "system_name": "MIGI - Multi-Intelligence Global Interface",
        "active_modules": len(active_modules),
        "total_modules": len(DEFAULT_MIGI_WEIGHTS),
        "dominant_module": {
            "name": dominant_module[0],
            "weight": dominant_module[1],
            "description": MIGI_DESCRIPTIONS[dominant_module[0]]
        },
        "efficiency": round(sum(DEFAULT_MIGI_WEIGHTS.values()) / (len(DEFAULT_MIGI_WEIGHTS) * 9.0) * 100, 1),
        "last_update": datetime.now().isoformat()
    }

@app.post("/v1/migi/adjust")
def adjust_migi_weights(weights: Dict[str, float]):
    """Adjust MIGI module weights (temporary for session)"""
    try:
        # Walidacja
        for module, weight in weights.items():
            if module not in DEFAULT_MIGI_WEIGHTS:
                raise HTTPException(400, f"Unknown module: {module}")
            if not (0 <= weight <= 9):
                raise HTTPException(400, f"Weight for {module} must be between 0 and 9")
        
        # Symulacja nowych wag (nie zapisujemy stale)
        new_weights = DEFAULT_MIGI_WEIGHTS.copy()
        new_weights.update(weights)
        
        total = sum(new_weights.values())
        
        return {
            "status": "adjusted",
            "new_weights": new_weights,
            "total_weight": total,
            "efficiency": round(total / (len(new_weights) * 9.0) * 100, 1),
            "note": "Weights adjusted for this session only",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Adjustment error: {str(e)}")

@app.get("/v1/migi/transmissions")
def migi_transmissions():
    """Recent MIGI transmissions and signals"""
    return {
        "recent_transmissions": [
            {
                "id": "TRANSMISSION_002",
                "timestamp": "2025-09-02T08:00:00+02:00",
                "phase": "Punkt 0",
                "matrix": [2.47, 6.14, 9.0, 9.0, 6.14, 2.47],
                "energy": "9π",
                "probability_success": 0.85,
                "trace_id": "9267b2b2-bf75-494f-a788-95686733b90b"
            },
            {
                "id": "TRANSMISSION_001", 
                "timestamp": "2025-09-02T07:15:00+02:00",
                "phase": "Rozwój",
                "matrix": [3, 6, 9, 9, 6, 3],
                "energy": "9π",
                "probability_success": 0.78,
                "trace_id": "6ab2ae3c-0b37-4baa-b4a3-e60f00f1c887"
            }
        ],
        "total_transmissions": 2,
        "last_transmission": datetime.now().isoformat()
    }

# RAG Search endpoint
@app.get("/v1/rag/search")
def rag_search(q: str = Query(..., min_length=2), k: int = 3):
    """Search RAG documents"""
    rag_path = Path("data/rag_index.jsonl")
    if not rag_path.exists():
        raise HTTPException(404, "RAG index not found")
    
    docs = []
    with rag_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except Exception:
                pass
    
    if not docs:
        raise HTTPException(404, "RAG index empty")
    
    # Simple scoring: count query terms in text
    def score(query, text):
        query_terms = query.lower().split()
        text_lower = text.lower()
        return sum(1 for term in query_terms if term in text_lower)
    
    # Rank and return top k
    ranked = sorted(docs, key=lambda d: score(q, d.get("text", "")), reverse=True)
    
    results = []
    for doc in ranked[:k]:
        snippet = doc["text"][:360].replace("\n", " ").strip()
        results.append({
            "id": doc["id"], 
            "source": doc["source"], 
            "score": score(q, doc["text"]), 
            "snippet": snippet
        })
    
    return {"query": q, "results": results}

# Dodanie routera świadomości 7G
app.include_router(consciousness_router, tags=["7G Consciousness"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)