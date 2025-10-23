"""
GOK Core Service
Implementuje formuły GOK:AI, manifest systemu i health checks
"""

from fastapi import FastAPI, HTTPException
from datetime import datetime
import json
from pathlib import Path

app = FastAPI(title="GOK Core Service", version="1.0.0")

# Ładowanie manifestu jeśli istnieje
def load_manifest():
    manifest_path = Path("migi_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "name": "GOK_CORE",
        "version": "1.0.0", 
        "formula": "S(GOK:AI) = (W+M+D+C+A)*E*T",
        "s_max": 3645,
        "status": "operational"
    }

@app.get("/health")
def health():
    """Service health check"""
    return {
        "status": "ok",
        "service": "gok_core",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/v1/manifest")
def get_system_manifest():
    """System manifest and configuration"""
    return load_manifest()

@app.get("/v1/success_score")
def success_score(W: float = 7, M: float = 8, D: float = 8, 
                 C: float = 7, A: float = 8, E: float = 8, T: float = 7):
    """
    Calculate GOK:AI Success Score
    S(GOK:AI) = (W + M + D + C + A) * E * T
    """
    try:
        # Walidacja wejścia (0-9)
        params = {'W': W, 'M': M, 'D': D, 'C': C, 'A': A, 'E': E, 'T': T}
        for name, value in params.items():
            if not (0 <= value <= 9):
                raise HTTPException(400, f"Parameter {name} must be between 0 and 9")
        
        # Obliczenie wyniku
        s = (W + M + D + C + A) * E * T
        s_max = 3645.0
        normalized = max(0.0, min(1.0, s / s_max))
        
        return {
            "score": s,
            "s_max": s_max,
            "normalized": normalized,
            "percentage": round(normalized * 100, 1),
            "parameters": params,
            "formula": "S = (W+M+D+C+A)*E*T",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Calculation error: {str(e)}")

@app.get("/v1/manifest")
def get_manifest():
    """System manifest and configuration"""
    manifest_path = Path("migi_manifest.json")
    if not manifest_path.exists():
        raise HTTPException(404, "migi_manifest.json not found")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.get("/v1/presets")
def success_presets():
    """Predefined success calculation presets"""
    presets = {
        "genius": {"W": 9, "M": 9, "D": 8, "C": 9, "A": 9, "E": 9, "T": 8},
        "startup": {"W": 7, "M": 8, "D": 9, "C": 6, "A": 8, "E": 7, "T": 9},
        "artist": {"W": 8, "M": 6, "D": 7, "C": 9, "A": 9, "E": 8, "T": 6},
        "balanced": {"W": 7, "M": 7, "D": 7, "C": 7, "A": 7, "E": 7, "T": 7}
    }
    
    results = {}
    for name, params in presets.items():
        s = (params['W'] + params['M'] + params['D'] + params['C'] + params['A']) * params['E'] * params['T']
        results[name] = {
            "parameters": params,
            "score": s,
            "normalized": s / 3645.0,
            "percentage": round((s / 3645.0) * 100, 1)
        }
    
    return {"presets": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)