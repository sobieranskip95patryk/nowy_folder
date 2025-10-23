"""
Meta-Genius Unified Gateway
FastAPI gateway łączący wszystkie serwisy ekosystemu

Główny punkt wejścia do całego Meta-Genius Digital Empire
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import json
import logging
from datetime import datetime
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Meta-Genius Unified Gateway",
    description="Centralny gateway dla ekosystemu Meta-Genius",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji ogranicz do swoich domen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ładowanie konfiguracji z workspace.json
def load_workspace_config():
    """Load workspace configuration"""
    workspace_file = Path(__file__).parent / "workspace.json"
    if workspace_file.exists():
        with open(workspace_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"repos": [], "services_map": {}}

WORKSPACE = load_workspace_config()

# Mapa serwisów z portami
SERVICES = {}
for repo in WORKSPACE.get("repos", []):
    if "service" in repo and "port" in repo:
        SERVICES[repo["service"]] = f"http://localhost:{repo['port']}"

# Role i autoryzacja (z wcześniejszej implementacji)
class Role:
    METAGENIUSZ = "MetaGeniusz"
    GPT_PRESIDENT = "GPT-President" 
    GPT_KING = "GPT-King"
    GPT_ORGANIZATION = "GPT-Organization"
    USER = "User"

ROLE_HIERARCHY = {
    Role.METAGENIUSZ: 5,
    Role.GPT_PRESIDENT: 4,
    Role.GPT_KING: 3,
    Role.GPT_ORGANIZATION: 2,
    Role.USER: 1
}

def check_permission(user_role: str, required_role: str) -> bool:
    """Check if user has required permissions"""
    user_level = ROLE_HIERARCHY.get(user_role, 1)
    required_level = ROLE_HIERARCHY.get(required_role, 1)
    return user_level >= required_level

async def proxy_request(service: str, path: str, method: str = "GET", **kwargs):
    """Proxy request to service"""
    base_url = SERVICES.get(service)
    if not base_url:
        raise HTTPException(404, f"Service {service} not found")
    
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if method.upper() == "GET":
                response = await client.get(url, **kwargs)
            elif method.upper() == "POST":
                response = await client.post(url, **kwargs)
            elif method.upper() == "PUT":
                response = await client.put(url, **kwargs)
            elif method.upper() == "DELETE":
                response = await client.delete(url, **kwargs)
            else:
                raise HTTPException(405, f"Method {method} not supported")
            
            return response.json()
    except httpx.ConnectError:
        raise HTTPException(503, f"Service {service} unavailable")
    except Exception as e:
        logger.error(f"Proxy error for {service}: {e}")
        raise HTTPException(500, f"Proxy error: {str(e)}")

# === HEALTH & STATUS ===

@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "services": list(SERVICES.keys()),
        "version": "1.0.0"
    }

@app.get("/services")
async def list_services():
    """List all available services"""
    service_status = {}
    
    for service, url in SERVICES.items():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{url}/health")
                service_status[service] = {
                    "status": "online",
                    "url": url,
                    "response_time": response.elapsed.total_seconds()
                }
        except Exception:
            service_status[service] = {
                "status": "offline",
                "url": url,
                "response_time": None
            }
    
    return {
        "services": service_status,
        "total": len(SERVICES),
        "online": len([s for s in service_status.values() if s["status"] == "online"])
    }

# === GOD INTERFACE ===

@app.get("/god/dashboard")
async def god_dashboard(x_role: str = Header("User")):
    """God Interface dashboard - tylko dla MetaGeniusz"""
    if not check_permission(x_role, Role.METAGENIUSZ):
        raise HTTPException(403, "Access denied - MetaGeniusz role required")
    
    return await proxy_request("god_interface", "dashboard")

@app.get("/god/migi/state")
async def god_migi_state(x_role: str = Header("User")):
    """MIGI system state"""
    if not check_permission(x_role, Role.GPT_KING):
        raise HTTPException(403, "Access denied - GPT-King role or higher required")
    
    return await proxy_request("migi_core", "state")

# === MTA QUEST (główny produkt) ===

@app.get("/")
async def root():
    """Redirect to MTA Quest landing page"""
    return await proxy_request("mta_quest", "")

@app.post("/api/success-probability")
async def calculate_success(request: Request):
    """Proxy to MTA Quest success calculation"""
    body = await request.json()
    return await proxy_request("mta_quest", "api/success-probability", "POST", json=body)

@app.post("/api/quick-insight")
async def quick_insight(request: Request):
    """Proxy to MTA Quest quick insight"""
    body = await request.json()
    return await proxy_request("mta_quest", "api/quick-insight", "POST", json=body)

# === CONTENT SERVICES ===

@app.get("/mixtape/latest")
async def mixtape_latest():
    """Latest GOK mixtape content"""
    return await proxy_request("mixtape", "latest")

@app.get("/hhu/tracks")
async def hip_hop_tracks():
    """Hip-hop universe tracks"""
    return await proxy_request("hhu", "tracks")

@app.get("/rfg/gallery")
async def rfg_gallery(x_role: str = Header("User")):
    """Rocket Fuel Girls gallery - age verification required"""
    # Tu powinna być weryfikacja wieku w prawdziwej implementacji
    return await proxy_request("rfg", "gallery")

# === ECONOMY ===

@app.get("/drift/balance")
async def drift_balance(user_id: str):
    """Check Drift Money balance"""
    return await proxy_request("drift", f"balance/{user_id}")

@app.post("/drift/transfer")
async def drift_transfer(request: Request):
    """Transfer Drift tokens"""
    body = await request.json()
    return await proxy_request("drift", "transfer", "POST", json=body)

# === PORTFOLIO & SHOWCASE ===

@app.get("/portfolio/projects")
async def portfolio_projects():
    """Portfolio projects list"""
    return await proxy_request("portfolio", "projects")

@app.get("/global-vision/overview")
async def global_vision():
    """Global ecosystem overview"""
    return await proxy_request("global_vision", "overview")

# === ADMIN ENDPOINTS ===

@app.get("/admin/audit")
async def admin_audit(x_role: str = Header("User")):
    """System audit - for GPT-King and above"""
    if not check_permission(x_role, Role.GPT_KING):
        raise HTTPException(403, "Access denied")
    
    audit_data = {
        "timestamp": datetime.now().isoformat(),
        "services_status": {},
        "role_accessed": x_role
    }
    
    # Check each service health
    for service, url in SERVICES.items():
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                await client.get(f"{url}/health")
                audit_data["services_status"][service] = "healthy"
        except Exception:
            audit_data["services_status"][service] = "unhealthy"
    
    return audit_data

@app.get("/admin/ecosystem-status")
async def ecosystem_status(x_role: str = Header("User")):
    """Full ecosystem status"""
    if not check_permission(x_role, Role.GPT_PRESIDENT):
        raise HTTPException(403, "Access denied")
    
    return {
        "workspace_config": WORKSPACE,
        "active_services": len(SERVICES),
        "total_repos": len(WORKSPACE.get("repos", [])),
        "role_accessed": x_role,
        "timestamp": datetime.now().isoformat()
    }

# === ERROR HANDLERS ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# === STARTUP ===

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Meta-Genius Unified Gateway starting...")
    logger.info(f"📡 Configured services: {list(SERVICES.keys())}")
    logger.info("✅ Gateway ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)