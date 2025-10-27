from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import json
import logging
from datetime import datetime
from pathlib import Path
import os
from typing import Optional, Dict, Any
from .auth import verify as verify_token, sign as create_token

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Security Setup
security = HTTPBearer()

def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Require valid JWT token"""
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

def require_admin(user: dict = Depends(require_auth)):
    """Require admin role (MetaGeniusz)"""
    if user.get("role") != "MetaGeniusz":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# Mapa serwisów z ENV lub defaulty
SERVICES = {
    "gok_core": os.getenv("SVC_GOK_CORE", "http://localhost:8001"),
    "migi_core": os.getenv("SVC_MIGI_CORE", "http://localhost:8002"), 
    "hhu": os.getenv("SVC_HHU", "http://localhost:8004"),
    "mta_quest": os.getenv("SVC_MTA_QUEST", "http://localhost:5000")
}

# Telemetry setup
EVENTS_LOG = Path("events.jsonl")

def log_event(event_type: str, data: Dict[str, Any], source: str = None, user_id: str = None):
    """Log event to JSONL file"""
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "data": data,
        "source": source or "gateway",
        "user_id": user_id
    }
    
    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

async def proxy_request(service: str, path: str, method: str = "GET", **kwargs):
    """Proxy request to service with telemetry"""
    base_url = SERVICES.get(service)
    if not base_url:
        raise HTTPException(404, f"Service {service} not found")
    
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    
    # Log request telemetry
    log_event("proxy_request", {
        "service": service,
        "path": path,
        "method": method,
        "url": url
    })
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            if method.upper() == "GET":
                response = await client.get(url, **kwargs)
            elif method.upper() == "POST":
                response = await client.post(url, **kwargs)
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
    """List all available services with status"""
    service_status = {}
    
    for service, url in SERVICES.items():
        try:
            async with httpx.AsyncClient(timeout=3) as client:
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

# === PROXY ENDPOINTS ===

@app.get("/migi/state")
async def migi_state():
    """MIGI system state"""
    return await proxy_request("migi_core", "v1/migi/state")

@app.get("/gok/success")
async def gok_success():
    """GOK success score calculation"""
    return await proxy_request("gok_core", "v1/success_score")

@app.get("/gok/manifest")
async def gok_manifest():
    """GOK system manifest"""
    return await proxy_request("gok_core", "v1/manifest")

@app.get("/hhu/stats")
async def hhu_stats():
    """Hip-Hop Universe statistics"""
    return await proxy_request("hhu", "v1/hhu/stats")

# === MTA QUEST INTEGRATION ===

@app.get("/")
async def root():
    """Redirect to MTA Quest or portal"""
    return {"message": "Meta-Genius Gateway", "portal": "/portal", "mta_quest": SERVICES["mta_quest"]}

# === AUTH ENDPOINTS ===

class TokenRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/token")
async def login_for_access_token(request: TokenRequest):
    """Generate JWT token for authentication"""
    # TODO: SECURITY - Use proper user database and hashed passwords
    # This is DEMO ONLY - DO NOT USE IN PRODUCTION!
    
    # Get credentials from environment variables
    DEMO_USERNAME = os.getenv("DEMO_USERNAME", "admin")
    DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "change-this-password")
    
    if request.username == DEMO_USERNAME and request.password == DEMO_PASSWORD:
        token = create_token({
            "sub": request.username,
            "role": "MetaGeniusz",
            "iat": datetime.now().timestamp()
        })
        
        log_event("auth_login", {
            "username": request.username,
            "role": "MetaGeniusz",
            "success": True
        })
        
        return {
            "access_token": token,
            "token_type": "bearer", 
            "role": "MetaGeniusz",
            "expires_in": 86400
        }
    else:
        log_event("auth_login", {
            "username": request.username,
            "success": False,
            "reason": "invalid_credentials"
        })
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/auth/verify")
async def verify_current_token(user: dict = Depends(require_auth)):
    """Verify current token validity"""
    return {
        "valid": True,
        "user": user.get("sub"),
        "role": user.get("role"),
        "issued_at": user.get("iat")
    }

# === PROTECTED ADMIN ENDPOINTS ===

@app.get("/admin/audit")
async def admin_audit(user: dict = Depends(require_admin)):
    """Admin audit endpoint - requires MetaGeniusz role"""
    log_event("admin_audit", {
        "user": user.get("sub"),
        "accessed_at": datetime.now().isoformat()
    })
    
    # Mock audit data (in real system would fetch from services)
    return {
        "audit_timestamp": datetime.now().isoformat(),
        "auditor": user.get("sub"),
        "system_status": "operational",
        "services_count": len(SERVICES),
        "event_file_exists": EVENTS_LOG.exists(),
        "uptime_check": "gateway_operational"
    }

@app.get("/god")
async def god_panel_access(user: dict = Depends(require_admin)):
    """God panel access - requires admin"""
    with open("portal/god_v11.html", "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/html")

# === TELEMETRY SYSTEM ===

class TelemetryEvent(BaseModel):
    event_type: str
    data: Dict[str, Any]
    source: Optional[str] = None
    user_id: Optional[str] = None

@app.post("/v1/events")
async def log_telemetry_event(event: TelemetryEvent, request: Request):
    """Log telemetry event"""
    try:
        log_event(
            event_type=event.event_type,
            data=event.data,
            source=event.source,
            user_id=event.user_id
        )
        return {"status": "logged", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logging.error(f"Failed to log event: {e}")
        raise HTTPException(status_code=500, detail="Failed to log event")

@app.get("/v1/events")
async def get_recent_events(limit: int = 50):
    """Get recent telemetry events"""
    try:
        if not EVENTS_LOG.exists():
            return {"events": []}
        
        events = []
        with open(EVENTS_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                events.append(json.loads(line.strip()))
        
        return {"events": events}
    except Exception as e:
        logging.error(f"Failed to read events: {e}")
        raise HTTPException(status_code=500, detail="Failed to read events")

@app.post("/api/success-probability")
async def calculate_success(request: Request):
    """Proxy to MTA Quest success calculation"""
    body = await request.json()
    
    # Log telemetry for success calculation
    log_event("success_calculation", {
        "input_data": body,
        "endpoint": "/api/success-probability"
    })
    
    result = await proxy_request("mta_quest", "api/success-probability", "POST", json=body)
    
    # Log result
    log_event("success_result", {
        "result": result,
        "endpoint": "/api/success-probability"
    })
    
    return result

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)