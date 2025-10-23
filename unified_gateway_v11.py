"""
Meta-Genius Unified Gateway - Enhanced Version
Combining current working implementation with improved architecture
"""

import uvicorn
import json
import time
import httpx
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import os

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Import working JWT functions from current system
try:
    from .auth import verify as verify_token, sign as create_token
except ImportError:
    # Fallback for standalone testing
    import time, hmac, hashlib, base64, json, os
    SECRET = os.getenv("JWT_SECRET", "dev-secret")
    
    def create_token(payload: dict, exp_sec: int = 3600):
        p = payload | {"exp": int(time.time()) + exp_sec}
        raw = base64.urlsafe_b64encode(json.dumps(p).encode()).rstrip(b"=")
        sig = hmac.new(SECRET.encode(), raw, hashlib.sha256).digest()
        tok = raw + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")
        return tok.decode()
    
    def verify_token(token: str):
        try:
            raw, sig = token.split(".")
            expect = base64.urlsafe_b64encode(hmac.new(SECRET.encode(), raw.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
            if sig != expect: return None
            data = json.loads(base64.urlsafe_b64decode(raw + "=="))
            if data.get("exp", 0) < int(time.time()): return None
            return data
        except Exception:
            return None

# --- Core Constants ---
GATEWAY_PORT = 8800
ALLOWED_ROLES = ["MetaGeniusz", "Admin", "User", "Guest"]
ROLE_PRIORITY = {"MetaGeniusz": 4, "Admin": 3, "User": 2, "Guest": 1}

# Service mapping
SERVICES = {
    "gok_core": os.getenv("SVC_GOK_CORE", "http://localhost:8001"),
    "migi_core": os.getenv("SVC_MIGI_CORE", "http://localhost:8002"), 
    "hhu": os.getenv("SVC_HHU", "http://localhost:8004"),
    "mta_quest": os.getenv("SVC_MTA_QUEST", "http://localhost:5000")
}

# Telemetry setup
EVENTS_LOG = Path("events.jsonl")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Enhanced Telemetry System ---
def log_telemetry_event(event_type: str, data: Dict[str, Any], user_id: str = None, source: str = "gateway"):
    """Enhanced JSONL logging with structured data"""
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "data": data,
        "source": source,
        "user_id": user_id,
        "unix_time": time.time()
    }
    
    # Write to JSONL file
    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    
    # Also log to console for development
    logger.info(f"EVENT: {event_type} | USER: {user_id} | {json.dumps(data)}")

# --- Models ---
class EventLog(BaseModel):
    event_type: str = Field(..., description="Event type (e.g., 'auth_success', 'gok_call')")
    user_id: Optional[str] = Field(None, description="User identifier")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    source: Optional[str] = Field("client", description="Event source")

class TokenRequest(BaseModel):
    username: str
    password: str

# --- Enhanced Authentication System ---
security = HTTPBearer()

def get_current_user_role(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    required_role: str = "User"
) -> Dict[str, Any]:
    """
    Enhanced JWT verification with role-based access control
    Roles: MetaGeniusz > Admin > User > Guest
    """
    try:
        token = credentials.credentials
        payload = verify_token(token)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_role = payload.get("role", "Guest")
        user_id = payload.get("sub", "unknown")
        
        # Check role hierarchy
        user_priority = ROLE_PRIORITY.get(user_role, 0)
        required_priority = ROLE_PRIORITY.get(required_role, 0)
        
        if user_priority < required_priority:
            log_telemetry_event("auth_failure", {
                "user_role": user_role,
                "required_role": required_role,
                "reason": "insufficient_privileges"
            }, user_id)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role}, current: {user_role}",
            )
        
        log_telemetry_event("auth_success", {
            "role": user_role,
            "required_role": required_role
        }, user_id)
        
        return {
            "user_id": user_id,
            "role": user_role,
            "token_data": payload
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

# Role-specific dependencies
def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return get_current_user_role(credentials, "User")

def require_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return get_current_user_role(credentials, "Admin")

def require_meta_geniusz(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return get_current_user_role(credentials, "MetaGeniusz")

# --- FastAPI App ---
app = FastAPI(
    title="Meta-Genius Unified Gateway v1.1",
    description="Enhanced central orchestrator for Meta-Genius ecosystem",
    version="1.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Proxy Request Function ---
async def proxy_request(service: str, path: str, method: str = "GET", **kwargs):
    """Enhanced proxy with telemetry"""
    base_url = SERVICES.get(service)
    if not base_url:
        raise HTTPException(404, f"Service {service} not found")
    
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    
    log_telemetry_event("proxy_request", {
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
                response = await client.request(method, url, **kwargs)
            
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(503, f"Service {service} unavailable")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e}")
        raise HTTPException(e.response.status_code, f"Service error: {e}")

# --- Health & System Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def system_health():
    """Enhanced health check with real service status"""
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
                "url": url
            }
    
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "gateway_version": "1.1.0",
        "services": service_status
    }

@app.get("/services", tags=["System"])
async def list_services():
    """List all available services"""
    return {
        "services": SERVICES,
        "total": len(SERVICES),
        "gateway_port": GATEWAY_PORT
    }

# --- Authentication Endpoints ---
@app.post("/auth/token", tags=["Authentication"])
async def login_for_access_token(request: TokenRequest):
    """Generate JWT token - enhanced with proper validation"""
    # Simple auth check (in production: use proper user DB)
    if request.username == "patryk" and request.password == "metageniusz":
        token = create_token({
            "sub": request.username,
            "role": "MetaGeniusz",
            "iat": datetime.now().timestamp()
        }, exp_sec=86400)
        
        log_telemetry_event("auth_login", {
            "username": request.username,
            "role": "MetaGeniusz",
            "success": True
        })
        
        return {
            "access_token": token,
            "token_type": "bearer", 
            "role": "MetaGeniusz",
            "expires_in": 86400,
            "user": request.username
        }
    else:
        log_telemetry_event("auth_login", {
            "username": request.username,
            "success": False,
            "reason": "invalid_credentials"
        })
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/auth/verify", tags=["Authentication"])
async def verify_current_token(user: dict = Depends(require_auth)):
    """Verify current token validity"""
    return {
        "valid": True,
        "user": user.get("user_id"),
        "role": user.get("role"),
        "token_data": user.get("token_data")
    }

# --- Telemetry Endpoints ---
@app.post("/v1/events", status_code=status.HTTP_202_ACCEPTED, tags=["Telemetry"])
async def receive_telemetry_events(events: List[EventLog], user: dict = Depends(require_auth)):
    """Enhanced telemetry endpoint with authentication"""
    for event in events:
        log_telemetry_event(
            event_type=event.event_type,
            data=event.details,
            user_id=event.user_id or user.get("user_id"),
            source=event.source
        )
    
    return {
        "message": f"Processed {len(events)} telemetry events",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/v1/events", tags=["Telemetry"])
async def get_recent_events(limit: int = 50, user: dict = Depends(require_admin)):
    """Get recent telemetry events - admin only"""
    try:
        if not EVENTS_LOG.exists():
            return {"events": []}
        
        events = []
        with open(EVENTS_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                events.append(json.loads(line.strip()))
        
        return {
            "events": events,
            "total": len(events),
            "requested_by": user.get("user_id")
        }
    except Exception as e:
        logger.error(f"Failed to read events: {e}")
        raise HTTPException(status_code=500, detail="Failed to read events")

# --- Protected Admin Endpoints ---
@app.get("/admin/audit", tags=["Admin"])
async def admin_audit_log(user: dict = Depends(require_meta_geniusz)):
    """Enhanced audit endpoint - MetaGeniusz only"""
    log_telemetry_event("admin_audit", {
        "auditor": user.get("user_id"),
        "timestamp": datetime.now().isoformat()
    }, user.get("user_id"))
    
    return {
        "audit_timestamp": datetime.now().isoformat(),
        "auditor": user.get("user_id"),
        "system_status": "operational",
        "services_count": len(SERVICES),
        "events_file_size": EVENTS_LOG.stat().st_size if EVENTS_LOG.exists() else 0,
        "gateway_version": "1.1.0"
    }

@app.get("/god", tags=["Admin"])
async def god_panel_access(user: dict = Depends(require_meta_geniusz)):
    """God panel access - MetaGeniusz only"""
    try:
        god_panel_path = Path("portal/god_v11.html")
        if not god_panel_path.exists():
            god_panel_path = Path("portal/god.html")
        
        with open(god_panel_path, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/html")
    except Exception as e:
        logger.error(f"God panel access failed: {e}")
        raise HTTPException(status_code=500, detail="God panel unavailable")

# --- Proxy Routing to Microservices ---
@app.get("/gok/success", tags=["GOK Formula"])
async def proxy_gok_success(user: dict = Depends(require_auth)):
    """Proxy to GOK Core Service - calculate success"""
    result = await proxy_request("gok_core", "v1/success")
    
    log_telemetry_event("gok_success_call", {
        "user": user.get("user_id"),
        "result": result
    }, user.get("user_id"))
    
    return result

@app.get("/migi/state", tags=["MIGI Core"])
async def proxy_migi_state(user: dict = Depends(require_auth)):
    """Proxy to MIGI Core Service - get AI state"""
    result = await proxy_request("migi_core", "v1/state")
    
    log_telemetry_event("migi_state_call", {
        "user": user.get("user_id")
    }, user.get("user_id"))
    
    return result

@app.get("/hhu/stats", tags=["HHU Service"])
async def proxy_hhu_stats(user: dict = Depends(require_auth)):
    """Proxy to HHU Service - get stats"""
    result = await proxy_request("hhu", "v1/stats")
    
    log_telemetry_event("hhu_stats_call", {
        "user": user.get("user_id")
    }, user.get("user_id"))
    
    return result

# --- Development/Testing ---
if __name__ == "__main__":
    print(f"🚀 Meta-Genius Unified Gateway v1.1 starting on port {GATEWAY_PORT}")
    print("🔐 Protected endpoints: /admin/*, /god")
    print("🔑 Auth endpoint: /auth/token")
    print("📊 Telemetry: /v1/events")
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, reload=True)