#!/usr/bin/env python3
import json, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
inv = json.loads((root/"data/inventory.json").read_text())
compose = {
  "version": "3.9",
  "services": {
    "gateway": {
      "build": "./gateway",
      "ports": ["8800:8800"],
      "environment": {}
    }
  }
}
svc_env = compose["services"]["gateway"]["environment"]

port_cursor = 8010
for it in inv:
    name = it["name"]
    svc_name = name.replace("-","_").lower()
    if "python" in it.get("has", {}):
        compose["services"][svc_name] = {
            "build": f"./repos/{name}",
            "command": f"uvicorn app.main:app --host 0.0.0.0 --port {port_cursor}",
            "ports": [f"{port_cursor}:{port_cursor}"]
        }
        svc_env[f"SVC_{svc_name.upper()}"] = f"http://{svc_name}:{port_cursor}"
        port_cursor += 1
    elif "node" in it.get("has", {}):
        compose["services"][svc_name] = {
            "build": f"./repos/{name}",
            "command": "npm run start",
            "ports": [f"{port_cursor}:3000"]
        }
        svc_env[f"SVC_{svc_name.upper()}"] = f"http://{svc_name}:3000"
        port_cursor += 1

out = root/"docker-compose.generated.yml"
out.write_text(json.dumps(compose, indent=2))
print("compose ->", out)