#!/usr/bin/env python3
import json, pathlib
from datetime import datetime

root = pathlib.Path(__file__).resolve().parents[1]
inv = json.loads((root/"data/inventory.json").read_text())
md = [f"# MetaGeniusz — Raport Ekosystemu ({datetime.now().isoformat(timespec='seconds')})\n\n"]

def bullet(k, v): return f"- **{k}**: {v}\n"

md.append("## Podsumowanie\n\n")
total_repos = len(inv)
python_repos = len([x for x in inv if "python" in x.get("has", {})])
node_repos = len([x for x in inv if "node" in x.get("has", {})])
docker_repos = len([x for x in inv if "docker" in x.get("has", {})])
with_routes = len([x for x in inv if x.get("routes")])

md.append(f"- **Całkowita liczba repozytoriów**: {total_repos}\n")
md.append(f"- **Repozytoria Python**: {python_repos}\n")
md.append(f"- **Repozytoria Node.js**: {node_repos}\n")
md.append(f"- **Z Dockerfile**: {docker_repos}\n")
md.append(f"- **Z wykrytymi endpointami**: {with_routes}\n\n")

md.append("## Szczegóły repozytoriów\n\n")

for it in sorted(inv, key=lambda x: x["name"].lower()):
    md.append(f"### {it['name']}\n\n")
    langs = ", ".join([f"{k}:{v}" for k,v in it.get("langs",{}).items()]) or "brak danych"
    routes = ", ".join(it.get("routes",[])[:12]) or "brak wykrytych"
    if len(it.get("routes",[])) > 12:
        routes += f" (+{len(it.get('routes',[])) - 12} więcej)"
    ports = ", ".join(it.get("ports",[])[:6]) or "nie wykryto"
    
    md.append(bullet("Ścieżka", it["path"]))
    md.append(bullet("Języki/pliki", langs))
    md.append(bullet("Licencja", it.get("license") or "brak pliku"))
    md.append(bullet("Python", "✓" if "python" in it.get("has", {}) else "✗"))
    md.append(bullet("Node.js", "✓" if "node" in it.get("has", {}) else "✗"))
    md.append(bullet("Docker", "✓" if "docker" in it.get("has", {}) else "✗"))
    md.append(bullet("Wykryte endpointy", routes))
    md.append(bullet("Porty", ports))
    md.append("\n")

out = root/"data"/"REPORT.md"
out.write_text("".join(md), encoding="utf-8")
print("report ->", out)