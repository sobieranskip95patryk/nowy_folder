#!/usr/bin/env python3
import pathlib, json, re, hashlib
from collections import Counter
root = pathlib.Path(__file__).resolve().parents[1]
repos = (root/"repos").glob("*")
out = root/"data"; out.mkdir(parents=True, exist_ok=True)
inventory = []

def sha(path):
    try: return hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    except: return None

def grep_routes(text):
    pats = [
        r"@app\.get\(['\"]([^'\"]+)",
        r"@app\.post\(['\"]([^'\"]+)",
        r"router\.get\(['\"]([^'\"]+)",
        r"router\.post\(['\"]([^'\"]+)",
        r"app\.get\(['\"]([^'\"]+)",   # express
        r"app\.post\(['\"]([^'\"]+)",
    ]
    found = []
    for p in pats:
        found += re.findall(p, text)
    return sorted(set(found))

for repo in repos:
    item = {"name": repo.name, "path": str(repo), "hash": None,
            "langs": {}, "has": {}, "routes": [], "ports": [], "license": None}
    langs = Counter()
    for p in repo.rglob("*"):
        if p.is_file():
            suf = p.suffix.lower()
            if suf in [".py",".ts",".js",".html",".css",".md",".json",".yml",".yaml",".sol",".ipynb",".tsx"]:
                langs[suf] += 1
            if p.name.lower().startswith("license"):
                item["license"] = "present"
            if p.name == "requirements.txt":
                item["has"]["python"] = True
            if p.name == "package.json":
                item["has"]["node"] = True
            if p.name.lower() == "dockerfile":
                item["has"]["docker"] = True
            if suf in [".py",".ts",".js"]:
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    routes = grep_routes(txt)
                    if routes:
                        item["routes"] += [r for r in routes if r.startswith("/")]
                    ports = re.findall(r"--port\s+(\d{3,5})", txt) + re.findall(r"PORT\s*=\s*(\d{3,5})", txt)
                    if ports:
                        item["ports"] += ports
                except: pass
    item["langs"] = dict(langs)
    item["routes"] = sorted(set(item["routes"]))
    item["ports"] = sorted(set(item["ports"]))
    # repo-level hash (best effort)
    readme = next((x for x in repo.iterdir() if x.name.lower().startswith("readme")), None)
    if readme: item["hash"] = sha(readme)
    inventory.append(item)

(out/"inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
print("inventory ->", out/"inventory.json")