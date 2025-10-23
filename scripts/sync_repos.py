#!/usr/bin/env python3
import subprocess, pathlib, sys

root = pathlib.Path(__file__).resolve().parents[1]
repos_dir = root / "repos"
repos_dir.mkdir(parents=True, exist_ok=True)

def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

urls = [l.strip() for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
for url in urls:
    name = url.split("/")[-1].removesuffix(".git")
    dest = repos_dir / name
    if dest.exists():
        print(f"Updating {name}...")
        run(["git", "pull", "--rebase"], cwd=dest)
    else:
        print(f"Cloning {name}...")
        run(["git", "clone", "--depth", "1", url, str(dest)])
print("OK - All repos synchronized")