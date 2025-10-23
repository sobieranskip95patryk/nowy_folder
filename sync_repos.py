#!/usr/bin/env python3
"""
Repository Synchronization Script
Klonuje i synchronizuje wszystkie repozytoria z workspace.json
"""

import json
import subprocess
import sys
import pathlib
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_command(cmd, cwd=None):
    """Execute command and return result"""
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        log(f"❌ Command failed: {e}")
        return False

def sync_repo(repo_info, repos_dir):
    """Clone or update a single repository"""
    name = repo_info["name"]
    url = repo_info["url"]
    repo_path = repos_dir / name
    
    if repo_path.exists():
        log(f"📥 Updating {name}...")
        success = run_command(["git", "pull", "--rebase"], cwd=repo_path)
        if success:
            log(f"✅ {name} updated")
        else:
            log(f"⚠️ Failed to update {name}")
        return success
    else:
        log(f"📦 Cloning {name}...")
        repos_dir.mkdir(parents=True, exist_ok=True)
        success = run_command(["git", "clone", "--depth", "1", url, str(repo_path)])
        if success:
            log(f"✅ {name} cloned")
        else:
            log(f"❌ Failed to clone {name}")
        return success

def main():
    root = pathlib.Path(__file__).resolve().parent
    workspace_file = root / "workspace.json"
    
    if not workspace_file.exists():
        log("❌ workspace.json not found!")
        return 1
    
    # Load workspace configuration
    with open(workspace_file, 'r', encoding='utf-8') as f:
        workspace = json.load(f)
    
    repos_dir = root / "repos"
    log(f"🚀 Starting sync for {len(workspace['repos'])} repositories...")
    
    # Sync all repositories
    failed_repos = []
    for repo in workspace["repos"]:
        if not sync_repo(repo, repos_dir):
            failed_repos.append(repo["name"])
    
    # Summary
    total = len(workspace["repos"])
    success = total - len(failed_repos)
    
    log(f"📊 Sync complete: {success}/{total} repositories updated")
    
    if failed_repos:
        log("⚠️ Failed repositories:")
        for repo in failed_repos:
            log(f"  - {repo}")
        return 1
    
    log("🎉 All repositories synchronized successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())