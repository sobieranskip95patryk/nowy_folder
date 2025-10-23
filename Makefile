# Meta-Genius Master Workspace
all: help

help:
	@echo "🌍 Meta-Genius Digital Empire Commands:"
	@echo "  make init     - Setup Python environment"
	@echo "  make sync     - Sync all repositories" 
	@echo "  make gateway  - Run unified gateway"
	@echo "  make test     - Test all services"
	@echo "  make docker   - Docker compose up"

init:
	python -m venv .venv
	.\.venv\Scripts\activate && pip install -U pip fastapi uvicorn httpx pypdf pytest
	@echo "✅ Environment ready"

sync:
	python sync_repos.py
	@echo "✅ Repositories synchronized"

gateway:
	.\.venv\Scripts\activate && python unified_gateway.py
	@echo "🚀 Gateway starting on port 8800"

test:
	.\.venv\Scripts\activate && python -m pytest tests/ -v
	@echo "🧪 Tests complete"

docker:
	docker compose up --build

clean:
	rmdir /s repos
	rmdir /s .venv

.PHONY: help init sync gateway test docker clean