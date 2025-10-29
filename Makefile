# Meta-Genius Master Workspace
all: help

help:
	@echo "🌍 Meta-Genius Digital Empire Commands:"
	@echo "  make init     - Setup Python environment"
	@echo "  make sync     - Sync all repositories" 
	@echo "  make gateway  - Run unified gateway"
	@echo "  make test     - Test all services"
	@echo "  make docker   - Docker compose up"
	@echo "  make dev      - Development stack (build + up)"
	@echo "  make prod     - Production stack" 
	@echo "  make down     - Stop all containers"
	@echo "  make logs     - Show container logs"
	@echo "  make check    - Health check & validation"

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

dev:
	cp .env.example .env
	docker compose up --build

prod:
	docker compose -f docker-compose.prod.yml up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

check:
	python -c "import yaml; print('✅ JSK Config OK')" && \
	python -c "import core.jsk.config; print('✅ Core Imports OK')" && \
	echo "✅ System validation complete"

clean:
	rmdir /s repos
	rmdir /s .venv

.PHONY: help init sync gateway test docker clean