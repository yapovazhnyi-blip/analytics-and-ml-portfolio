# Crucible — developer Makefile
# Run `make help` to list all targets.

PYTHON      := python3
PIP         := pip
PYTEST      := pytest
RUFF        := ruff
UVICORN     := uvicorn
ALEMBIC     := alembic
DOCKER      := docker
COMPOSE     := docker compose

BACKEND_DIR := backend
FRONTEND_DIR := frontend

.PHONY: help install dev test test-cov lint format \
        migrate migrate-new db-reset \
        docker-build docker-up docker-down docker-logs \
        frontend clean

# ── Help ──────────────────────────────────────────────────────────────────────
help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────────────────────────
install:       ## Install all backend dependencies
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt

install-dev:   ## Install backend + dev tools (ruff, pytest-cov, moto)
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt \
	  ruff pytest-cov coverage boto3 moto

# ── Local development server ──────────────────────────────────────────────────
dev:           ## Start backend dev server with auto-reload
	cd $(BACKEND_DIR) && $(UVICORN) main:app \
	  --host 0.0.0.0 \
	  --port 8000 \
	  --reload \
	  --reload-dir .

# ── Tests ─────────────────────────────────────────────────────────────────────
test:          ## Run full test suite
	cd $(BACKEND_DIR) && $(PYTEST) tests/ -q

test-cov:      ## Run tests with coverage report
	cd $(BACKEND_DIR) && $(PYTEST) tests/ \
	  --cov=. \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  --cov-fail-under=75 \
	  -q

test-fast:     ## Run tests excluding slow integration tests
	cd $(BACKEND_DIR) && $(PYTEST) tests/ -q -m "not slow"

test-integration: ## Run real-server integration tests (spawns live uvicorn)
	cd $(BACKEND_DIR) && $(PYTEST) tests/test_integration_real_server.py -m integration -v

test-watch:    ## Re-run tests on file change (requires pytest-watch)
	cd $(BACKEND_DIR) && ptw tests/ -- -q

# ── Lint & format ─────────────────────────────────────────────────────────────
lint:          ## Run ruff linter (errors only, no auto-fix)
	$(RUFF) check $(BACKEND_DIR)/

lint-fix:      ## Run ruff linter with auto-fix
	$(RUFF) check --fix $(BACKEND_DIR)/

format:        ## Auto-format with ruff format
	$(RUFF) format $(BACKEND_DIR)/

format-check:  ## Check formatting without modifying files (for CI)
	$(RUFF) format --check $(BACKEND_DIR)/

# ── Database migrations ────────────────────────────────────────────────────────
migrate:       ## Apply all pending Alembic migrations
	cd $(BACKEND_DIR) && $(ALEMBIC) upgrade head

migrate-new:   ## Create a new migration (usage: make migrate-new MSG="add_column_x")
	cd $(BACKEND_DIR) && $(ALEMBIC) revision --autogenerate -m "$(MSG)"

migrate-down:  ## Rollback the last migration
	cd $(BACKEND_DIR) && $(ALEMBIC) downgrade -1

migrate-status: ## Show current migration state
	cd $(BACKEND_DIR) && $(ALEMBIC) current

db-reset:      ## Drop and recreate the database from migrations (DESTRUCTIVE)
	@echo "⚠️  This will DELETE all data. Press Ctrl-C to cancel."
	@sleep 3
	cd $(BACKEND_DIR) && \
	  rm -f data/crucible.db && \
	  $(ALEMBIC) upgrade head && \
	  echo "✓ Database reset complete"

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:  ## Build the backend Docker image
	$(DOCKER) build -t crucible-backend:local $(BACKEND_DIR)/

docker-up:     ## Start the full stack with docker compose
	$(COMPOSE) up --build -d

docker-down:   ## Stop and remove containers (keeps volumes)
	$(COMPOSE) down

docker-nuke:   ## Stop containers AND remove volumes (DESTRUCTIVE)
	$(COMPOSE) down -v

docker-logs:   ## Tail all container logs
	$(COMPOSE) logs -f

docker-shell:  ## Open a shell in the running backend container
	$(COMPOSE) exec backend bash

docker-test:   ## Run tests inside a Docker container
	$(DOCKER) run --rm \
	  -e DATABASE_URL="sqlite+aiosqlite:///:memory:" \
	  -e DISABLE_AUTH=true \
	  crucible-backend:local \
	  pytest tests/ -q

# ── Frontend ──────────────────────────────────────────────────────────────────
frontend:      ## Start the frontend Vite dev server
	cd $(FRONTEND_DIR) && npm run dev

frontend-build: ## Build the frontend for production
	cd $(FRONTEND_DIR) && npm run build

frontend-install: ## Install frontend npm dependencies
	cd $(FRONTEND_DIR) && npm install

# ── Maintenance ───────────────────────────────────────────────────────────────
clean:         ## Remove build artifacts and cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -f $(BACKEND_DIR)/coverage.xml
	rm -rf $(BACKEND_DIR)/htmlcov/
	@echo "✓ Clean complete"
