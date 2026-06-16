# ==============================================================================
# FlowEdit — Makefile
# Convenience commands for development.  Run: make <target>
# ==============================================================================

.PHONY: setup run dev test clean purge lint help

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:           ## Full one-time environment setup (installs all deps + weights)
	@chmod +x setup.sh && ./setup.sh

weights:         ## Download model weights only (skip pip installs)
	@python3 download_weights.py

# ── Running ────────────────────────────────────────────────────────────────────
run:             ## Start backend + frontend (recommended)
	@chmod +x run.sh && ./run.sh

api:             ## Start backend only (no frontend server)
	@uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level info

frontend:        ## Serve frontend only on port 5500
	@cd frontend && python3 -m http.server 5500

# ── Testing ───────────────────────────────────────────────────────────────────
test:            ## Run full pytest suite
	@pytest tests/ -v --tb=short

test-fast:       ## Run tests excluding slow GPU tests
	@pytest tests/ -v --tb=short -m "not slow"

test-api:        ## API route tests only
	@pytest tests/test_api.py -v

test-pipeline:   ## Pipeline stage tests only
	@pytest tests/test_extractor.py tests/test_temporal.py -v

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:           ## Clear processed frames and output videos (keeps uploads)
	@echo "Clearing processed frames and outputs…"
	@rm -rf storage/frames/* storage/processed/* storage/outputs/*
	@echo "Done."

purge:           ## Clear ALL storage including uploads (irreversible)
	@read -p "Delete all uploads, frames, and outputs? [y/N] " ans; \
	 [[ $$ans == [yY] ]] && rm -rf storage/uploads/* storage/frames/* \
	 storage/processed/* storage/outputs/* && echo "Purged." || echo "Cancelled."

# ── Code quality ──────────────────────────────────────────────────────────────
lint:            ## Run ruff linter + mypy type checker
	@ruff check app/ main.py
	@mypy app/ main.py --ignore-missing-imports

format:          ## Auto-format with ruff
	@ruff format app/ main.py tests/

# ── Help ──────────────────────────────────────────────────────────────────────
help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
