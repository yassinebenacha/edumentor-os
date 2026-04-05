# Makefile — EduMentor OS
# Compatible with GNU Make (Git Bash / WSL / Linux / macOS).
# On Windows PowerShell, use: .\make.ps1 <target>

PYTHON   = python
PIP      = pip
UVICORN  = uvicorn
PYTEST   = pytest
ENV      = edumentor-os

.PHONY: help install install-dev demo test test-unit test-fast api ingest lint clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  EduMentor OS — Available make targets"
	@echo "  ══════════════════════════════════════"
	@echo "  make install      Install all production dependencies"
	@echo "  make install-dev  Install deps + dev tools (pytest, rich)"
	@echo "  make demo         Run the terminal demo (demo/run_demo.py)"
	@echo "  make test         Run the full pytest test suite"
	@echo "  make test-unit    Run only unit tests (no API calls)"
	@echo "  make test-fast    Run smoke tests only (fastest)"
	@echo "  make api          Start the FastAPI server (hot-reload)"
	@echo "  make ingest       Ingest data/ files into ChromaDB"
	@echo "  make ingest-dry   Dry-run ingestion (no writes)"
	@echo "  make lint         Run ruff linter on the project"
	@echo "  make clean        Remove __pycache__, .pytest_cache, chroma_db"
	@echo ""

# ── Installation ──────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install ruff black

# ── Demo ─────────────────────────────────────────────────────────────────────
demo:
	$(PYTHON) demo/run_demo.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short --asyncio-mode=auto

test-unit:
	$(PYTEST) tests/ -v --tb=short --asyncio-mode=auto \
		-k "not integration"

test-fast:
	$(PYTEST) tests/test_system.py::TestSmoke \
		tests/test_system.py::TestCodeExecutor \
		-v --tb=short --asyncio-mode=auto

test-coverage:
	$(PYTEST) tests/ --cov=. --cov-report=term-missing --asyncio-mode=auto

# ── API ───────────────────────────────────────────────────────────────────────
api:
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

api-prod:
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Data ingestion ────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) data/ingest_data.py

ingest-dry:
	$(PYTHON) data/ingest_data.py --dry-run

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check . --select E,W,F,I --ignore E501

format:
	black . --line-length 100

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean done."

clean-db:
	rm -rf data/chroma_db/
	@echo "ChromaDB cleared."
