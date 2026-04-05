<#
.SYNOPSIS
    EduMentor OS — PowerShell equivalent of the Makefile (Windows-native).

.USAGE
    .\make.ps1 install
    .\make.ps1 demo
    .\make.ps1 test
    .\make.ps1 api
    .\make.ps1 ingest
#>

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Header($msg) {
    Write-Host ""
    Write-Host "  ══ $msg ══" -ForegroundColor Cyan
    Write-Host ""
}

# ── Targets ───────────────────────────────────────────────────────────────────
switch ($Target) {

    "help" {
        Write-Host ""
        Write-Host "  EduMentor OS — make.ps1 targets" -ForegroundColor Yellow
        Write-Host "  ════════════════════════════════"
        @(
            "install     Install all production dependencies",
            "demo        Run the terminal demo",
            "test        Run the full pytest test suite",
            "test-unit   Run only unit tests (no API calls)",
            "test-fast   Run smoke tests only",
            "api         Start the FastAPI server (hot-reload)",
            "ingest      Ingest data/ files into ChromaDB",
            "ingest-dry  Dry-run ingestion (no writes)",
            "clean       Remove __pycache__ and .pytest_cache",
            "clean-db    Remove data/chroma_db/"
        ) | ForEach-Object { Write-Host "  .\make.ps1 $_" }
        Write-Host ""
    }

    "install" {
        Write-Header "Installing dependencies"
        pip install -r requirements.txt
    }

    "install-dev" {
        Write-Header "Installing dependencies + dev tools"
        pip install -r requirements.txt
        pip install ruff black
    }

    "demo" {
        Write-Header "Running EduMentor OS Demo"
        python demo/run_demo.py
    }

    "test" {
        Write-Header "Running full test suite"
        pytest tests/ -v --tb=short
    }

    "test-unit" {
        Write-Header "Running unit tests (no integration)"
        pytest tests/ -v --tb=short -k "not integration"
    }

    "test-fast" {
        Write-Header "Running fast smoke tests"
        pytest tests/test_system.py::TestSmoke tests/test_system.py::TestCodeExecutor -v --tb=short
    }

    "api" {
        Write-Header "Starting FastAPI server on :8000"
        uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    }

    "api-prod" {
        Write-Header "Starting FastAPI (production)"
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
    }

    "ingest" {
        Write-Header "Ingesting data/ into ChromaDB"
        python data/ingest_data.py
    }

    "ingest-dry" {
        Write-Header "Dry-run ingestion"
        python data/ingest_data.py --dry-run
    }

    "clean" {
        Write-Header "Cleaning cache files"
        Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host "  Clean done." -ForegroundColor Green
    }

    "clean-db" {
        Write-Header "Clearing ChromaDB"
        if (Test-Path "data/chroma_db") {
            Remove-Item -Recurse -Force "data/chroma_db"
            Write-Host "  ChromaDB cleared." -ForegroundColor Green
        } else {
            Write-Host "  data/chroma_db does not exist." -ForegroundColor Yellow
        }
    }

    default {
        Write-Host "Unknown target: '$Target'. Run .\make.ps1 help" -ForegroundColor Red
        exit 1
    }
}
