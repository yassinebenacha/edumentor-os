"""
config.py
Centralized, validated configuration for EduMentor-OS.
Loaded once at import time; all modules import `settings` from here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All environment-driven settings with sensible defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (Groq) ─────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key for LLM inference")
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq chat model id",
    )
    llm_temperature: float = Field(
        default=0.2,
        description="Sampling temperature (0.2 grading, 0.7 tutoring)",
    )

    # ── Embeddings (HuggingFace — LOCAL, no API key needed) ─────────────────
    # The model is downloaded once to ~/.cache/huggingface (~90 MB).
    # Options:
    #   "sentence-transformers/all-MiniLM-L6-v2"   fast, 384 dims  ✅ default
    #   "BAAI/bge-small-en-v1.5"                   slightly better, 384 dims
    #   "BAAI/bge-large-en-v1.5"                   best quality, 1024 dims
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model (local, free)",
    )
    embedding_dimensions: int = Field(
        default=384,
        description="Output dimensionality (384 for MiniLM, 1024 for bge-large)",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for inference: 'cpu' or 'cuda' (if GPU available)",
    )

    # ── Tavily Web Search ───────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", description="Tavily Search API key")

    # ── LangSmith Observability ─────────────────────────────────────────────
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(default="edumentor-os", description="LangSmith project name")
    langchain_tracing_v2: str = Field(default="false", description="Enable LangSmith tracing")

    # ── Storage paths ───────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("./data"), description="Root data directory")
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="ChromaDB persistence directory",
    )
    knowledge_base_dir: Path = Field(
        default=Path("./data/knowledge_base"),
        description="Source documents for RAG ingestion",
    )
    rubrics_dir: Path = Field(
        default=Path("./data/rubrics"),
        description="Grading rubric JSON/YAML files",
    )

    # ── FastAPI ─────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="Uvicorn bind host")
    api_port: int = Field(default=8000, description="Uvicorn bind port")
    admin_api_key: str = Field(default="changeme", description="Secret key for admin endpoints")

    # ── Code Executor ────────────────────────────────────────────────────────
    code_exec_timeout: int = Field(
        default=10,
        description="Max seconds allowed for student code execution",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton. Re-import-safe."""
    return Settings()


# Module-level convenience alias — use `from config import settings`
settings: Settings = get_settings()
