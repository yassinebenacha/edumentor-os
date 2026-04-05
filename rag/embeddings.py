"""
rag/embeddings.py
Centralized embedding model factory for EduMentor-OS.

Uses HuggingFace sentence-transformers running LOCALLY.
  • No API key required
  • No internet needed after the first download (~90 MB)
  • Fully free and offline-capable

Default model : all-MiniLM-L6-v2  (384 dims, very fast, great quality)
Better quality : BAAI/bge-small-en-v1.5  (384 dims, slightly slower)
Best quality   : BAAI/bge-large-en-v1.5  (1024 dims, needs more RAM)
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the singleton HuggingFaceEmbeddings instance.

    The model is downloaded once to ~/.cache/huggingface on first run,
    then loaded from disk on every subsequent run (no internet required).

    Cached with lru_cache so the model stays in memory across calls.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
