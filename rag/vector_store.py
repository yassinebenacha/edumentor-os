"""
rag/vector_store.py
ChromaDB vector store manager for EduMentor-OS.

Manages three dedicated collections:
  • knowledge_base   — pedagogical resources (courses, documents)
  • student_profiles — per-student learning gap history (namespaced by student_id)
  • rubric_store     — grading rubrics and evaluation criteria

All collections persist to disk at settings.chroma_persist_dir.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import settings
from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# ── Collection name constants ────────────────────────────────────────────────

COLLECTION_KNOWLEDGE_BASE = "knowledge_base"
COLLECTION_STUDENT_PROFILES = "student_profiles"
COLLECTION_RUBRIC_STORE = "rubric_store"

_ALL_COLLECTIONS = (
    COLLECTION_KNOWLEDGE_BASE,
    COLLECTION_STUDENT_PROFILES,
    COLLECTION_RUBRIC_STORE,
)


class VectorStoreManager:
    """
    Unified interface over three ChromaDB collections.

    Usage
    -----
    >>> vsm = VectorStoreManager()
    >>> vsm.add_documents("knowledge_base", ["text…"], [{"subject": "math"}])
    >>> docs = vsm.similarity_search("knowledge_base", "intégrale de Riemann", k=3)
    """

    def __init__(self) -> None:
        self._embeddings = get_embeddings()
        self._persist_dir = str(settings.chroma_persist_dir)

        # Lazily-initialised Chroma instances, keyed by collection name
        self._stores: dict[str, Chroma] = {}

        # Ensure all collections are created at startup
        for name in _ALL_COLLECTIONS:
            self._get_store(name)

        logger.info(
            "VectorStoreManager ready — persist_dir=%s  collections=%s",
            self._persist_dir,
            _ALL_COLLECTIONS,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_store(self, collection_name: str) -> Chroma:
        """Return (or create) the Chroma store for *collection_name*."""
        if collection_name not in self._stores:
            if collection_name not in _ALL_COLLECTIONS:
                raise ValueError(
                    f"Unknown collection '{collection_name}'. "
                    f"Valid options: {_ALL_COLLECTIONS}"
                )
            self._stores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        return self._stores[collection_name]

    # ── Generic CRUD ─────────────────────────────────────────────────────────

    def add_documents(
        self,
        collection_name: str,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Embed *texts* and upsert them into *collection_name*.

        Parameters
        ----------
        collection_name : str
            One of COLLECTION_* constants.
        texts : list[str]
            Raw text chunks to embed and store.
        metadatas : list[dict], optional
            Per-chunk metadata dicts (must be same length as *texts*).
        ids : list[str], optional
            Stable document IDs. Auto-generated (UUID4) if not provided.

        Returns
        -------
        list[str]
            The IDs of the inserted documents.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        store = self._get_store(collection_name)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        logger.debug(
            "add_documents: collection=%s  n=%d", collection_name, len(texts)
        )
        return ids

    def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[Document]:
        """
        Return the *k* most semantically similar Documents.

        Parameters
        ----------
        collection_name : str
            Target collection.
        query : str
            Natural-language search query.
        k : int
            Number of results to return (default 5).
        filter : dict, optional
            ChromaDB metadata filter, e.g. ``{"subject": "mathematics"}``.

        Returns
        -------
        list[Document]
            LangChain Document objects with *page_content* and *metadata*.
        """
        store = self._get_store(collection_name)
        kwargs: dict[str, Any] = {"k": k}
        if filter:
            kwargs["filter"] = filter

        results = store.similarity_search(query, **kwargs)
        logger.debug(
            "similarity_search: collection=%s  query='%s'  k=%d  hits=%d",
            collection_name,
            query[:60],
            k,
            len(results),
        )
        return results

    # ── Student Profile helpers ──────────────────────────────────────────────

    def update_student_profile(
        self,
        student_id: str,
        lacunes_json: dict[str, Any],
    ) -> str:
        """
        Append a new learning-gap snapshot to the student's profile collection.

        Each call stores a new Document so the full history is preserved.
        The document text is a human-readable summary; the raw JSON is kept
        in the metadata for exact retrieval.

        Parameters
        ----------
        student_id : str
            Unique student identifier (used as namespace in metadata).
        lacunes_json : dict
            Structured gap analysis, e.g.::

                {
                    "session": "2024-01-15",
                    "subject": "algebra",
                    "gaps": ["factorisation", "polynômes du 2nd degré"],
                    "score": 11.5,
                }

        Returns
        -------
        str
            The ID of the newly inserted document.
        """
        subject = lacunes_json.get("subject", "general")
        gaps = lacunes_json.get("gaps", [])
        score = lacunes_json.get("score", "N/A")
        session = lacunes_json.get("session", "unknown")

        summary_text = (
            f"Étudiant {student_id} | Session {session} | Matière: {subject} | "
            f"Score: {score} | Lacunes: {', '.join(gaps) if gaps else 'aucune'}"
        )

        metadata = {
            "student_id": student_id,
            "subject": subject,
            "session": session,
            "score": str(score),
            "raw_json": json.dumps(lacunes_json, ensure_ascii=False),
        }

        doc_id = str(uuid.uuid4())
        self.add_documents(
            collection_name=COLLECTION_STUDENT_PROFILES,
            texts=[summary_text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.info(
            "update_student_profile: student_id=%s  session=%s  gaps=%s",
            student_id,
            session,
            gaps,
        )
        return doc_id

    def get_student_profile(self, student_id: str) -> dict[str, Any]:
        """
        Retrieve the complete history of learning-gap snapshots for *student_id*.

        Uses a metadata filter so only documents belonging to this student
        are returned. All snapshots are decoded from their stored JSON.

        Parameters
        ----------
        student_id : str
            Target student.

        Returns
        -------
        dict with keys:
            - ``student_id``: str
            - ``total_sessions``: int
            - ``sessions``: list[dict]   — each entry is the original lacunes_json
            - ``all_gaps``: list[str]    — deduplicated union of all recorded gaps
        """
        store = self._get_store(COLLECTION_STUDENT_PROFILES)

        # Chroma get() with a where filter is the cheapest way to fetch by metadata
        raw = store.get(where={"student_id": student_id}, include=["metadatas", "documents"])

        sessions: list[dict[str, Any]] = []
        all_gaps: set[str] = set()

        for meta in raw.get("metadatas", []):
            raw_json_str = meta.get("raw_json", "{}")
            try:
                entry = json.loads(raw_json_str)
            except json.JSONDecodeError:
                entry = {"raw": raw_json_str}
            sessions.append(entry)
            for gap in entry.get("gaps", []):
                all_gaps.add(gap)

        profile = {
            "student_id": student_id,
            "total_sessions": len(sessions),
            "sessions": sessions,
            "all_gaps": sorted(all_gaps),
        }
        logger.debug(
            "get_student_profile: student_id=%s  sessions=%d",
            student_id,
            len(sessions),
        )
        return profile

    # ── Rubric helpers ───────────────────────────────────────────────────────

    def add_rubric(
        self,
        subject: str,
        rubric_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a grading rubric in the rubric_store collection.

        Parameters
        ----------
        subject : str
            Academic subject (e.g., "mathematics", "python_programming").
        rubric_text : str
            Full rubric description (free-text or structured markdown).
        metadata : dict, optional
            Additional metadata, e.g. ``{"assignment": "DM1", "max_score": 20}``.

        Returns
        -------
        str
            Document ID of the stored rubric.
        """
        base_meta: dict[str, Any] = {"subject": subject}
        if metadata:
            base_meta.update(metadata)

        doc_id = str(uuid.uuid4())
        self.add_documents(
            collection_name=COLLECTION_RUBRIC_STORE,
            texts=[rubric_text],
            metadatas=[base_meta],
            ids=[doc_id],
        )
        logger.info("add_rubric: subject=%s  id=%s", subject, doc_id)
        return doc_id

    def search_rubric(
        self,
        subject: str,
        query: str,
        k: int = 3,
    ) -> list[Document]:
        """
        Find grading rubrics relevant to *query*, filtered by *subject*.

        Parameters
        ----------
        subject : str
            Subject to filter on (exact metadata match).
        query : str
            Description of the assignment or criteria to match.
        k : int
            Maximum number of rubric excerpts to return (default 3).

        Returns
        -------
        list[Document]
            Matching rubric Documents sorted by similarity.
        """
        return self.similarity_search(
            collection_name=COLLECTION_RUBRIC_STORE,
            query=query,
            k=k,
            filter={"subject": subject},
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_vsm_instance: VectorStoreManager | None = None


def get_vector_store() -> VectorStoreManager:
    """
    Return the application-wide VectorStoreManager singleton.

    Lazy-initialised on first call so import of this module is fast
    and does not touch disk or the network.
    """
    global _vsm_instance  # noqa: PLW0603
    if _vsm_instance is None:
        _vsm_instance = VectorStoreManager()
    return _vsm_instance
