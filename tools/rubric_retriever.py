"""
tools/rubric_retriever.py
LangChain tool for fetching grading rubrics — file-based + vector-store search.

Two retrieval modes:
  1. Direct file lookup  : reads a JSON/YAML rubric from data/rubrics/ by filename.
  2. Semantic search     : queries the ChromaDB rubric_store for the closest
                           rubric matching a subject + assignment description.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


# ── Input schema ──────────────────────────────────────────────────────────────

class RubricRetrieverInput(BaseModel):
    query: str = Field(
        description=(
            "Nom du fichier barème (ex: 'programmation_python.json') OU "
            "description libre de la matière/exercice pour une recherche sémantique "
            "(ex: 'barème algorithmique récursivité Python')."
        )
    )
    subject: str = Field(
        default="",
        description="Filtre optionnel par matière (ex: 'mathematics', 'python').",
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

class RubricRetrieverTool(BaseTool):
    """
    LangChain tool to retrieve a grading rubric.

    Priority:
      1. If *query* matches a filename in data/rubrics/, load it directly.
      2. Otherwise, perform a semantic search in the ChromaDB rubric_store.
      3. If nothing is found, return an empty rubric template.
    """

    name: str = "rubric_retriever"
    description: str = (
        "Récupère un barème de correction. "
        "Input: nom du fichier barème (ex: 'algo_python.json') OU description "
        "de la matière/exercice. Retourne le barème en texte structuré."
    )
    args_schema: Type[BaseModel] = RubricRetrieverInput

    def _run(self, query: str, subject: str = "") -> str:  # noqa: C901
        """
        Retrieve the best-matching rubric for *query*.

        Returns
        -------
        str
            JSON string with keys: ``found``, ``source``, ``rubric_text``,
            ``subject``, ``metadata``.
        """
        rubrics_dir = settings.rubrics_dir

        # ── 1. Try direct file lookup ─────────────────────────────────────
        if query.endswith((".json", ".yaml", ".yml", ".txt", ".md")):
            file_result = _load_rubric_file(rubrics_dir, query)
            if file_result:
                logger.info("rubric_retriever: file hit — '%s'", query)
                return json.dumps(file_result, ensure_ascii=False, indent=2)

        # ── 2. Fuzzy filename search (partial match) ──────────────────────
        if rubrics_dir.exists():
            for rubric_file in rubrics_dir.iterdir():
                if query.lower() in rubric_file.name.lower():
                    file_result = _load_rubric_file(rubrics_dir, rubric_file.name)
                    if file_result:
                        logger.info(
                            "rubric_retriever: fuzzy file hit — '%s'", rubric_file.name
                        )
                        return json.dumps(file_result, ensure_ascii=False, indent=2)

        # ── 3. Semantic search in ChromaDB ────────────────────────────────
        try:
            from rag.vector_store import get_vector_store  # lazy import

            vsm = get_vector_store()
            effective_subject = subject or _infer_subject(query)
            docs = vsm.search_rubric(
                subject=effective_subject,
                query=query,
                k=2,
            )
            if docs:
                combined = "\n\n---\n\n".join(
                    f"[Source: {d.metadata.get('assignment', 'unknown')}]\n{d.page_content}"
                    for d in docs
                )
                result = {
                    "found": True,
                    "source": "vector_store",
                    "rubric_text": combined,
                    "subject": effective_subject,
                    "metadata": [d.metadata for d in docs],
                }
                logger.info(
                    "rubric_retriever: vector hit — subject='%s' docs=%d",
                    effective_subject,
                    len(docs),
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rubric_retriever: vector search failed — %s", exc)

        # ── 4. Fallback: empty template ───────────────────────────────────
        logger.warning("rubric_retriever: no rubric found for query='%s'", query)
        return json.dumps(
            {
                "found": False,
                "source": "none",
                "rubric_text": (
                    "Aucun barème trouvé. Évalue la copie selon les critères pédagogiques "
                    "standards : clarté, exactitude, démarche et présentation."
                ),
                "subject": subject,
                "metadata": [],
            },
            ensure_ascii=False,
            indent=2,
        )

    async def _arun(self, query: str, subject: str = "") -> str:
        return self._run(query=query, subject=subject)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_rubric_file(
    rubrics_dir: Path, filename: str
) -> dict[str, Any] | None:
    """Load a rubric from *rubrics_dir / filename*. Returns None if not found."""
    path = rubrics_dir / filename
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
        # Try to parse as JSON for richer metadata
        if filename.endswith(".json"):
            data = json.loads(text)
            return {
                "found": True,
                "source": f"file:{filename}",
                "rubric_text": json.dumps(data, ensure_ascii=False, indent=2),
                "subject": data.get("subject", ""),
                "metadata": data,
            }
        # Plain text / YAML / markdown — return as-is
        return {
            "found": True,
            "source": f"file:{filename}",
            "rubric_text": text,
            "subject": "",
            "metadata": {"filename": filename},
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("rubric_retriever: error reading '%s' — %s", filename, exc)
        return None


def _infer_subject(query: str) -> str:
    """Very lightweight heuristic to derive a subject tag from free text."""
    query_lower = query.lower()
    if any(k in query_lower for k in ["python", "code", "programm", "algo"]):
        return "python_programming"
    if any(k in query_lower for k in ["math", "calcul", "algèbre", "intégr"]):
        return "mathematics"
    if any(k in query_lower for k in ["physi", "mécani", "électr"]):
        return "physics"
    if any(k in query_lower for k in ["stat", "probab", "donnée"]):
        return "statistics"
    return "general"
