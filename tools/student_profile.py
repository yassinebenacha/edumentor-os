"""
tools/student_profile.py
LangChain tool for reading and updating persistent student learning profiles.

Wraps the ChromaDB student_profiles collection via VectorStoreManager.
Used by TutorAgent to personalise content and AnalystAgent for reports.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Input schemas ─────────────────────────────────────────────────────────────

class StudentProfileGetInput(BaseModel):
    student_id: str = Field(description="Identifiant unique de l'étudiant")


class StudentProfileUpdateInput(BaseModel):
    student_id: str = Field(description="Identifiant unique de l'étudiant")
    lacunes_json: str = Field(
        description=(
            "JSON stringifié du profil de lacunes à enregistrer. "
            "Exemple: '{\"subject\": \"python\", \"gaps\": [\"récursivité\"], \"score\": 12}'"
        )
    )


# ── GET Tool ──────────────────────────────────────────────────────────────────

class StudentProfileTool(BaseTool):
    """
    LangChain tool to fetch a student's complete learning profile from ChromaDB.

    Returns all recorded sessions (gaps, scores, subjects) so the TutorAgent
    can adapt its explanations to the student's actual knowledge level.
    """

    name: str = "student_profile"
    description: str = (
        "Récupère le profil d'apprentissage complet d'un étudiant (lacunes, scores, "
        "historique de sessions). Input: student_id (str). "
        "Retourne un JSON avec toutes les lacunes identifiées et l'historique."
    )
    args_schema: Type[BaseModel] = StudentProfileGetInput

    def _run(self, student_id: str) -> str:
        """
        Fetch the full profile for *student_id* from the vector store.

        Returns
        -------
        str
            JSON-encoded profile dict with keys:
            student_id, total_sessions, sessions[], all_gaps[],
            subject_summary{}, average_score.
        """
        try:
            from rag.vector_store import get_vector_store

            vsm = get_vector_store()
            profile = vsm.get_student_profile(student_id)

            # Enrich with computed summary
            profile["subject_summary"] = _build_subject_summary(profile["sessions"])
            profile["average_score"] = _compute_average_score(profile["sessions"])
            profile["mastery_level"] = _infer_mastery(profile["average_score"])

            logger.info(
                "student_profile get: student_id='%s'  sessions=%d  gaps=%d",
                student_id,
                profile["total_sessions"],
                len(profile["all_gaps"]),
            )
            return json.dumps(profile, ensure_ascii=False, indent=2)

        except Exception as exc:  # noqa: BLE001
            logger.warning("student_profile get error: %s", exc)
            # Return an empty profile so the agent can still function
            return json.dumps(
                {
                    "student_id": student_id,
                    "total_sessions": 0,
                    "sessions": [],
                    "all_gaps": [],
                    "subject_summary": {},
                    "average_score": None,
                    "mastery_level": "inconnu",
                    "note": f"Profil non trouvé ou erreur : {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )

    async def _arun(self, student_id: str) -> str:
        return self._run(student_id)


# ── UPDATE Tool ───────────────────────────────────────────────────────────────

class StudentProfileUpdateTool(BaseTool):
    """
    LangChain tool to update a student's learning profile in ChromaDB.
    Called by GraderAgent and AnalystAgent after each grading session.
    """

    name: str = "student_profile_update"
    description: str = (
        "Met à jour le profil d'apprentissage d'un étudiant avec les lacunes "
        "détectées lors d'une session. "
        "Inputs: student_id (str), lacunes_json (str JSON)."
    )
    args_schema: Type[BaseModel] = StudentProfileUpdateInput

    def _run(self, student_id: str, lacunes_json: str) -> str:
        """
        Persist a new gap-analysis snapshot for *student_id*.

        Returns
        -------
        str
            JSON with ``success`` and the new document ``doc_id``.
        """
        try:
            lacunes = json.loads(lacunes_json)
        except json.JSONDecodeError as exc:
            return json.dumps(
                {"success": False, "error": f"JSON invalide : {exc}"},
                ensure_ascii=False,
            )

        try:
            from rag.vector_store import get_vector_store

            vsm = get_vector_store()
            doc_id = vsm.update_student_profile(student_id, lacunes)
            logger.info(
                "student_profile update: student_id='%s'  doc_id=%s", student_id, doc_id
            )
            return json.dumps(
                {"success": True, "doc_id": doc_id, "student_id": student_id},
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("student_profile update error: %s", exc)
            return json.dumps(
                {"success": False, "error": str(exc)},
                ensure_ascii=False,
            )

    async def _arun(self, student_id: str, lacunes_json: str) -> str:
        return self._run(student_id=student_id, lacunes_json=lacunes_json)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_subject_summary(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate gap counts and average scores per subject."""
    summary: dict[str, dict[str, Any]] = {}
    for session in sessions:
        subject = session.get("subject", "general")
        if subject not in summary:
            summary[subject] = {"sessions": 0, "total_score": 0, "gaps": set()}
        summary[subject]["sessions"] += 1
        try:
            summary[subject]["total_score"] += float(session.get("score", 0))
        except (TypeError, ValueError):
            pass
        for gap in session.get("gaps", []):
            summary[subject]["gaps"].add(gap)

    # Convert sets to lists for JSON serialisation
    return {
        subj: {
            "sessions": data["sessions"],
            "average_score": round(
                data["total_score"] / data["sessions"], 2
            ) if data["sessions"] else 0,
            "gaps": sorted(data["gaps"]),
        }
        for subj, data in summary.items()
    }


def _compute_average_score(sessions: list[dict[str, Any]]) -> float | None:
    scores = []
    for s in sessions:
        try:
            max_score = float(s.get("max_score", 20) or 20)
            score = float(s.get("score", 0))
            if max_score > 0:
                scores.append(score / max_score)
        except (TypeError, ValueError):
            pass
    return round(sum(scores) / len(scores), 3) if scores else None


def _infer_mastery(avg_score: float | None) -> str:
    """Human-readable mastery label from normalised average score."""
    if avg_score is None:
        return "inconnu"
    if avg_score >= 0.8:
        return "avancé"
    if avg_score >= 0.5:
        return "intermédiaire"
    return "débutant"
