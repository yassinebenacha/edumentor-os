"""
tools/quiz_generator.py
LangChain tool for adaptive quiz generation using the Groq LLM.

Generates contextualised questions adapted to the student's mastery level.
Output is a structured JSON Quiz consumed by TutorAgent.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Type

from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

# ── Input schema ──────────────────────────────────────────────────────────────

class QuizGeneratorInput(BaseModel):
    topic: str = Field(description="Sujet / concept sur lequel générer le quiz")
    student_id: str = Field(
        default="",
        description="ID de l'étudiant pour adapter la difficulté (optionnel)",
    )
    num_questions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre de questions à générer (1-10, défaut 3)",
    )
    difficulty: str = Field(
        default="auto",
        description=(
            "Niveau de difficulté : 'débutant', 'intermédiaire', 'avancé', "
            "ou 'auto' pour détecter depuis le profil étudiant."
        ),
    )
    question_types: str = Field(
        default="mixed",
        description=(
            "Types de questions : 'qcm', 'vrai_faux', 'réponse_courte', ou 'mixed'."
        ),
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

class QuizGeneratorTool(BaseTool):
    """
    Generates an adaptive quiz on a given topic using the Groq LLM.

    When student_id is provided, the tool fetches the student profile
    from ChromaDB to automatically calibrate difficulty.
    """

    name: str = "quiz_generator"
    description: str = (
        "Génère un quiz adaptatif sur un sujet donné. "
        "Inputs: topic (str), student_id optionnel, num_questions (1-10), "
        "difficulty ('débutant'/'intermédiaire'/'avancé'/'auto'), "
        "question_types ('qcm'/'vrai_faux'/'réponse_courte'/'mixed'). "
        "Retourne un JSON avec les questions, réponses et explications."
    )
    args_schema: Type[BaseModel] = QuizGeneratorInput

    def _run(  # noqa: C901
        self,
        topic: str,
        student_id: str = "",
        num_questions: int = 3,
        difficulty: str = "auto",
        question_types: str = "mixed",
    ) -> str:
        """
        Generate a quiz and return it as a JSON string.

        Returns
        -------
        str
            JSON with keys: topic, difficulty, num_questions, questions[].
            Each question has: type, question, choices (QCM), answer,
            explanation, concept_tested.
        """
        # ── Auto-detect difficulty from student profile ───────────────────
        if difficulty == "auto" and student_id:
            difficulty = _get_difficulty_from_profile(student_id)

        if difficulty == "auto":
            difficulty = "intermédiaire"

        # ── Build LLM prompt ──────────────────────────────────────────────
        prompt = _build_quiz_prompt(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty,
            question_types=question_types,
        )

        # ── Call Groq LLM ─────────────────────────────────────────────────
        try:
            llm = ChatGroq(
                model=settings.llm_model,
                temperature=0.5,   # some creativity for varied questions
                api_key=settings.groq_api_key,
                max_tokens=2048,
            )
            response = llm.invoke(prompt)
            raw_content = response.content

        except Exception as exc:  # noqa: BLE001
            logger.error("quiz_generator: LLM call failed — %s", exc)
            return json.dumps(
                {"error": f"Erreur LLM : {exc}", "topic": topic},
                ensure_ascii=False,
            )

        # ── Parse JSON from response ──────────────────────────────────────
        quiz_data = _extract_quiz_json(raw_content)
        quiz_data.setdefault("topic", topic)
        quiz_data.setdefault("difficulty", difficulty)
        quiz_data.setdefault("num_questions", num_questions)

        logger.info(
            "quiz_generator: topic='%s'  difficulty=%s  questions=%d",
            topic,
            difficulty,
            len(quiz_data.get("questions", [])),
        )
        return json.dumps(quiz_data, ensure_ascii=False, indent=2)

    async def _arun(
        self,
        topic: str,
        student_id: str = "",
        num_questions: int = 3,
        difficulty: str = "auto",
        question_types: str = "mixed",
    ) -> str:
        return self._run(
            topic=topic,
            student_id=student_id,
            num_questions=num_questions,
            difficulty=difficulty,
            question_types=question_types,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_difficulty_from_profile(student_id: str) -> str:
    """Map the student's mastery_level from ChromaDB to a difficulty label."""
    try:
        from tools.student_profile import StudentProfileTool

        tool = StudentProfileTool()
        raw = tool._run(student_id)
        profile = json.loads(raw)
        mastery = profile.get("mastery_level", "inconnu")
        mapping = {
            "débutant": "débutant",
            "intermédiaire": "intermédiaire",
            "avancé": "avancé",
        }
        return mapping.get(mastery, "intermédiaire")
    except Exception as exc:  # noqa: BLE001
        logger.warning("quiz_generator: could not load profile for '%s' — %s", student_id, exc)
        return "intermédiaire"


def _build_quiz_prompt(
    topic: str,
    num_questions: int,
    difficulty: str,
    question_types: str,
) -> str:
    """Return the prompt string for the LLM."""
    type_instruction = {
        "qcm": "Toutes les questions sont à choix multiples (4 options, 1 bonne réponse).",
        "vrai_faux": "Toutes les questions sont de type Vrai/Faux.",
        "réponse_courte": "Toutes les questions nécessitent une réponse courte écrite.",
        "mixed": "Mix de QCM, Vrai/Faux et réponse courte.",
    }.get(question_types, "Mix de types de questions.")

    return f"""\
Génère un quiz pédagogique en français sur le sujet : "{topic}".

Paramètres :
- Nombre de questions : {num_questions}
- Niveau de difficulté : {difficulty}
- Types de questions : {type_instruction}

Réponds UNIQUEMENT avec ce JSON valide, sans texte avant ni après :

{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "num_questions": {num_questions},
  "questions": [
    {{
      "id": 1,
      "type": "qcm",
      "question": "Quel est le résultat de len([1, 2, 3]) en Python ?",
      "choices": ["1", "2", "3", "4"],
      "answer": "3",
      "explanation": "len() retourne le nombre d'éléments de la liste.",
      "concept_tested": "fonctions built-in Python"
    }},
    {{
      "id": 2,
      "type": "vrai_faux",
      "question": "Une liste Python est immuable.",
      "choices": ["Vrai", "Faux"],
      "answer": "Faux",
      "explanation": "Les listes sont mutables. Les tuples sont immuables.",
      "concept_tested": "types mutables vs immuables"
    }}
  ]
}}
"""


def _extract_quiz_json(text: str) -> dict[str, Any]:
    """Extract and parse the JSON quiz from the LLM response text."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try fenced block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # Try first { … }
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("quiz_generator: could not parse JSON, returning raw text")
    return {"raw_response": text, "questions": []}
