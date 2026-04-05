"""
agents/grader_agent.py
Grader Agent for EduMentor-OS.

Orchestrates PDF parsing, rubric retrieval, optional code execution and
structured JSON grading using a Groq-hosted LLM (llama-3.3-70b-versatile).

Architecture
------------
• create_grader_agent()  → LangGraph ReAct agent with 3 tools
• grade_submission()     → high-level async orchestrator
  1. Parse PDF  (PDFParserTool)
  2. Fetch rubric (RubricRetrieverTool or raw text)
  3. Execute code if detected (CodeExecutorTool)
  4. LLM generates structured GradeResult JSON
  5. Persist lacunes in ChromaDB student profile
  6. Return full result dict
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from config import settings
from tools.code_executor import CodeExecutorTool
from tools.pdf_parser import PDFParserTool, extract_pdf_text
from tools.rubric_retriever import RubricRetrieverTool

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 1. SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

GRADER_SYSTEM_PROMPT = """\
Tu es un correcteur pédagogique expert et bienveillant. Tu analyses des copies \
d'examen avec précision et génères un profil de lacunes structuré.

Pour chaque copie, tu dois :
1. Lire attentivement la réponse de l'étudiant
2. Comparer avec le barème fourni
3. Identifier précisément : ce qui est correct, ce qui manque, le type d'erreur
4. Générer une note et un profil de lacunes JSON STRICT

Types d'erreurs possibles :
- "conceptuelle"  (ne comprend pas le concept)
- "application"   (comprend mais applique mal)
- "syntaxe"       (erreur de code/formule)
- "incomplète"    (réponse partielle)
- "hors_sujet"

Tu DOIS répondre uniquement avec ce JSON valide, sans texte avant ni après :

{
  "student_name": "nom si disponible sinon null",
  "subject": "matière",
  "total_score": 14.5,
  "max_score": 20,
  "percentage": 72.5,
  "confidence_score": 0.85,
  "detailed_scores": [
    {
      "question": "Q1",
      "score": 4,
      "max": 5,
      "feedback": "Bonne compréhension du concept, mais erreur dans l'application",
      "lacunes": [
        {"concept": "dérivées composées", "niveau": 0.6, "type_erreur": "application"}
      ]
    }
  ],
  "global_lacunes": [
    {
      "concept": "...",
      "niveau_maitrise": 0.3,
      "priorite": "haute",
      "type_erreur": "conceptuelle"
    }
  ],
  "points_forts": ["..."],
  "recommandations": ["Travailler les exercices de ...", "Revoir le cours sur ..."],
  "feedback_global": "Message encourageant et constructif pour l'étudiant"
}
"""

# ═════════════════════════════════════════════════════════════════════════════
# 2. TOOLS LIST
# ═════════════════════════════════════════════════════════════════════════════

def _build_tools() -> list:
    return [
        PDFParserTool(),
        CodeExecutorTool(),
        RubricRetrieverTool(),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 3. AGENT FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def create_grader_agent():
    """
    Build and return a LangGraph ReAct agent configured for grading.

    Model  : llama-3.3-70b-versatile on Groq (temperature=0.1 for precision)
    Tools  : PDFParserTool, CodeExecutorTool, RubricRetrieverTool

    Returns
    -------
    CompiledGraph
        A compiled LangGraph agent ready to be invoked with
        ``agent.invoke({"messages": [...]})``
    """
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.1,           # low temperature for deterministic grading
        api_key=settings.groq_api_key,
        max_tokens=4096,
    )

    tools = _build_tools()

    # LangGraph ReAct agent — handles tool calls automatically
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=GRADER_SYSTEM_PROMPT,
    )

    logger.info(
        "create_grader_agent: model=%s  tools=%s",
        settings.llm_model,
        [t.name for t in tools],
    )
    return agent


# ═════════════════════════════════════════════════════════════════════════════
# 4. GRADING ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

async def grade_submission(
    pdf_path: str,
    rubric_text: str = "",
    student_id: str = "unknown",
) -> dict[str, Any]:
    """
    Full grading pipeline for one student submission.

    Parameters
    ----------
    pdf_path : str
        Path to the student's PDF exam copy.
    rubric_text : str, optional
        Raw rubric/barème text. If empty, the RubricRetrieverTool is used
        to fetch one automatically from the vector store.
    student_id : str
        Unique student identifier (used to persist the lacune profile).

    Returns
    -------
    dict
        Structured grading result matching the GRADER_SYSTEM_PROMPT schema,
        enriched with ``student_id`` and ``pdf_path``.

    Raises
    ------
    ValueError
        If the PDF cannot be parsed or the LLM response is not valid JSON.
    """

    # ── Step 1 : Parse PDF ────────────────────────────────────────────────
    logger.info("grade_submission: parsing PDF '%s'", pdf_path)
    try:
        pdf_data = extract_pdf_text(pdf_path)
    except ValueError as exc:
        raise ValueError(f"Impossible de lire la copie PDF : {exc}") from exc

    submission_text = pdf_data["text"]
    detected_name = pdf_data.get("student_name") or student_id
    detected_subject = pdf_data.get("subject", "")
    has_code = pdf_data.get("has_code", False)
    code_blocks = pdf_data.get("code_blocks", [])

    logger.info(
        "grade_submission: pages=%d  has_code=%s  student_name='%s'",
        pdf_data.get("pages", 0),
        has_code,
        detected_name,
    )

    # ── Step 2 : Fetch rubric if not provided ─────────────────────────────
    if not rubric_text.strip():
        logger.info("grade_submission: no rubric provided, searching vector store")
        retriever = RubricRetrieverTool()
        raw_rubric = retriever._run(
            query=detected_subject or "barème général",
            subject=detected_subject,
        )
        rubric_data = json.loads(raw_rubric)
        rubric_text = rubric_data.get("rubric_text", "Aucun barème disponible.")

    # ── Step 3 : Execute code blocks if any ──────────────────────────────
    code_execution_summary = ""
    if has_code and code_blocks:
        logger.info("grade_submission: executing %d code block(s)", len(code_blocks))
        from tools.code_executor import execute_code

        exec_results = []
        for idx, block in enumerate(code_blocks[:3]):  # cap at 3 blocks
            try:
                result = execute_code(block)
                exec_results.append(
                    f"[Bloc {idx+1}] success={result['success']} "
                    f"output={result['output']!r} error={result['error']!r}"
                )
            except Exception as exc:  # noqa: BLE001
                exec_results.append(f"[Bloc {idx+1}] erreur d'exécution : {exc}")

        code_execution_summary = "\n".join(exec_results)

    # ── Step 4 : Build the grading prompt ─────────────────────────────────
    user_message = _build_grading_prompt(
        submission_text=submission_text,
        rubric_text=rubric_text,
        student_name=detected_name,
        subject=detected_subject,
        code_execution_summary=code_execution_summary,
    )

    # ── Step 5 : Invoke the LangGraph agent ───────────────────────────────
    logger.info("grade_submission: invoking grader agent for student_id='%s'", student_id)
    agent = create_grader_agent()

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_message)]}
    )

    # Extract the last AI message
    last_message = response["messages"][-1]
    raw_content: str = (
        last_message.content
        if isinstance(last_message.content, str)
        else str(last_message.content)
    )

    # ── Step 6 : Parse JSON from LLM response ────────────────────────────
    grade_result = _extract_json(raw_content)

    # Enrich with metadata
    grade_result.setdefault("student_name", detected_name)
    grade_result.setdefault("subject", detected_subject)
    grade_result["student_id"] = student_id
    grade_result["pdf_path"] = pdf_path

    # ── Step 7 : Persist lacunes in ChromaDB ─────────────────────────────
    try:
        from rag.vector_store import get_vector_store
        import datetime

        vsm = get_vector_store()
        lacunes_payload = {
            "session": datetime.date.today().isoformat(),
            "subject": grade_result.get("subject", ""),
            "score": grade_result.get("total_score", 0),
            "max_score": grade_result.get("max_score", 20),
            "percentage": grade_result.get("percentage", 0),
            "gaps": [
                g["concept"]
                for g in grade_result.get("global_lacunes", [])
            ],
            "recommandations": grade_result.get("recommandations", []),
        }
        vsm.update_student_profile(student_id, lacunes_payload)
        logger.info(
            "grade_submission: student profile updated for student_id='%s'", student_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("grade_submission: failed to update student profile — %s", exc)

    logger.info(
        "grade_submission: done — score=%.1f/%.1f (%.0f%%)",
        grade_result.get("total_score", 0),
        grade_result.get("max_score", 20),
        grade_result.get("percentage", 0),
    )
    return grade_result


# ═════════════════════════════════════════════════════════════════════════════
# 5. PRIVATE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _build_grading_prompt(
    submission_text: str,
    rubric_text: str,
    student_name: str,
    subject: str,
    code_execution_summary: str,
) -> str:
    """Assemble the human message sent to the grader agent."""
    parts = [
        f"## Copie à corriger",
        f"**Étudiant :** {student_name}",
        f"**Matière :** {subject or 'Non spécifiée'}",
        "",
        "### Texte de la copie",
        submission_text[:6000],  # cap to ~6 000 chars for context window safety
    ]

    if code_execution_summary:
        parts += [
            "",
            "### Résultats d'exécution du code",
            code_execution_summary,
        ]

    parts += [
        "",
        "## Barème de correction",
        rubric_text[:3000],
        "",
        "---",
        "Génère maintenant la correction complète au format JSON exact demandé.",
    ]

    return "\n".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract and parse the first JSON object found in *text*.

    Tries three strategies in order:
      1. Direct json.loads (if the response is pure JSON)
      2. Regex: first { … } block
      3. Markdown code fence: ```json … ```

    Raises
    ------
    ValueError
        If no valid JSON object can be found or parsed.
    """
    text = text.strip()

    # Strategy 1: pure JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: first {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "L'agent n'a pas retourné un JSON valide. "
        f"Réponse brute (100 chars) : {text[:100]!r}"
    )
