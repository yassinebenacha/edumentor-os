"""
graphs/main_graph.py
LangGraph StateGraph orchestrating EduMentor-OS's three agents.

Flow
----
                        START
                          │
              ┌───────────▼────────────┐
              │     route_by_task      │  (conditional router)
              └──┬──────────┬─────────┘
           grade │    chat  │  analyze_class
                 ▼          ▼           ▼
          node_grader  node_tutor  node_analyst
                 │          │           │
                 ▼          │           │
        node_quality_check  │           │
                 │          │           │
       OK ───────┘    ──────┘    ───────┘
       LOW ──► node_human_review          
       ERR ──► node_error_handler        
                          END

Checkpointing
-------------
MemorySaver provides in-memory thread-based persistence.
Pass ``config={"configurable": {"thread_id": student_id}}`` to preserve
state across multiple calls for the same student.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, Optional

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import settings

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 1. SHARED STATE DEFINITION
# ═════════════════════════════════════════════════════════════════════════════

class EduMentorState(TypedDict, total=False):
    """
    Shared mutable state passed between all nodes of the EduMentor graph.

    Fields with ``total=False`` are optional — nodes only write what they
    produce and leave other fields untouched.
    """

    # ── Input fields (set by the caller) ─────────────────────────────────
    task_type: str            # "grade" | "analyze_class" | "tutor_chat"
    student_id: str           # Unique student identifier
    pdf_path: Optional[str]   # Path to the student's PDF submission
    rubric_text: Optional[str]# Raw rubric / barème text
    user_message: Optional[str]        # Student's chat message (tutor tasks)
    conversation_history: list[dict]   # [{"role": "human"|"ai", "content": "..."}]
    class_data: Optional[list[dict]]   # List of grading results for class analysis

    # ── Intermediate results (set by nodes) ──────────────────────────────
    grading_result: Optional[dict]    # Output from GraderAgent
    class_analysis: Optional[dict]   # Output from AnalystAgent
    tutor_response: Optional[str]    # Tutor's reply
    updated_history: Optional[list[dict]]   # History after tutor response

    # ── Flow control ─────────────────────────────────────────────────────
    error: Optional[str]          # Error message (if any node fails)
    confidence_score: float        # Grader confidence [0.0, 1.0]
    needs_human_review: bool       # True if confidence < threshold


# ─── Default state factory ────────────────────────────────────────────────────

def _default_state() -> EduMentorState:
    """Return a skeleton state with safe defaults for optional fields."""
    return EduMentorState(
        task_type="grade",
        student_id="unknown",
        pdf_path=None,
        rubric_text=None,
        user_message=None,
        conversation_history=[],
        class_data=None,
        grading_result=None,
        class_analysis=None,
        tutor_response=None,
        updated_history=None,
        error=None,
        confidence_score=1.0,
        needs_human_review=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. GRAPH NODES
# ═════════════════════════════════════════════════════════════════════════════

# ── node_grader ───────────────────────────────────────────────────────────────

async def node_grader(state: EduMentorState) -> EduMentorState:
    """
    Invoke GraderAgent to evaluate a student submission.

    Reads  : pdf_path, rubric_text, student_id
    Writes : grading_result, confidence_score, error
    """
    logger.info("node_grader: student_id='%s'  pdf='%s'", state["student_id"], state.get("pdf_path"))

    pdf_path = state.get("pdf_path", "")
    if not pdf_path:
        return {**state, "error": "pdf_path manquant pour la tâche 'grade'."}

    try:
        from agents.grader_agent import grade_submission

        result = await grade_submission(
            pdf_path=pdf_path,
            rubric_text=state.get("rubric_text") or "",
            student_id=state["student_id"],
        )
        confidence = float(result.get("confidence_score", 0.8))
        logger.info(
            "node_grader: done  score=%.1f/%.1f  confidence=%.2f",
            result.get("total_score", 0),
            result.get("max_score", 20),
            confidence,
        )
        return {**state, "grading_result": result, "confidence_score": confidence, "error": None}

    except Exception as exc:  # noqa: BLE001
        logger.error("node_grader: failed — %s", exc)
        return {**state, "error": f"Erreur GraderAgent : {exc}", "confidence_score": 0.0}


# ── node_quality_check ────────────────────────────────────────────────────────

async def node_quality_check(state: EduMentorState) -> EduMentorState:
    """
    Assess grading quality and flag low-confidence results for human review.

    Threshold : confidence_score < 0.6 → needs_human_review = True

    Reads  : confidence_score, grading_result, error
    Writes : needs_human_review
    """
    CONFIDENCE_THRESHOLD = 0.6  # noqa: N806

    if state.get("error"):
        logger.warning("node_quality_check: error state detected, skipping")
        return {**state, "needs_human_review": True}

    confidence = state.get("confidence_score", 1.0)
    needs_review = confidence < CONFIDENCE_THRESHOLD

    if needs_review:
        logger.warning(
            "node_quality_check: LOW confidence=%.2f → flagging for human review",
            confidence,
        )
    else:
        logger.info(
            "node_quality_check: OK confidence=%.2f → proceeding to analyst",
            confidence,
        )

    return {**state, "needs_human_review": needs_review}


# ── node_analyst ──────────────────────────────────────────────────────────────

async def node_analyst(state: EduMentorState) -> EduMentorState:
    """
    Invoke AnalystAgent to generate class-level or individual learning reports.

    Two modes:
      • task_type == "grade"          → post-grading individual analysis
      • task_type == "analyze_class"  → class-wide aggregated analysis

    Reads  : grading_result | class_data, student_id
    Writes : class_analysis, error
    """
    logger.info("node_analyst: task_type='%s'  student_id='%s'", state.get("task_type"), state["student_id"])

    try:
        analysis = await _run_analyst(state)
        logger.info("node_analyst: analysis complete")
        return {**state, "class_analysis": analysis, "error": None}

    except Exception as exc:  # noqa: BLE001
        logger.error("node_analyst: failed — %s", exc)
        return {**state, "error": f"Erreur AnalystAgent : {exc}"}


async def _run_analyst(state: EduMentorState) -> dict[str, Any]:
    """
    Build the analyst prompt and call the LLM directly.

    For now uses a direct LLM call with a structured prompt.
    Will be replaced by AnalystAgent when agents/analyst_agent.py is implemented.
    """
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.2,
        api_key=settings.groq_api_key,
        max_tokens=2048,
    )

    if state.get("task_type") == "analyze_class":
        class_data = state.get("class_data") or []
        prompt = _build_class_analysis_prompt(class_data)
    else:
        # Post-grading individual analysis
        grading_result = state.get("grading_result") or {}
        prompt = _build_individual_analysis_prompt(grading_result, state["student_id"])

    response = await llm.ainvoke(prompt)
    raw = response.content if isinstance(response.content, str) else str(response.content)

    # Try to extract JSON; fall back to plain text summary
    import json, re  # noqa: E401

    try:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(brace.group(0)) if brace else {"summary": raw}
    except (json.JSONDecodeError, AttributeError):
        return {"summary": raw}


def _build_individual_analysis_prompt(grading: dict[str, Any], student_id: str) -> str:
    import json

    return (
        f"Analyse le résultat de correction suivant pour l'étudiant '{student_id}' "
        f"et génère des recommandations pédagogiques détaillées en JSON :\n\n"
        f"```json\n{json.dumps(grading, ensure_ascii=False, indent=2)[:3000]}\n```\n\n"
        "Retourne un JSON avec : "
        "{ summary, key_weaknesses[], priority_topics[], study_plan[], "
        "estimated_revision_hours, encouragement_message }"
    )


def _build_class_analysis_prompt(class_data: list[dict[str, Any]]) -> str:
    import json

    return (
        f"Analyse les résultats de {len(class_data)} étudiants et génère un rapport "
        f"de classe synthétique en JSON :\n\n"
        f"```json\n{json.dumps(class_data, ensure_ascii=False, indent=2)[:4000]}\n```\n\n"
        "Retourne un JSON avec : "
        "{ class_average, class_median, top_performers[], at_risk_students[], "
        "shared_weaknesses[], recommended_class_activities[], overall_summary }"
    )


# ── node_tutor ────────────────────────────────────────────────────────────────

async def node_tutor(state: EduMentorState) -> EduMentorState:
    """
    Invoke TutorAgent for an adaptive tutoring response.

    Reads  : student_id, user_message, conversation_history
    Writes : tutor_response, updated_history, error
    """
    student_id = state["student_id"]
    message = state.get("user_message", "")
    history = state.get("conversation_history") or []

    logger.info(
        "node_tutor: student_id='%s'  message='%s'  history_turns=%d",
        student_id,
        message[:60],
        len(history),
    )

    if not message:
        return {**state, "error": "user_message manquant pour la tâche 'tutor_chat'."}

    try:
        from agents.tutor_agent import chat_with_tutor

        response, updated_history = await chat_with_tutor(
            student_id=student_id,
            message=message,
            conversation_history=history,
        )
        logger.info("node_tutor: response generated (%d chars)", len(response))
        return {
            **state,
            "tutor_response": response,
            "updated_history": updated_history,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("node_tutor: failed — %s", exc)
        return {**state, "error": f"Erreur TutorAgent : {exc}"}


# ── node_human_review ─────────────────────────────────────────────────────────

async def node_human_review(state: EduMentorState) -> EduMentorState:
    """
    Flag the grading result for manual review by an educator.

    This node does NOT block — it annotates the state and lets the graph
    reach END. The API layer reads ``needs_human_review=True`` and
    notifies the educator through its own notification channel.

    Reads  : grading_result, confidence_score, student_id
    Writes : grading_result (adds review flag and message)
    """
    logger.warning(
        "node_human_review: flagging student_id='%s'  confidence=%.2f",
        state["student_id"],
        state.get("confidence_score", 0.0),
    )
    grading = dict(state.get("grading_result") or {})
    grading["_review_required"] = True
    grading["_review_reason"] = (
        f"Confiance de correction trop faible ({state.get('confidence_score', 0):.0%}). "
        "Une vérification manuelle par l'enseignant est recommandée."
    )
    return {**state, "grading_result": grading}


# ── node_error_handler ────────────────────────────────────────────────────────

async def node_error_handler(state: EduMentorState) -> EduMentorState:
    """
    Log and surface any error accumulated in the state.

    Reads  : error, task_type, student_id
    Writes : (state unchanged, only logs)
    """
    error_msg = state.get("error", "Erreur inconnue")
    logger.error(
        "node_error_handler: task='%s'  student_id='%s'  error=%s",
        state.get("task_type"),
        state.get("student_id"),
        error_msg,
    )
    return state  # Pass through — API layer reads state.error


# ═════════════════════════════════════════════════════════════════════════════
# 3. ROUTING FUNCTIONS (conditional edges)
# ═════════════════════════════════════════════════════════════════════════════

def route_by_task(
    state: EduMentorState,
) -> Literal["node_grader", "node_tutor", "node_analyst", "node_error_handler"]:
    """Entry router: dispatch to the right first node based on task_type."""
    task = state.get("task_type", "")
    if task == "grade":
        return "node_grader"
    if task == "tutor_chat":
        return "node_tutor"
    if task == "analyze_class":
        return "node_analyst"
    logger.error("route_by_task: unknown task_type='%s'", task)
    return "node_error_handler"


def route_after_quality_check(
    state: EduMentorState,
) -> Literal["node_analyst", "node_human_review", "node_error_handler"]:
    """Post-quality-check router."""
    if state.get("error"):
        return "node_error_handler"
    if state.get("needs_human_review"):
        return "node_human_review"
    return "node_analyst"


def route_after_error(state: EduMentorState) -> Literal["__end__"]:
    """All error paths lead to END."""
    return END


# ═════════════════════════════════════════════════════════════════════════════
# 4. GRAPH COMPILATION
# ═════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Assemble and compile the EduMentor StateGraph with MemorySaver checkpointing.

    Returns
    -------
    CompiledGraph
        Ready-to-invoke LangGraph application.
    """
    builder = StateGraph(EduMentorState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("node_grader",        node_grader)
    builder.add_node("node_quality_check", node_quality_check)
    builder.add_node("node_analyst",       node_analyst)
    builder.add_node("node_tutor",         node_tutor)
    builder.add_node("node_human_review",  node_human_review)
    builder.add_node("node_error_handler", node_error_handler)

    # ── Entry conditional edge ────────────────────────────────────────────
    builder.add_conditional_edges(
        START,
        route_by_task,
        {
            "node_grader":        "node_grader",
            "node_tutor":         "node_tutor",
            "node_analyst":       "node_analyst",
            "node_error_handler": "node_error_handler",
        },
    )

    # ── Grading path ──────────────────────────────────────────────────────
    # grader → quality_check → (analyst | human_review | error_handler)
    builder.add_edge("node_grader", "node_quality_check")
    builder.add_conditional_edges(
        "node_quality_check",
        route_after_quality_check,
        {
            "node_analyst":       "node_analyst",
            "node_human_review":  "node_human_review",
            "node_error_handler": "node_error_handler",
        },
    )

    # ── Terminal edges ────────────────────────────────────────────────────
    builder.add_edge("node_tutor",         END)
    builder.add_edge("node_analyst",       END)
    builder.add_edge("node_human_review",  END)
    builder.add_edge("node_error_handler", END)

    # ── Compile with in-memory checkpointer ───────────────────────────────
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    logger.info("EduMentor graph compiled successfully")
    return app


# ── Module-level compiled app (lazy) ─────────────────────────────────────────
_app = None


def get_app():
    """Return (or build) the compiled LangGraph application singleton."""
    global _app  # noqa: PLW0603
    if _app is None:
        _app = build_graph()
    return _app


# ═════════════════════════════════════════════════════════════════════════════
# 5. PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

async def run_graph(
    initial_state: dict[str, Any],
    thread_id: str | None = None,
) -> EduMentorState:
    """
    Invoke the EduMentor StateGraph with *initial_state*.

    Parameters
    ----------
    initial_state : dict
        Fields to inject into the graph state. Must include at minimum
        ``task_type`` and ``student_id``.
    thread_id : str, optional
        Checkpointer thread identifier — use the student's ID for persistent
        memory across multiple calls (tutor chat sessions).
        Defaults to ``initial_state["student_id"]``.

    Returns
    -------
    EduMentorState
        Final state after the graph has reached END.

    Examples
    --------
    Grade a submission::

        result = await run_graph({
            "task_type": "grade",
            "student_id": "alice_01",
            "pdf_path": "data/sample_copies/exam_alice.pdf",
            "rubric_text": "Q1 (5pts): ...",
        })
        print(result["grading_result"]["total_score"])

    Chat with tutor::

        result = await run_graph({
            "task_type": "tutor_chat",
            "student_id": "alice_01",
            "user_message": "Explique-moi la récursivité",
            "conversation_history": [],
        }, thread_id="alice_01")
        print(result["tutor_response"])
    """
    # Merge with defaults so optional fields are always present
    full_state: EduMentorState = {**_default_state(), **initial_state}

    if thread_id is None:
        thread_id = full_state.get("student_id", "default")

    config = {"configurable": {"thread_id": thread_id}}

    app = get_app()
    logger.info(
        "run_graph: task='%s'  student_id='%s'  thread_id='%s'",
        full_state.get("task_type"),
        full_state.get("student_id"),
        thread_id,
    )

    final_state: EduMentorState = await app.ainvoke(full_state, config=config)

    if final_state.get("error"):
        logger.error("run_graph: finished with error — %s", final_state["error"])
    else:
        logger.info("run_graph: completed successfully")

    return final_state


def run_graph_sync(
    initial_state: dict[str, Any],
    thread_id: str | None = None,
) -> EduMentorState:
    """
    Synchronous wrapper around ``run_graph`` for scripts and notebooks.

    Example
    -------
    >>> state = run_graph_sync({
    ...     "task_type": "tutor_chat",
    ...     "student_id": "alice_01",
    ...     "user_message": "Qu'est-ce qu'une liste en Python ?",
    ... })
    >>> print(state["tutor_response"])
    """
    return asyncio.run(run_graph(initial_state, thread_id))
