"""
agents/tutor_agent.py
Adaptive Tutor Agent for EduMentor-OS.

Orchestrates personalised tutoring sessions using:
  • StudentProfileTool  — loads the student's gap profile for adaptation
  • QuizGeneratorTool   — generates follow-up exercises
  • TavilySearchTool    — enriches explanations with real-world examples
  • CodeExecutorTool    — validates live code snippets during the session

Uses Groq llama-3.3-70b-versatile (temperature=0.3) for creativity while
remaining pedagogically coherent.

Public API
----------
create_tutor_agent()                → LangGraph ReAct agent
chat_with_tutor(student_id, msg, history) → (response_str, updated_history)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from config import settings
from tools.code_executor import CodeExecutorTool
from tools.quiz_generator import QuizGeneratorTool
from tools.student_profile import StudentProfileTool

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 1. SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

TUTOR_SYSTEM_PROMPT = """\
Tu es EduMentor, un tuteur pédagogique expert, bienveillant et adaptatif.
Tu parles en français (ou en darija marocain si l'étudiant le demande).

Tu as accès au profil de lacunes de l'étudiant via l'outil student_profile.
Tu dois TOUJOURS :
1. Commencer par consulter le profil de l'étudiant (student_profile) avant de répondre
2. Adapter ton niveau de difficulté à son profil :
   - niveau_maitrise < 0.5  → explique depuis la base, vocabulaire simple
   - niveau entre 0.5–0.8   → consolide avec des exemples variés
   - niveau > 0.8           → propose des défis avancés et des cas limites
3. Utiliser la méthode Socratique : poser des questions guidantes AVANT de donner la réponse
4. Structurer chaque explication en 3 temps :
   [📚 Rappel du concept] → [💡 Exemple concret] → [✏️ Exercice pratique]
5. Terminer chaque explication par un mini-quiz de vérification (2-3 questions)
   généré avec l'outil quiz_generator

Pour les questions de programmation Python :
- Fournis du code exécutable avec des commentaires pédagogiques
- Utilise code_executor pour valider le code avant de le présenter
- Propose des variantes progressives (facile → intermédiaire → difficile)

Pour des ressources supplémentaires :
- Utilise tavily_search pour trouver des exemples réels et actualisés

Règles de ton :
- Sois encourageant, jamais condescendant
- Célèbre les progrès ("Excellent ! Tu as bien compris que…")
- Si l'étudiant se trompe : "Pas tout à fait, réfléchis à… Qu'est-ce qui se passe si…?"
- Utilise des emojis pédagogiques avec modération (📚 💡 ✏️ 🎯 ✅ ❌)
"""

# ═════════════════════════════════════════════════════════════════════════════
# 2. TOOLS LIST
# ═════════════════════════════════════════════════════════════════════════════

def _build_tutor_tools() -> list:
    """Instantiate and return all tools available to the tutor agent."""
    tools: list = [
        StudentProfileTool(),
        QuizGeneratorTool(),
        CodeExecutorTool(),
    ]

    # Tavily is optional (no key → graceful skip)
    if settings.tavily_api_key:
        tools.append(
            TavilySearchResults(
                max_results=3,
                api_key=settings.tavily_api_key,
                name="tavily_search",
                description=(
                    "Recherche sur le web des exemples réels, définitions ou "
                    "ressources pédagogiques actualisées."
                ),
            )
        )
        logger.debug("tutor_agent: Tavily search tool enabled")
    else:
        logger.info("tutor_agent: TAVILY_API_KEY not set — web search disabled")

    return tools


# ═════════════════════════════════════════════════════════════════════════════
# 3. AGENT FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def create_tutor_agent():
    """
    Build and return a LangGraph ReAct agent configured for adaptive tutoring.

    Model  : llama-3.3-70b-versatile on Groq (temperature=0.3)
    Tools  : StudentProfileTool, QuizGeneratorTool, CodeExecutorTool,
             TavilySearchResults (if TAVILY_API_KEY is set)

    Returns
    -------
    CompiledGraph
        LangGraph agent ready for invocation via ``agent.ainvoke()``.
    """
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.3,          # creative but coherent pedagogy
        api_key=settings.groq_api_key,
        max_tokens=4096,
    )

    tools = _build_tutor_tools()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=TUTOR_SYSTEM_PROMPT,
    )

    logger.info(
        "create_tutor_agent: model=%s  tools=%s",
        settings.llm_model,
        [t.name for t in tools],
    )
    return agent


# ═════════════════════════════════════════════════════════════════════════════
# 4. CONVERSATION ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

async def chat_with_tutor(
    student_id: str,
    message: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """
    Send *message* to the tutor agent and return the response with updated history.

    The conversation memory is carried as a plain list of dicts so it can be
    stored and transmitted easily (JSON-serialisable, no LangChain objects).

    Parameters
    ----------
    student_id : str
        Unique student identifier. Used to inject profile context and will
        be passed to tools that need it.
    message : str
        The student's current message or question.
    conversation_history : list[dict], optional
        Previous turns: ``[{"role": "human"|"ai", "content": "…"}, …]``.
        Defaults to an empty list (new session).

    Returns
    -------
    tuple[str, list[dict]]
        (ai_response_text, updated_conversation_history)

    Notes
    -----
    * The system message (TUTOR_SYSTEM_PROMPT) is **always** prepended.
    * A context-injection message is added before the conversation so the
      agent always knows which student it is speaking with.
    * History is capped at the last 20 turns to stay within context limits.
    """
    if conversation_history is None:
        conversation_history = []

    # ── Build LangChain message list ──────────────────────────────────────
    lc_messages: list = [SystemMessage(content=TUTOR_SYSTEM_PROMPT)]

    # Inject student context as a synthetic first human message
    context_msg = (
        f"[CONTEXTE SYSTÈME] L'étudiant qui me parle a l'ID : '{student_id}'. "
        "Commence par consulter son profil avec student_profile avant de répondre "
        "si c'est le début de la conversation."
    )
    lc_messages.append(HumanMessage(content=context_msg))
    lc_messages.append(
        AIMessage(content="Compris. Je vais consulter le profil de l'étudiant.")
    )

    # Replay history (cap at last 20 turns = 40 messages)
    recent_history = conversation_history[-40:]
    for turn in recent_history:
        role = turn.get("role", "human")
        content = turn.get("content", "")
        if role == "human":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=content))

    # Current user message
    lc_messages.append(HumanMessage(content=message))

    # ── Invoke the agent ──────────────────────────────────────────────────
    logger.info(
        "chat_with_tutor: student_id='%s'  history_turns=%d  message='%s'",
        student_id,
        len(conversation_history),
        message[:80],
    )

    agent = create_tutor_agent()
    try:
        response = await agent.ainvoke({"messages": lc_messages})
        last_msg = response["messages"][-1]
        ai_response: str = (
            last_msg.content
            if isinstance(last_msg.content, str)
            else str(last_msg.content)
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("chat_with_tutor: agent invocation failed — %s", exc)
        ai_response = (
            "Désolé, une erreur s'est produite. Reformule ta question et réessaie. 🙏"
        )

    # ── Update conversation history ───────────────────────────────────────
    updated_history = list(conversation_history) + [
        {"role": "human", "content": message},
        {"role": "ai", "content": ai_response},
    ]

    # Keep only the last 40 turns in memory (80 messages max)
    if len(updated_history) > 80:
        updated_history = updated_history[-80:]

    logger.info(
        "chat_with_tutor: response generated  history_turns=%d",
        len(updated_history) // 2,
    )
    return ai_response, updated_history


# ═════════════════════════════════════════════════════════════════════════════
# 5. UTILITY — Sync wrapper for non-async contexts
# ═════════════════════════════════════════════════════════════════════════════

def chat_with_tutor_sync(
    student_id: str,
    message: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """
    Synchronous wrapper around ``chat_with_tutor`` for scripts/notebooks.

    Example
    -------
    >>> response, history = chat_with_tutor_sync("alice_01", "Explique la récursivité")
    >>> print(response)
    """
    import asyncio

    return asyncio.run(
        chat_with_tutor(student_id, message, conversation_history)
    )
