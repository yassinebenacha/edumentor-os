"""
api/main.py
EduMentor OS — FastAPI HTTP layer.

Endpoints
---------
POST /api/grade                  Grade a student PDF submission
POST /api/tutor/chat             Adaptive tutoring conversation
GET  /api/student/{id}/profile   Fetch student learning profile
POST /api/class/analyze          Class-level analytics report
POST /api/knowledge/upload       Ingest a document into the knowledge base
GET  /api/health                 Service health check

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

# ═════════════════════════════════════════════════════════════════════════════
# 1. APPLICATION INIT
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="EduMentor OS API",
    version="1.0.0",
    description=(
        "Multi-agent educational platform — automated grading, adaptive tutoring "
        "and learning analytics powered by LangGraph + Groq."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (open for hackathon) ─────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request timing / logging middleware ───────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status code and elapsed time for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.0f}"
    return response


# ═════════════════════════════════════════════════════════════════════════════
# 2. REQUEST / RESPONSE SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class TutorChatRequest(BaseModel):
    student_id: str = Field(..., description="Unique student identifier")
    message: str = Field(..., min_length=1, description="Student's message")
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous turns: [{role: human|ai, content: ...}]",
    )


class ClassAnalyzeRequest(BaseModel):
    student_ids: list[str] = Field(..., min_length=1, description="List of student IDs to analyse")
    subject: str = Field(default="", description="Subject filter (optional)")


class TutorChatResponse(BaseModel):
    response: str
    conversation_history: list[dict[str, str]]
    student_profile_summary: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    agents: str
    vectordb: str
    llm_model: str
    version: str


# ═════════════════════════════════════════════════════════════════════════════
# 3. STARTUP / SHUTDOWN EVENTS
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup() -> None:
    """Pre-warm the vector store and compile the graph on startup."""
    logger.info("EduMentor OS API starting up …")

    # Ensure data directories exist
    for path in (settings.data_dir, settings.chroma_persist_dir,
                 settings.knowledge_base_dir, settings.rubrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Pre-warm ChromaDB (creates collections if they don't exist)
    try:
        from rag.vector_store import get_vector_store
        get_vector_store()
        logger.info("ChromaDB: collections ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ChromaDB warm-up failed (non-fatal): %s", exc)

    # Pre-compile the LangGraph application
    try:
        from graphs.main_graph import get_app
        get_app()
        logger.info("LangGraph: graph compiled and ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("LangGraph pre-compile failed (non-fatal): %s", exc)

    logger.info("Startup complete — listening on port %d", settings.api_port)


# ═════════════════════════════════════════════════════════════════════════════
# 4. ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

# ── 4.1 Health check ──────────────────────────────────────────────────────────

@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Service health check",
)
async def health_check() -> HealthResponse:
    """
    Verify that the API, ChromaDB and LLM configuration are reachable.
    Returns a 200 OK with component statuses.
    """
    vectordb_status = "unknown"
    try:
        from rag.vector_store import get_vector_store
        get_vector_store()
        vectordb_status = "connected"
    except Exception as exc:  # noqa: BLE001
        vectordb_status = f"error: {exc}"

    agents_status = "ready" if settings.groq_api_key else "no GROQ_API_KEY set"

    return HealthResponse(
        status="ok",
        agents=agents_status,
        vectordb=vectordb_status,
        llm_model=settings.llm_model,
        version="1.0.0",
    )


# ── 4.2 Grade a student submission ────────────────────────────────────────────

@app.post(
    "/api/grade",
    tags=["grading"],
    summary="Grade a student PDF exam submission",
)
async def grade_submission(
    file: UploadFile = File(..., description="Student exam PDF"),
    rubric_text: str = Form(default="", description="Grading rubric text (optional)"),
    student_id: str = Form(..., description="Unique student identifier"),
) -> JSONResponse:
    """
    Upload a PDF exam copy and receive a structured grade with lacune profile.

    - Saves the PDF to a temporary file
    - Invokes the LangGraph grading pipeline (timeout: 120 s)
    - Returns the full GradeResult JSON
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted.",
        )

    # Save to a temp file (LangChain tools need a real path)
    suffix = f"_{student_id}.pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(
        "POST /api/grade  student_id='%s'  file='%s'  size=%d bytes",
        student_id, file.filename, len(content),
    )

    try:
        from graphs.main_graph import run_graph

        result_state = await asyncio.wait_for(
            run_graph(
                {
                    "task_type": "grade",
                    "student_id": student_id,
                    "pdf_path": tmp_path,
                    "rubric_text": rubric_text,
                }
            ),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Grading timed out after 120 s. Try a shorter document.",
        )
    finally:
        # Always clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result_state.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result_state["error"],
        )

    return JSONResponse(
        content={
            "success": True,
            "student_id": student_id,
            "needs_human_review": result_state.get("needs_human_review", False),
            "grading_result": result_state.get("grading_result"),
            "class_analysis": result_state.get("class_analysis"),
        }
    )


# ── 4.3 Tutor chat ────────────────────────────────────────────────────────────

@app.post(
    "/api/tutor/chat",
    response_model=TutorChatResponse,
    tags=["tutoring"],
    summary="Send a message to the adaptive tutor",
)
async def tutor_chat(body: TutorChatRequest) -> TutorChatResponse:
    """
    Engage in a tutoring conversation.

    Pass ``conversation_history`` from the previous response to maintain context.
    The tutor automatically adapts its pedagogical level to the student's profile.
    """
    logger.info(
        "POST /api/tutor/chat  student_id='%s'  message='%s'",
        body.student_id, body.message[:60],
    )

    try:
        from graphs.main_graph import run_graph

        result_state = await asyncio.wait_for(
            run_graph(
                {
                    "task_type": "tutor_chat",
                    "student_id": body.student_id,
                    "user_message": body.message,
                    "conversation_history": body.conversation_history,
                },
                thread_id=body.student_id,
            ),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Tutor response timed out. Please retry.",
        )

    if result_state.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result_state["error"],
        )

    # Fetch a lightweight profile summary to help the frontend
    profile_summary = await _get_profile_summary(body.student_id)

    return TutorChatResponse(
        response=result_state.get("tutor_response", ""),
        conversation_history=result_state.get("updated_history", body.conversation_history),
        student_profile_summary=profile_summary,
    )


# ── 4.4 Student profile ───────────────────────────────────────────────────────

@app.get(
    "/api/student/{student_id}/profile",
    tags=["students"],
    summary="Get a student's full learning profile",
)
async def get_student_profile(student_id: str) -> JSONResponse:
    """
    Retrieve the complete learning history and gap profile for *student_id*
    from the ChromaDB student_profiles collection.
    """
    logger.info("GET /api/student/%s/profile", student_id)

    try:
        from rag.vector_store import get_vector_store

        vsm = get_vector_store()
        profile = vsm.get_student_profile(student_id)

        # Compute progression trend (last 5 sessions)
        profile["progression"] = _compute_progression(profile.get("sessions", []))

        return JSONResponse(content={"success": True, "profile": profile})

    except Exception as exc:  # noqa: BLE001
        logger.error("get_student_profile: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ── 4.5 Class analysis ────────────────────────────────────────────────────────

@app.post(
    "/api/class/analyze",
    tags=["analytics"],
    summary="Generate a class-level analytics report",
)
async def analyze_class(body: ClassAnalyzeRequest) -> JSONResponse:
    """
    Aggregate learning profiles for a list of students and generate a
    collective report identifying shared weaknesses and at-risk students.
    """
    logger.info(
        "POST /api/class/analyze  students=%d  subject='%s'",
        len(body.student_ids), body.subject,
    )

    try:
        from rag.vector_store import get_vector_store

        vsm = get_vector_store()
        class_data: list[dict] = []
        for sid in body.student_ids:
            profile = vsm.get_student_profile(sid)
            profile["student_id"] = sid
            if body.subject:
                profile["sessions"] = [
                    s for s in profile.get("sessions", [])
                    if s.get("subject", "") == body.subject
                ]
            class_data.append(profile)

    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur collecte des profils : {exc}",
        )

    try:
        from graphs.main_graph import run_graph

        result_state = await asyncio.wait_for(
            run_graph(
                {
                    "task_type": "analyze_class",
                    "student_id": "class_report",
                    "class_data": class_data,
                }
            ),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Class analysis timed out after 90 s.",
        )

    if result_state.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result_state["error"],
        )

    return JSONResponse(
        content={
            "success": True,
            "subject": body.subject,
            "num_students": len(body.student_ids),
            "class_analysis": result_state.get("class_analysis"),
        }
    )


# ── 4.6 Knowledge base upload ─────────────────────────────────────────────────

@app.post(
    "/api/knowledge/upload",
    tags=["knowledge"],
    summary="Upload a document to the knowledge base or rubric store",
)
async def upload_knowledge(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or TXT document to ingest"),
    subject: str = Form(default="general", description="Subject category"),
    doc_type: str = Form(
        default="course",
        description="Document type: 'course', 'exercise', or 'rubric'",
    ),
) -> JSONResponse:
    """
    Ingest a PDF or TXT document into the appropriate ChromaDB collection.

    - ``rubric`` type → ``rubric_store`` collection
    - ``course`` / ``exercise`` → ``knowledge_base`` collection

    Ingestion runs in the background (task queued, returns immediately).
    """
    if doc_type not in ("course", "exercise", "rubric"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="doc_type must be one of: course, exercise, rubric",
        )

    filename = file.filename or "upload"
    content = await file.read()

    logger.info(
        "POST /api/knowledge/upload  file='%s'  subject='%s'  type='%s'  size=%d",
        filename, subject, doc_type, len(content),
    )

    background_tasks.add_task(
        _ingest_document,
        content=content,
        filename=filename,
        subject=subject,
        doc_type=doc_type,
    )

    return JSONResponse(
        content={
            "success": True,
            "message": f"Document '{filename}' queued for ingestion.",
            "subject": subject,
            "doc_type": doc_type,
        },
        status_code=status.HTTP_202_ACCEPTED,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. BACKGROUND HELPERS
# ═════════════════════════════════════════════════════════════════════════════

async def _ingest_document(
    content: bytes,
    filename: str,
    subject: str,
    doc_type: str,
) -> None:
    """
    Background task: extract text from *content* and upsert into ChromaDB.

    Splits the raw text into chunks before embedding to stay within
    ChromaDB's recommended document size.
    """
    logger.info("_ingest_document: start  file='%s'  type='%s'", filename, doc_type)

    # 1. Extract text
    try:
        if filename.lower().endswith(".pdf"):
            import fitz  # PyMuPDF

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                doc = fitz.open(tmp_path)
                raw_text = "\n".join(page.get_text("text") for page in doc)
                doc.close()
            finally:
                os.unlink(tmp_path)
        else:
            raw_text = content.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        logger.error("_ingest_document: text extraction failed — %s", exc)
        return

    if not raw_text.strip():
        logger.warning("_ingest_document: empty text, skipping '%s'", filename)
        return

    # 2. Chunk text (simple fixed-size split)
    chunks = _chunk_text(raw_text, chunk_size=1000, overlap=100)

    # 3. Upsert into ChromaDB
    try:
        from rag.vector_store import get_vector_store

        vsm = get_vector_store()
        collection = (
            "rubric_store"
            if doc_type == "rubric"
            else "knowledge_base"
        )
        metadatas = [
            {
                "filename": filename,
                "subject": subject,
                "doc_type": doc_type,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        ids = vsm.add_documents(
            collection_name=collection,
            texts=chunks,
            metadatas=metadatas,
        )
        logger.info(
            "_ingest_document: done  file='%s'  chunks=%d  collection='%s'",
            filename, len(ids), collection,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("_ingest_document: ChromaDB upsert failed — %s", exc)


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Split *text* into overlapping character-level chunks."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # Keep last `overlap` chars worth of words for the next chunk
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current):
                if overlap_len + len(w) > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            current = overlap_words
            current_len = overlap_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


async def _get_profile_summary(student_id: str) -> dict[str, Any]:
    """Return a lightweight profile summary for the tutor chat response."""
    try:
        from rag.vector_store import get_vector_store

        vsm = get_vector_store()
        profile = vsm.get_student_profile(student_id)
        return {
            "total_sessions": profile["total_sessions"],
            "all_gaps": profile["all_gaps"][:10],  # top 10 gaps
        }
    except Exception:  # noqa: BLE001
        return {"total_sessions": 0, "all_gaps": []}


def _compute_progression(sessions: list[dict]) -> list[dict]:
    """Return the score evolution for the last 5 sessions (for charts)."""
    progression = []
    for s in sessions[-5:]:
        try:
            score = float(s.get("score", 0))
            max_score = float(s.get("max_score", 20) or 20)
            progression.append(
                {
                    "session": s.get("session", "?"),
                    "subject": s.get("subject", "?"),
                    "score": score,
                    "max_score": max_score,
                    "percentage": round(score / max_score * 100, 1) if max_score else 0,
                }
            )
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return progression


# ═════════════════════════════════════════════════════════════════════════════
# 6. GLOBAL EXCEPTION HANDLER
# ═════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )
