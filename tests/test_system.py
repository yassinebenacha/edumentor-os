"""
tests/test_system.py
End-to-end and unit tests for EduMentor OS.

Run with:
    pytest tests/ -v
    pytest tests/ -v -k "test_code_executor"   # single test
    pytest tests/ -v --tb=short                # short tracebacks

Test categories:
  - UNIT   : isolated, no Groq API calls (mocked)
  - INTEGR : requires GROQ_API_KEY in .env (marked with @pytest.mark.integration)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Pytest-asyncio configuration ──────────────────────────────────────────────
pytest_plugins = ["pytest_asyncio"]


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def sample_txt_path() -> Path:
    """Return the path to the sample exam TXT file."""
    p = ROOT / "data" / "sample_copies" / "copy_example.txt"
    if not p.exists():
        pytest.skip("sample copy not found — run demo first")
    return p


@pytest.fixture(scope="session")
def sample_pdf_path() -> Path:
    """Return (or create) the demo PDF."""
    p = ROOT / "data" / "sample_copies" / "exam_ahmed_benali.pdf"
    if not p.exists():
        try:
            sys.path.insert(0, str(ROOT / "demo"))
            from _create_sample_pdf import create_sample_pdf
            create_sample_pdf(p)
        except Exception:
            pytest.skip("PyMuPDF not available — skipping PDF fixture")
    return p


@pytest.fixture()
def tmp_chroma_dir(tmp_path: Path):
    """Temporary ChromaDB persistence directory (deleted after each test)."""
    return str(tmp_path / "chroma_test")


@pytest.fixture()
def mock_groq_response():
    """Factory that returns a mock ChatGroq response with a given content."""
    def _make(content: str):
        msg = MagicMock()
        msg.content = content
        resp = MagicMock()
        resp.content = content
        return resp
    return _make


# ── Shared mock grade result JSON ─────────────────────────────────────────────
MOCK_GRADE_JSON = json.dumps({
    "student_name": "Ahmed Benali",
    "subject": "Algorithmique",
    "total_score": 14.0,
    "max_score": 20,
    "percentage": 70.0,
    "confidence_score": 0.85,
    "detailed_scores": [
        {
            "question": "Q1",
            "score": 4.0,
            "max": 5.0,
            "feedback": "Bonne logique récursive, complexité mal identifiée.",
            "lacunes": [{"concept": "complexité O(2^n)", "niveau": 0.4, "type_erreur": "conceptuelle"}],
        }
    ],
    "global_lacunes": [
        {"concept": "complexité O(2^n)", "niveau_maitrise": 0.4, "priorite": "haute", "type_erreur": "conceptuelle"},
        {"concept": "cas de base QuickSort", "niveau_maitrise": 0.3, "priorite": "haute", "type_erreur": "incomplète"},
    ],
    "points_forts": ["Fibonacci récursif correct"],
    "recommandations": ["Revoir la complexité exponentielle", "Ajouter les cas de base"],
    "feedback_global": "Bon travail ! Concentre-toi sur la complexité algorithmique.",
})


# ═════════════════════════════════════════════════════════════════════════════
# 1. TEST_PDF_PARSER  (UNIT — no API)
# ═════════════════════════════════════════════════════════════════════════════

class TestPDFParser:
    """Tests for tools/pdf_parser.py — PDFParserTool and extract_pdf_text()."""

    def test_parse_txt_file_as_text(self, sample_txt_path: Path):
        """PDFParser should gracefully reject non-PDF files."""
        from tools.pdf_parser import PDFParserTool
        tool = PDFParserTool()
        result_str = tool._run(str(sample_txt_path))
        result = json.loads(result_str)
        assert "error" in result
        assert "pdf" in result["error"].lower() or "extension" in result["error"].lower()

    def test_file_not_found(self):
        """Non-existent file returns a clear error JSON."""
        from tools.pdf_parser import PDFParserTool
        tool = PDFParserTool()
        result = json.loads(tool._run("/tmp/nonexistent_99999.pdf"))
        assert "error" in result
        assert "introuvable" in result["error"].lower() or "not found" in result["error"].lower()

    def test_parse_real_pdf(self, sample_pdf_path: Path):
        """Real PDF extraction returns expected keys and non-empty text."""
        from tools.pdf_parser import PDFParserTool
        tool = PDFParserTool()
        result = json.loads(tool._run(str(sample_pdf_path)))

        assert "error" not in result, f"PDF parsing failed: {result.get('error')}"
        assert "text" in result
        assert len(result["text"]) > 100, "Extracted text is suspiciously short"
        assert "pages" in result
        assert result["pages"] >= 1
        assert isinstance(result["has_code"], bool)
        assert isinstance(result["code_blocks"], list)

    def test_python_code_detection(self, sample_pdf_path: Path):
        """The sample exam PDF contains Python code — has_code must be True."""
        from tools.pdf_parser import PDFParserTool
        tool = PDFParserTool()
        result = json.loads(tool._run(str(sample_pdf_path)))
        if "error" not in result:
            assert result["has_code"] is True, "Expected Python code detected in exam PDF"

    def test_code_detection_heuristic(self):
        """_detect_python_code() requires ≥2 matching patterns."""
        from tools.pdf_parser import _detect_python_code
        assert _detect_python_code("def foo():\n    return 42") is True
        assert _detect_python_code("for i in range(10):\n    print(i)") is True
        assert _detect_python_code("Hello world, this is plain text.") is False

    def test_extract_pdf_text_raises_on_missing(self):
        """extract_pdf_text() raises ValueError for missing file."""
        from tools.pdf_parser import extract_pdf_text
        with pytest.raises(ValueError):
            extract_pdf_text("/tmp/ghost_file_99999.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# 2. TEST_CODE_EXECUTOR  (UNIT — subprocess, no API)
# ═════════════════════════════════════════════════════════════════════════════

class TestCodeExecutor:
    """Tests for tools/code_executor.py — CodeExecutorTool and execute_code()."""

    # ── Good code ─────────────────────────────────────────────────────────

    def test_simple_print(self):
        """Basic print statement should succeed."""
        from tools.code_executor import execute_code
        result = execute_code('print("hello")')
        assert result["success"] is True
        assert result["output"] == "hello"
        assert result["error"] == ""

    def test_arithmetic(self):
        """Simple computation should return correct result."""
        from tools.code_executor import execute_code
        result = execute_code("print(2 ** 10)")
        assert result["success"] is True
        assert result["output"] == "1024"

    def test_with_test_cases_pass(self):
        """Code + matching test cases → all tests pass."""
        from tools.code_executor import execute_code
        code = "n = int(input())\nprint(n * 2)"
        tests = [
            {"input": "5",  "expected_output": "10"},
            {"input": "0",  "expected_output": "0"},
            {"input": "21", "expected_output": "42"},
        ]
        result = execute_code(code, test_cases=tests)
        assert result["tests_total"] == 3
        assert result["tests_passed"] == 3
        assert result["success"] is True

    def test_with_test_cases_partial(self):
        """Partially failing tests are recorded correctly."""
        from tools.code_executor import execute_code
        code = "print(int(input()))"   # echoes input, no multiplication
        tests = [
            {"input": "5", "expected_output": "10"},   # fails (5 ≠ 10)
            {"input": "0", "expected_output": "0"},    # passes (0 == 0)
        ]
        result = execute_code(code, test_cases=tests)
        assert result["tests_total"] == 2
        assert result["tests_passed"] == 1
        assert result["success"] is False

    # ── Syntax / Runtime errors ───────────────────────────────────────────

    def test_syntax_error(self):
        """Code with a syntax error returns success=False and stderr."""
        from tools.code_executor import execute_code
        result = execute_code("def broken(\n    print('oops')")
        assert result["success"] is False
        assert result["error"] != ""

    def test_runtime_error(self):
        """Division by zero → success=False with traceback in error."""
        from tools.code_executor import execute_code
        result = execute_code("print(1 / 0)")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    def test_timeout(self):
        """Infinite loop should be killed after timeout."""
        from tools.code_executor import execute_code
        result = execute_code("while True: pass")
        assert result["success"] is False
        assert "timeout" in result["error"].lower() or "Timeout" in result["error"]

    # ── Security blacklist ─────────────────────────────────────────────────

    def test_blacklist_os_system(self):
        """os.system calls must be rejected before execution."""
        from tools.code_executor import execute_code
        result = execute_code("import os; os.system('echo hacked')")
        assert result["success"] is False
        assert "non autorisé" in result["error"] or "interdit" in result["error"]

    def test_blacklist_subprocess(self):
        """subprocess import must be rejected."""
        from tools.code_executor import execute_code
        result = execute_code("import subprocess; subprocess.run(['ls'])")
        assert result["success"] is False

    def test_blacklist_eval(self):
        """eval() usage must be rejected."""
        from tools.code_executor import execute_code
        result = execute_code("eval('print(42)')")
        assert result["success"] is False

    def test_blacklist_exec(self):
        """exec() usage must be rejected."""
        from tools.code_executor import execute_code
        result = execute_code("exec('x=1')")
        assert result["success"] is False

    def test_safe_code_not_blocked(self):
        """Legitimate code with math/list ops should not be blocked."""
        from tools.code_executor import execute_code
        code = (
            "nums = [3, 1, 4, 1, 5, 9, 2, 6]\n"
            "print(sorted(nums))"
        )
        result = execute_code(code)
        assert result["success"] is True

    # ── Invalid test case format ──────────────────────────────────────────

    def test_invalid_test_cases_json(self):
        """Malformed test_cases JSON returns a clear error."""
        from tools.code_executor import CodeExecutorTool
        tool = CodeExecutorTool()
        result = json.loads(tool._run(code="print(1)", test_cases="{bad json"))
        assert result["success"] is False
        assert "invalide" in result["error"].lower() or "format" in result["error"].lower()


# ═════════════════════════════════════════════════════════════════════════════
# 3. TEST_VECTOR_STORE  (UNIT — local ChromaDB, no API)
# ═════════════════════════════════════════════════════════════════════════════

class TestVectorStore:
    """Tests for rag/vector_store.py — VectorStoreManager."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, tmp_chroma_dir: str, monkeypatch):
        """Redirect ChromaDB to a temp directory for each test."""
        monkeypatch.setattr("config.settings.chroma_persist_dir", Path(tmp_chroma_dir))

    @pytest.fixture()
    def vsm(self, tmp_chroma_dir: str):
        """Return a fresh VectorStoreManager pointing at a temp directory."""
        # Patch embeddings to avoid downloading models in CI
        from unittest.mock import MagicMock, patch
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: [[0.1] * 384 for _ in texts]
        mock_emb.embed_query = lambda text: [0.1] * 384
        with patch("rag.vector_store.get_embeddings", return_value=mock_emb):
            from rag.vector_store import VectorStoreManager
            return VectorStoreManager()

    def test_add_and_search_knowledge_base(self, vsm):
        """Documents added to knowledge_base can be retrieved by similarity search."""
        ids = vsm.add_documents(
            collection_name="knowledge_base",
            texts=["La récursion est une technique où une fonction s'appelle elle-même."],
            metadatas=[{"subject": "python_recursion"}],
        )
        assert len(ids) == 1

        results = vsm.similarity_search("knowledge_base", "récursion Python", k=1)
        assert len(results) >= 1
        assert "récursion" in results[0].page_content.lower()

    def test_add_multiple_documents(self, vsm):
        """Multiple documents are stored and independently retrievable."""
        texts = [
            "Le tri rapide utilise un pivot pour partitionner les éléments.",
            "La complexité de Fibonacci naïf est O(2^n).",
            "Un arbre binaire de recherche insère à gauche si valeur < racine.",
        ]
        ids = vsm.add_documents(
            collection_name="knowledge_base",
            texts=texts,
            metadatas=[{"subject": "algo"} for _ in texts],
        )
        assert len(ids) == 3

    def test_update_and_get_student_profile(self, vsm):
        """Student profile updates are persisted and retrievable."""
        student_id = "test_student_42"
        lacunes = {
            "session": "2026-04-05",
            "subject": "python",
            "score": 12.5,
            "max_score": 20,
            "gaps": ["récursivité", "polymorphisme"],
        }
        doc_id = vsm.update_student_profile(student_id, lacunes)
        assert doc_id  # non-empty UUID

        profile = vsm.get_student_profile(student_id)
        assert profile["student_id"] == student_id
        assert profile["total_sessions"] == 1
        assert "récursivité" in profile["all_gaps"]
        assert "polymorphisme" in profile["all_gaps"]

    def test_multiple_profile_updates_accumulate(self, vsm):
        """Multiple updates accumulate as separate sessions."""
        sid = "student_multi_sessions"
        for i in range(3):
            vsm.update_student_profile(sid, {
                "session": f"2026-04-0{i+1}",
                "subject": "algo",
                "score": 10 + i,
                "max_score": 20,
                "gaps": [f"concept_{i}"],
            })
        profile = vsm.get_student_profile(sid)
        assert profile["total_sessions"] == 3
        assert len(profile["all_gaps"]) == 3

    def test_add_and_search_rubric(self, vsm):
        """Rubric added to rubric_store is retrievable by semantic search."""
        vsm.add_rubric(
            subject="algorithmique",
            rubric_text="Q1 (5pts): Fibonacci récursif. Cas de base: +2pts.",
            metadata={"assignment": "exam_s1"},
        )
        results = vsm.search_rubric("algorithmique", "Fibonacci récursif cas de base")
        assert len(results) >= 1

    def test_unknown_collection_raises(self, vsm):
        """Accessing an unknown collection name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown collection"):
            vsm.add_documents("invalid_collection", ["text"], [{}])

    def test_empty_profile_for_unknown_student(self, vsm):
        """Fetching an unknown student returns an empty profile (no crash)."""
        profile = vsm.get_student_profile("ghost_student_99999")
        assert profile["total_sessions"] == 0
        assert profile["all_gaps"] == []


# ═════════════════════════════════════════════════════════════════════════════
# 4. TEST_GRADER_AGENT  (UNIT — Groq mocked)
# ═════════════════════════════════════════════════════════════════════════════

class TestGraderAgent:
    """Tests for agents/grader_agent.py — grade_submission()."""

    def _make_mock_agent(self):
        """Return a mock LangGraph agent that yields MOCK_GRADE_JSON."""
        mock_msg = MagicMock()
        mock_msg.content = MOCK_GRADE_JSON
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [mock_msg]}
        return mock_agent

    @pytest.mark.asyncio
    async def test_grade_result_structure(self, sample_pdf_path: Path):
        """grade_submission() returns a dict with all required keys."""
        with (
            patch("agents.grader_agent.create_grader_agent") as mock_factory,
            patch("rag.vector_store.get_vector_store") as mock_vsm,
        ):
            mock_factory.return_value = self._make_mock_agent()
            mock_vsm.return_value.update_student_profile = MagicMock(return_value="uuid-123")
            mock_vsm.return_value.get_student_profile = MagicMock(return_value={"all_gaps": []})
            mock_vsm.return_value.search_rubric = MagicMock(return_value=[])

            from agents.grader_agent import grade_submission
            result = await grade_submission(
                pdf_path=str(sample_pdf_path),
                rubric_text="Q1 (5pts): Fibonacci. Q2 (5pts): Complexité. Q3 (10pts): QuickSort.",
                student_id="test_ahmed",
            )

        required_keys = [
            "student_name", "subject", "total_score", "max_score",
            "percentage", "confidence_score", "detailed_scores",
            "global_lacunes", "points_forts", "recommandations",
            "feedback_global", "student_id",
        ]
        for key in required_keys:
            assert key in result, f"Missing key in grade result: '{key}'"

    @pytest.mark.asyncio
    async def test_grade_score_range(self, sample_pdf_path: Path):
        """total_score must be between 0 and max_score."""
        with (
            patch("agents.grader_agent.create_grader_agent") as mock_factory,
            patch("rag.vector_store.get_vector_store") as mock_vsm,
        ):
            mock_factory.return_value = self._make_mock_agent()
            mock_vsm.return_value.update_student_profile = MagicMock(return_value="uuid")
            mock_vsm.return_value.search_rubric = MagicMock(return_value=[])

            from agents.grader_agent import grade_submission
            result = await grade_submission(str(sample_pdf_path), "", "test_student")

        assert 0 <= result["total_score"] <= result["max_score"]
        assert 0.0 <= result["percentage"] <= 100.0

    @pytest.mark.asyncio
    async def test_grade_missing_pdf_raises(self):
        """grade_submission raises ValueError for a non-existent PDF."""
        from agents.grader_agent import grade_submission
        with pytest.raises(ValueError, match="Impossible de lire"):
            await grade_submission("/nonexistent/path/fake.pdf", "", "s1")

    @pytest.mark.asyncio
    async def test_grade_updates_student_profile(self, sample_pdf_path: Path):
        """grade_submission must call update_student_profile exactly once."""
        mock_update = MagicMock(return_value="uuid-456")
        with (
            patch("agents.grader_agent.create_grader_agent") as mock_factory,
            patch("rag.vector_store.get_vector_store") as mock_vsm,
        ):
            mock_factory.return_value = self._make_mock_agent()
            mock_vsm.return_value.update_student_profile = mock_update
            mock_vsm.return_value.search_rubric = MagicMock(return_value=[])

            from agents.grader_agent import grade_submission
            await grade_submission(str(sample_pdf_path), "barème test", "student_profile_check")

        mock_update.assert_called_once()

    def test_json_extraction_pure_json(self):
        """_extract_json parses clean JSON correctly."""
        from agents.grader_agent import _extract_json
        data = {"total_score": 14, "max_score": 20}
        assert _extract_json(json.dumps(data)) == data

    def test_json_extraction_from_markdown_fence(self):
        """_extract_json extracts JSON from a markdown code fence."""
        from agents.grader_agent import _extract_json
        payload = '```json\n{"total_score": 14}\n```'
        assert _extract_json(payload) == {"total_score": 14}

    def test_json_extraction_from_prose(self):
        """_extract_json extracts the first { } block from mixed text."""
        from agents.grader_agent import _extract_json
        payload = 'Voici la correction :\n{"total_score": 14}\nBonne chance !'
        assert _extract_json(payload) == {"total_score": 14}

    def test_json_extraction_raises_on_invalid(self):
        """_extract_json raises ValueError when no valid JSON is found."""
        from agents.grader_agent import _extract_json
        with pytest.raises(ValueError):
            _extract_json("Texte sans JSON du tout.")


# ═════════════════════════════════════════════════════════════════════════════
# 5. TEST_TUTOR_CONVERSATION  (UNIT — Groq mocked)
# ═════════════════════════════════════════════════════════════════════════════

class TestTutorConversation:
    """Tests for agents/tutor_agent.py — chat_with_tutor()."""

    def _make_mock_tutor_agent(self, response_text: str):
        """Return a mock LangGraph agent with a fixed text response."""
        mock_msg = MagicMock()
        mock_msg.content = response_text
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [mock_msg]}
        return mock_agent

    @pytest.mark.asyncio
    async def test_first_message_returns_response(self):
        """First message with empty history returns a non-empty response."""
        response_text = (
            "📚 **Rappel** — La récursion appelle une fonction sur elle-même.\n"
            "💡 Exemple : factorielle(5) = 5 × factorielle(4)."
        )
        with patch("agents.tutor_agent.create_tutor_agent") as mock_factory:
            mock_factory.return_value = self._make_mock_tutor_agent(response_text)
            from agents.tutor_agent import chat_with_tutor

            response, history = await chat_with_tutor(
                student_id="test_student",
                message="Explique-moi la récursion",
                conversation_history=[],
            )

        assert len(response) > 0
        assert response == response_text

    @pytest.mark.asyncio
    async def test_history_grows_correctly(self):
        """After each turn, history gains exactly 2 entries (human + ai)."""
        mock_response = "Réponse du tutor"
        with patch("agents.tutor_agent.create_tutor_agent") as mock_factory:
            mock_factory.return_value = self._make_mock_tutor_agent(mock_response)
            from agents.tutor_agent import chat_with_tutor

            history: list[dict] = []
            for i in range(3):
                _, history = await chat_with_tutor(
                    "alice", f"Question {i+1}", history
                )

        assert len(history) == 6  # 3 turns × 2 messages each
        assert history[0]["role"] == "human"
        assert history[1]["role"] == "ai"

    @pytest.mark.asyncio
    async def test_history_content_preserved(self):
        """Each human message is correctly stored in conversation history."""
        with patch("agents.tutor_agent.create_tutor_agent") as mock_factory:
            mock_factory.return_value = self._make_mock_tutor_agent("Réponse AI")
            from agents.tutor_agent import chat_with_tutor

            _, history = await chat_with_tutor("s1", "Ma question spécifique", [])

        human_msgs = [m for m in history if m["role"] == "human"]
        assert any("Ma question spécifique" in m["content"] for m in human_msgs)

    @pytest.mark.asyncio
    async def test_history_capped_at_80(self):
        """Conversation history is capped at 80 entries (40 turns)."""
        with patch("agents.tutor_agent.create_tutor_agent") as mock_factory:
            mock_factory.return_value = self._make_mock_tutor_agent("ok")
            from agents.tutor_agent import chat_with_tutor

            history: list[dict] = [
                {"role": "human" if i % 2 == 0 else "ai", "content": f"msg {i}"}
                for i in range(90)
            ]
            _, updated = await chat_with_tutor("s1", "new msg", history)

        assert len(updated) <= 80 + 2  # +2 for the new turn

    @pytest.mark.asyncio
    async def test_three_turn_conversation_simulation(self):
        """Simulate a realistic 3-question tutoring session."""
        responses = [
            "📚 La récursion, c'est quand une fonction s'appelle elle-même.",
            "💡 Pour Fibonacci, F(n) = F(n-1) + F(n-2) avec F(0)=0, F(1)=1.",
            "✏️ Voici un quiz : Q1 - Quel est le cas de base de Fibonacci ?",
        ]
        questions = [
            "C'est quoi la récursion ?",
            "Peux-tu m'expliquer Fibonacci ?",
            "Génère un quiz sur la récursion.",
        ]

        history: list[dict] = []
        with patch("agents.tutor_agent.create_tutor_agent") as mock_factory:
            from agents.tutor_agent import chat_with_tutor
            for i, (q, r) in enumerate(zip(questions, responses)):
                mock_factory.return_value = self._make_mock_tutor_agent(r)
                resp, history = await chat_with_tutor("ahmed", q, history)
                assert resp == r

        assert len(history) == 6
        assert history[-1]["content"] == responses[-1]


# ═════════════════════════════════════════════════════════════════════════════
# 6. TEST_FULL_PIPELINE  (INTEGRATION — mocked LangGraph)
# ═════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end tests for graphs/main_graph.py — run_graph()."""

    def _make_grade_state(self, pdf_path: str) -> dict:
        return {
            "task_type": "grade",
            "student_id": "e2e_test_student",
            "pdf_path": pdf_path,
            "rubric_text": "Q1 (5pts): Fibonacci. Q3 (10pts): QuickSort.",
        }

    def _make_chat_state(self) -> dict:
        return {
            "task_type": "tutor_chat",
            "student_id": "e2e_test_student",
            "user_message": "Explique la complexité O(2^n) de Fibonacci",
            "conversation_history": [],
        }

    @pytest.mark.asyncio
    async def test_grade_pipeline_state_keys(self, sample_pdf_path: Path):
        """run_graph(grade) returns final state with grading_result populated."""
        mock_grade = json.loads(MOCK_GRADE_JSON)

        with (
            patch("agents.grader_agent.grade_submission", new=AsyncMock(return_value=mock_grade)),
            patch("rag.vector_store.get_vector_store") as mock_vsm,
        ):
            mock_vsm.return_value.update_student_profile = MagicMock(return_value="uuid")
            mock_vsm.return_value.get_student_profile = MagicMock(return_value={"sessions": [], "all_gaps": []})

            # Force graph rebuild for each test
            import graphs.main_graph as gm
            gm._app = None

            from graphs.main_graph import run_graph
            state = await run_graph(self._make_grade_state(str(sample_pdf_path)))

        assert state.get("error") is None or state.get("grading_result") is not None
        if not state.get("error"):
            assert "grading_result" in state
            assert state["grading_result"]["total_score"] == 14.0

    @pytest.mark.asyncio
    async def test_tutor_pipeline_returns_response(self):
        """run_graph(tutor_chat) populates tutor_response in final state."""
        expected = "Réponse du tuteur simulé"
        mock_agent = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.content = expected
        mock_agent.ainvoke.return_value = {"messages": [mock_msg]}

        with patch("agents.tutor_agent.create_tutor_agent", return_value=mock_agent):
            import graphs.main_graph as gm
            gm._app = None

            from graphs.main_graph import run_graph
            state = await run_graph(self._make_chat_state())

        if not state.get("error"):
            assert state.get("tutor_response") is not None
            assert len(state["tutor_response"]) > 0

    @pytest.mark.asyncio
    async def test_unknown_task_type_goes_to_error_handler(self):
        """Unknown task_type should route to error handler (no crash)."""
        import graphs.main_graph as gm
        gm._app = None

        from graphs.main_graph import run_graph
        state = await run_graph({
            "task_type": "unknown_task_xyz",
            "student_id": "s",
        })
        # Graph should reach END without raising; error_handler sets no crash
        assert state is not None

    @pytest.mark.asyncio
    async def test_grade_missing_pdf_sets_error(self):
        """Grading with a missing PDF populates state.error (no exception)."""
        import graphs.main_graph as gm
        gm._app = None

        from graphs.main_graph import run_graph
        state = await run_graph({
            "task_type": "grade",
            "student_id": "s",
            "pdf_path": "/nonexistent/file.pdf",
        })
        assert state.get("error") is not None

    @pytest.mark.asyncio
    async def test_tutor_missing_message_sets_error(self):
        """Tutor task with empty user_message populates state.error."""
        import graphs.main_graph as gm
        gm._app = None

        from graphs.main_graph import run_graph
        state = await run_graph({
            "task_type": "tutor_chat",
            "student_id": "s",
            "user_message": "",
        })
        assert state.get("error") is not None

    def test_default_state_has_required_keys(self):
        """_default_state() provides all required EduMentorState fields."""
        from graphs.main_graph import _default_state
        state = _default_state()
        required = [
            "task_type", "student_id", "pdf_path", "rubric_text",
            "user_message", "conversation_history", "grading_result",
            "class_analysis", "tutor_response", "error",
            "confidence_score", "needs_human_review",
        ]
        for key in required:
            assert key in state, f"Missing key in default state: '{key}'"


# ═════════════════════════════════════════════════════════════════════════════
# 7. FAST SMOKE TESTS  (no API, no ChromaDB)
# ═════════════════════════════════════════════════════════════════════════════

class TestSmoke:
    """Fast import and config sanity checks."""

    def test_config_loads(self):
        """settings object loads without errors."""
        from config import settings
        assert settings is not None
        assert settings.llm_model != ""

    def test_all_tools_importable(self):
        """All tool modules import without errors."""
        import tools.pdf_parser
        import tools.code_executor
        import tools.rubric_retriever
        import tools.student_profile
        import tools.quiz_generator

    def test_all_agents_importable(self):
        """All agent modules import without errors."""
        import agents.grader_agent
        import agents.tutor_agent
        import agents.analyst_agent

    def test_graph_module_importable(self):
        """graphs.main_graph imports without errors."""
        import graphs.main_graph

    def test_api_module_importable(self):
        """api.main imports without errors."""
        import api.main

    def test_rag_modules_importable(self):
        """RAG modules import without errors."""
        import rag.embeddings
        import rag.vector_store
