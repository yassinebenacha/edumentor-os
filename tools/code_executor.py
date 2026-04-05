"""
tools/code_executor.py
Secure Python code execution tool for EduMentor-OS.

Runs student-submitted code in an isolated subprocess with:
  • Hard timeout (default 10 s)
  • Static blacklist sanitisation before execution
  • Per-test-case execution and result comparison
  • Structured JSON output consumed by GraderAgent
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import time
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

# ── Security blacklist ────────────────────────────────────────────────────────

_BLACKLIST: list[str] = [
    "os.system",
    "os.popen",
    "os.remove",
    "os.rmdir",
    "os.unlink",
    "subprocess",
    "shutil",
    "__import__",
    "importlib",
    "open(",            # file writes / reads
    "eval(",
    "exec(",
    "compile(",
    "globals(",
    "locals(",
    "vars(",
    "getattr(",
    "setattr(",
    "delattr(",
    "socket",
    "requests",
    "urllib",
    "httpx",
    "ctypes",
    "multiprocessing",
    "threading",
    "pty",
    "pickle",
    "marshal",
]

# ── Input schema ──────────────────────────────────────────────────────────────

class CodeExecutorInput(BaseModel):
    code: str = Field(description="Code Python à exécuter")
    test_cases: str = Field(
        default="",
        description=(
            "Liste JSON de cas de test. Chaque entrée est un dict avec les clés "
            "'input' (str, stdin à injecter) et 'expected_output' (str). "
            "Exemple: [{\"input\": \"5\", \"expected_output\": \"120\"}]"
        ),
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

class CodeExecutorTool(BaseTool):
    """
    LangChain tool that safely executes student Python code in a subprocess.

    For each test case the tool:
      1. Prepends the student code to a minimal harness
      2. Launches a fresh Python subprocess with a hard timeout
      3. Compares stdout (stripped) against expected_output (stripped)

    The tool never modifies the host file system and does not share
    memory with the student's process.
    """

    name: str = "code_executor"
    description: str = (
        "Exécute du code Python de façon sécurisée et retourne le résultat. "
        "Utile pour corriger les exercices de programmation. "
        "Input: code Python (str) et optionnellement une liste JSON de cas de test."
    )
    args_schema: Type[BaseModel] = CodeExecutorInput

    def _run(self, code: str, test_cases: str = "") -> str:  # noqa: C901
        """
        Execute *code*, optionally against *test_cases*.

        Parameters
        ----------
        code : str
            Python source submitted by the student.
        test_cases : str
            JSON-encoded list of ``{"input": str, "expected_output": str}``
            dicts.  Empty string means "just run the code, no assertions."

        Returns
        -------
        str
            JSON-encoded result dict (see module docstring for schema).
        """
        # 1. Static security check
        violation = _find_violation(code)
        if violation:
            logger.warning("code_executor: blacklisted token '%s' detected", violation)
            return _result_json(
                success=False,
                output="",
                error=f"Code non autorisé : l'utilisation de '{violation}' est interdite.",
                execution_time=0.0,
            )

        # 2. Parse test cases
        parsed_tests: list[dict[str, str]] = []
        if test_cases.strip():
            try:
                parsed_tests = json.loads(test_cases)
                if not isinstance(parsed_tests, list):
                    raise ValueError("test_cases must be a JSON list")
            except (json.JSONDecodeError, ValueError) as exc:
                return _result_json(
                    success=False,
                    output="",
                    error=f"Format test_cases invalide : {exc}",
                    execution_time=0.0,
                )

        # 3. No test cases → plain execution
        if not parsed_tests:
            return _run_plain(code)

        # 4. Execute each test case
        return _run_with_tests(code, parsed_tests)

    async def _arun(self, code: str, test_cases: str = "") -> str:
        return self._run(code, test_cases)


# ── Execution helpers ─────────────────────────────────────────────────────────

def _find_violation(code: str) -> str | None:
    """Return the first blacklisted token found in *code*, or None."""
    # Strip comments before checking to avoid false positives
    code_no_comments = "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("#")
    )
    for token in _BLACKLIST:
        if token in code_no_comments:
            return token
    return None


def _run_subprocess(
    code: str,
    stdin_input: str = "",
    timeout: int | None = None,
) -> tuple[bool, str, str, float]:
    """
    Run *code* in a fresh Python subprocess.

    Returns
    -------
    (success, stdout, stderr, elapsed_seconds)
    """
    if timeout is None:
        timeout = settings.code_exec_timeout

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        success = proc.returncode == 0
        return success, proc.stdout, proc.stderr, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        logger.warning("code_executor: timeout after %.1f s", elapsed)
        return (
            False,
            "",
            f"⏱ Timeout : le code a dépassé la limite de {timeout} secondes.",
            elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return False, "", f"Erreur subprocess inattendue : {exc}", elapsed


def _run_plain(code: str) -> str:
    """Execute *code* without test cases and return a JSON result string."""
    success, stdout, stderr, elapsed = _run_subprocess(code)
    return _result_json(
        success=success,
        output=stdout.strip(),
        error=stderr.strip() if not success else "",
        execution_time=round(elapsed, 4),
    )


def _run_with_tests(
    code: str,
    test_cases: list[dict[str, str]],
) -> str:
    """
    Run *code* against each entry in *test_cases* and return aggregated results.

    Each test dict must have:
      ``input``           — text sent to stdin (may be empty)
      ``expected_output`` — expected stdout (stripped, compared exactly)
    """
    details: list[dict[str, Any]] = []
    tests_passed = 0
    total_elapsed = 0.0
    any_fatal = False
    fatal_error = ""

    for idx, tc in enumerate(test_cases):
        stdin_input: str = tc.get("input", "")
        expected: str = tc.get("expected_output", "").strip()

        # Wrap code so input() calls receive the test's stdin
        success, stdout, stderr, elapsed = _run_subprocess(
            code, stdin_input=stdin_input
        )
        total_elapsed += elapsed

        actual_output = stdout.strip()
        passed = success and (actual_output == expected)

        if passed:
            tests_passed += 1

        # Record a fatal runtime error on the first occurrence
        if not success and stderr and not any_fatal:
            any_fatal = True
            fatal_error = stderr.strip()

        details.append(
            {
                "test_index": idx + 1,
                "input": stdin_input,
                "expected_output": expected,
                "actual_output": actual_output,
                "passed": passed,
                "error": stderr.strip() if (stderr and not success) else "",
                "execution_time": round(elapsed, 4),
            }
        )

    overall_success = tests_passed == len(test_cases)

    return json.dumps(
        {
            "success": overall_success,
            "output": details[-1]["actual_output"] if details else "",
            "error": fatal_error,
            "execution_time": round(total_elapsed, 4),
            "tests_passed": tests_passed,
            "tests_total": len(test_cases),
            "test_details": details,
        },
        ensure_ascii=False,
        indent=2,
    )


def _result_json(
    success: bool,
    output: str,
    error: str,
    execution_time: float,
    tests_passed: int = 0,
    tests_total: int = 0,
    test_details: list[dict[str, Any]] | None = None,
) -> str:
    """Serialise a uniform result payload to a JSON string."""
    return json.dumps(
        {
            "success": success,
            "output": output,
            "error": error,
            "execution_time": execution_time,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "test_details": test_details or [],
        },
        ensure_ascii=False,
        indent=2,
    )


# ── Public utility ────────────────────────────────────────────────────────────

def execute_code(
    code: str,
    test_cases: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper for direct (non-agent) usage.

    Parameters
    ----------
    code : str
        Python source to execute.
    test_cases : list[dict], optional
        Each dict: ``{"input": str, "expected_output": str}``.

    Returns
    -------
    dict
        Same structure as CodeExecutorTool output, already decoded.

    Example
    -------
    >>> result = execute_code(
    ...     "n = int(input())\\nprint(n * 2)",
    ...     test_cases=[{"input": "5", "expected_output": "10"}],
    ... )
    >>> print(result["tests_passed"], "/", result["tests_total"])
    1 / 1
    """
    tool = CodeExecutorTool()
    tc_str = json.dumps(test_cases) if test_cases else ""
    raw = tool._run(code=code, test_cases=tc_str)
    return json.loads(raw)
