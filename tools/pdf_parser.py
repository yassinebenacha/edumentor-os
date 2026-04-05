"""
tools/pdf_parser.py
LangChain tool for parsing student exam submission PDFs.

Uses PyMuPDF (fitz) for fast, reliable text extraction.
Detects Python code blocks and attempts to identify student name / subject
from common header patterns found in academic copies.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Type

import fitz  # PyMuPDF
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# Python code indicators (order matters — most decisive first)
_CODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(def |class |import |from .+ import |if __name__)", re.MULTILINE),
    re.compile(r"^\s*(for |while |with |try:|except |finally:)", re.MULTILINE),
    re.compile(r"^\s*(return |yield |raise |assert |pass|break|continue)\b", re.MULTILINE),
    re.compile(r"(print\s*\(|input\s*\(|range\s*\(|len\s*\()", re.MULTILINE),
    re.compile(r"#.*$", re.MULTILINE),  # Python comments
]

# Heuristics to locate inline code blocks (indented or fenced sections)
_CODE_BLOCK_RE = re.compile(
    r"(?:```[\w]*\n(.*?)```|(?:(?:^    .+\n)+))",
    re.DOTALL | re.MULTILINE,
)

# Common French / international exam header patterns
_STUDENT_NAME_RE = re.compile(
    r"(?:nom\s*[:\-]?\s*|name\s*[:\-]?\s*|étudiant\s*[:\-]?\s*)([A-ZÀÂÄÉÈÊËÎÏÔÛÙÜ][A-Za-zÀ-ÿ\-' ]{1,40})",
    re.IGNORECASE,
)
_SUBJECT_RE = re.compile(
    r"(?:matière\s*[:\-]?\s*|module\s*[:\-]?\s*|subject\s*[:\-]?\s*|cours\s*[:\-]?\s*)([A-Za-zÀ-ÿ0-9 \-]{2,50})",
    re.IGNORECASE,
)


# ── Input schema ──────────────────────────────────────────────────────────────

class PDFParserInput(BaseModel):
    file_path: str = Field(description="Absolute or relative path to the student's PDF exam copy")


# ── Tool ──────────────────────────────────────────────────────────────────────

class PDFParserTool(BaseTool):
    """
    LangChain tool that extracts structured information from a student exam PDF.

    Returns a JSON string with:
      • full text
      • page count
      • code detection flag
      • extracted code blocks
      • detected student name (if present in header)
      • detected subject (if present in header)
    """

    name: str = "pdf_parser"
    description: str = (
        "Extrait le texte d'une copie d'examen PDF. "
        "Input: chemin du fichier PDF. "
        "Retourne un JSON avec le texte, le nombre de pages, les blocs de code Python détectés, "
        "le nom de l'étudiant et la matière si présents dans le document."
    )
    args_schema: Type[BaseModel] = PDFParserInput

    def _run(self, file_path: str) -> str:  # noqa: C901
        """
        Extract and analyse the content of *file_path*.

        Parameters
        ----------
        file_path : str
            Path to the PDF file (absolute or relative to CWD).

        Returns
        -------
        str
            JSON-encoded dict with keys:
            text, pages, has_code, code_blocks, student_name, subject.
            On error, returns a JSON dict with an "error" key.
        """
        path = Path(file_path)

        # ── Existence / type checks ───────────────────────────────────────
        if not path.exists():
            return _error_json(f"Fichier introuvable : '{file_path}'")

        if path.suffix.lower() != ".pdf":
            return _error_json(
                f"Le fichier '{path.name}' n'est pas un PDF (extension: '{path.suffix}')."
            )

        # ── PyMuPDF extraction ────────────────────────────────────────────
        try:
            doc = fitz.open(str(path))
        except fitz.FileDataError as exc:
            return _error_json(f"PDF corrompu ou illisible : {exc}")
        except Exception as exc:  # noqa: BLE001
            return _error_json(f"Erreur inattendue lors de l'ouverture du PDF : {exc}")

        try:
            page_texts: list[str] = []
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    page_texts.append(page.get_text("text"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Page %d illisible : %s", page_num + 1, exc)
                    page_texts.append("")
        finally:
            doc.close()

        full_text = "\n".join(page_texts).strip()

        if not full_text:
            return _error_json(
                "Le PDF ne contient pas de texte extractible "
                "(PDF scanné ou composé uniquement d'images)."
            )

        # ── Code detection ────────────────────────────────────────────────
        has_code = _detect_python_code(full_text)
        code_blocks = _extract_code_blocks(full_text) if has_code else []

        # ── Metadata heuristics ───────────────────────────────────────────
        # Only scan first ~2 000 chars (exam headers are at the top)
        header = full_text[:2000]
        student_name = _first_match(_STUDENT_NAME_RE, header)
        subject = _first_match(_SUBJECT_RE, header)

        result: dict[str, Any] = {
            "text": full_text,
            "pages": len(page_texts),
            "has_code": has_code,
            "code_blocks": code_blocks,
            "student_name": student_name,
            "subject": subject,
        }

        logger.info(
            "pdf_parser: file='%s'  pages=%d  has_code=%s  student='%s'  subject='%s'",
            path.name,
            len(page_texts),
            has_code,
            student_name,
            subject,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    # LangChain requires _arun — delegates to sync version
    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)


# ── Private helpers ───────────────────────────────────────────────────────────

def _error_json(message: str) -> str:
    """Return a standardised error payload as a JSON string."""
    logger.error("pdf_parser error: %s", message)
    return json.dumps({"error": message}, ensure_ascii=False)


def _detect_python_code(text: str) -> bool:
    """
    Return True if *text* contains recognisable Python code patterns.
    Requires at least 2 distinct pattern matches to avoid false positives.
    """
    hits = sum(1 for pat in _CODE_PATTERNS if pat.search(text))
    return hits >= 2


def _extract_code_blocks(text: str) -> list[str]:
    """
    Extract distinct code sections from *text*.

    Strategy (in order):
    1. Explicitly fenced blocks (``` … ```)
    2. Consistently indented blocks (4+ spaces, 3+ consecutive lines)
    3. Fallback: lines that match strong code patterns
    """
    blocks: list[str] = []

    # 1. Fenced markdown blocks
    fenced = re.findall(r"```[\w]*\n(.*?)```", text, re.DOTALL)
    blocks.extend(b.strip() for b in fenced if b.strip())

    # 2. Indented blocks (≥ 3 consecutive lines with 4-space indent)
    if not blocks:
        indented_block: list[str] = []
        for line in text.splitlines():
            if re.match(r"^    .+", line):
                indented_block.append(line)
            else:
                if len(indented_block) >= 3:
                    blocks.append("\n".join(indented_block).strip())
                indented_block = []
        if len(indented_block) >= 3:
            blocks.append("\n".join(indented_block).strip())

    # 3. Fallback: grab lines matching strong Python patterns
    if not blocks:
        strong = re.compile(
            r"^\s*(def |class |import |from .+ import |if __name__)", re.MULTILINE
        )
        matched_lines = [l.strip() for l in text.splitlines() if strong.match(l)]
        if matched_lines:
            blocks.append("\n".join(matched_lines))

    return blocks


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    """Return the first capture group of *pattern* in *text*, stripped, or None."""
    m = pattern.search(text)
    if m:
        return m.group(1).strip() or None
    return None


# ── Public utility function ───────────────────────────────────────────────────

def extract_pdf_text(path: str | Path) -> dict[str, Any]:
    """
    Convenience function for direct (non-agent) usage.

    Parameters
    ----------
    path : str | Path
        Path to the PDF file.

    Returns
    -------
    dict
        Same structure as PDFParserTool._run(), already decoded from JSON.

    Raises
    ------
    ValueError
        If the PDF cannot be parsed (error key present in result).

    Example
    -------
    >>> result = extract_pdf_text("data/sample_copies/exam_alice.pdf")
    >>> print(result["pages"], result["has_code"])
    """
    tool = PDFParserTool()
    raw = tool._run(str(path))
    result: dict[str, Any] = json.loads(raw)

    if "error" in result:
        raise ValueError(result["error"])

    return result
