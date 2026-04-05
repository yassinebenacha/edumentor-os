"""
data/ingest_data.py
Automatic ingestion of all data/ files into ChromaDB collections.

Usage:
    conda activate edumentor-os
    python data/ingest_data.py

What it does:
  • data/knowledge_base/*.txt|*.pdf  → collection "knowledge_base"
  • data/rubrics/*.txt|*.pdf|*.json  → collection "rubric_store"
  • data/sample_copies/              → skipped (student submissions, not ingested)

Supports: .txt, .pdf (via PyMuPDF), .json, .md, .yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Add project root to sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Rich for pretty output ────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )
    from rich.table import Table
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
    console = Console()

from config import settings


# ── Directory → collection mapping ────────────────────────────────────────────

INGESTION_PLAN: list[dict] = [
    {
        "directory": settings.knowledge_base_dir,
        "collection": "knowledge_base",
        "extensions": {".txt", ".pdf", ".md"},
        "label": "📚 Knowledge Base",
        "doc_type": "course",
    },
    {
        "directory": settings.rubrics_dir,
        "collection": "rubric_store",
        "extensions": {".txt", ".pdf", ".json", ".yaml", ".yml"},
        "label": "📋 Rubric Store",
        "doc_type": "rubric",
    },
]

# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text(path: Path) -> str | None:
    """Extract plain text from a file. Returns None on failure."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(path))
            text = "\n".join(page.get_text("text") for page in doc)
            doc.close()
            return text.strip() or None
        except Exception as exc:
            console.print(f"    [red]⚠ PDF error ({path.name}): {exc}[/red]")
            return None

    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")

    # .txt / .md / .yaml / .yml
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        console.print(f"    [red]⚠ Read error ({path.name}): {exc}[/red]")
        return None


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[str]:
    """Split *text* into overlapping word-level chunks."""
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        # slide forward by (chunk_size - overlap) words
        start += max(1, chunk_size - overlap)

    return chunks


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_all(dry_run: bool = False) -> dict[str, int]:
    """
    Ingest all configured directories into ChromaDB.

    Parameters
    ----------
    dry_run : bool
        If True, scan files and print what would be ingested without
        touching ChromaDB.

    Returns
    -------
    dict
        {collection_name: number_of_chunks_ingested}
    """
    console.print()
    console.print("╔══════════════════════════════════════════════════╗")
    console.print("║   EduMentor OS — Knowledge Base Ingestion        ║")
    console.print("╚══════════════════════════════════════════════════╝")
    console.print()

    if dry_run:
        console.print("[yellow]  DRY RUN — no data will be written to ChromaDB[/yellow]\n")

    # Initialise vector store once
    vsm = None
    if not dry_run:
        try:
            from rag.vector_store import get_vector_store
            vsm = get_vector_store()
            console.print("  ✅ ChromaDB connection established\n")
        except Exception as exc:
            console.print(f"  [red]❌ ChromaDB connection failed: {exc}[/red]")
            return {}

    summary: dict[str, int] = {}
    total_chunks_global = 0

    for plan in INGESTION_PLAN:
        directory: Path = Path(plan["directory"])
        collection: str = plan["collection"]
        extensions: set  = plan["extensions"]
        label: str       = plan["label"]
        doc_type: str    = plan["doc_type"]

        console.print(f"  {label}  →  collection: [cyan]{collection}[/cyan]")
        console.print(f"  Directory: [dim]{directory}[/dim]")

        if not directory.exists():
            console.print("    [yellow]⚠ Directory not found, skipping.[/yellow]\n")
            summary[collection] = 0
            continue

        # Collect eligible files
        files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        if not files:
            console.print("    [yellow]⚠ No eligible files found.[/yellow]\n")
            summary[collection] = 0
            continue

        console.print(f"    Found [bold]{len(files)}[/bold] file(s) to process")

        collection_chunks = 0

        for file_path in files:
            console.print(f"    📄 {file_path.name}", end="")

            # Extract text
            raw_text = extract_text(file_path)
            if not raw_text:
                console.print(" → [red]empty / unreadable[/red]")
                continue

            # Chunk text
            chunks = chunk_text(raw_text)
            console.print(f" → [green]{len(chunks)} chunk(s)[/green]", end="")

            if dry_run:
                console.print(f" [dim](dry run, skipped)[/dim]")
                collection_chunks += len(chunks)
                continue

            # Build metadata
            metadatas = [
                {
                    "filename": file_path.name,
                    "subject": _infer_subject(file_path.name, raw_text),
                    "doc_type": doc_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                for i in range(len(chunks))
            ]

            # Upsert into ChromaDB
            try:
                ids = vsm.add_documents(
                    collection_name=collection,
                    texts=chunks,
                    metadatas=metadatas,
                )
                console.print(f" ✅ ({len(ids)} ids)")
                collection_chunks += len(ids)
            except Exception as exc:
                console.print(f" [red]❌ ChromaDB error: {exc}[/red]")

        summary[collection] = collection_chunks
        total_chunks_global += collection_chunks
        console.print(
            f"\n    [bold green]✅ {label}: ingested {collection_chunks} chunk(s)[/bold green]\n"
        )

    # ── Summary table ─────────────────────────────────────────────────────
    if HAS_RICH:
        table = Table(title="📊 Ingestion Summary", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Collection", style="bold")
        table.add_column("Chunks ingested", justify="right", style="green")
    else:
        print("\n=== Ingestion Summary ===")

    for col, count in summary.items():
        if HAS_RICH:
            table.add_row(col, str(count))
        else:
            print(f"  {col}: {count} chunks")

    if HAS_RICH:
        console.print(table)

    console.print(f"\n  [bold]Total: {total_chunks_global} chunks across {len(summary)} collections[/bold]")
    console.print(
        "\n  [dim]Run `uvicorn api.main:app --reload` to start the API.[/dim]\n"
    )

    return summary


# ── Subject inference heuristic ───────────────────────────────────────────────

def _infer_subject(filename: str, text: str) -> str:
    """Guess a subject tag from filename and content keywords."""
    combined = (filename + " " + text[:500]).lower()
    if any(k in combined for k in ["récursion", "recursion", "fibonacci", "factoriel"]):
        return "python_recursion"
    if any(k in combined for k in ["tri", "sort", "quicksort", "complexité", "algorithme"]):
        return "algorithmique"
    if any(k in combined for k in ["python", "code", "programmation", "classe", "objet"]):
        return "python_programming"
    if any(k in combined for k in ["math", "algèbre", "calcul", "intégral"]):
        return "mathematics"
    if any(k in combined for k in ["barème", "bareme", "rubric", "correction", "points"]):
        return "evaluation"
    return "general"


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest data/ files into EduMentor ChromaDB collections."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files without writing to ChromaDB.",
    )
    args = parser.parse_args()

    results = ingest_all(dry_run=args.dry_run)
    total = sum(results.values())
    sys.exit(0 if total >= 0 else 1)
