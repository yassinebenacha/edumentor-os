"""
demo/run_demo.py
EduMentor OS — Démonstration complète en terminal.

Usage:
    conda activate edumentor-os
    cd edumentor-os
    python demo/run_demo.py

Requires a valid GROQ_API_KEY in .env
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Rich imports ──────────────────────────────────────────────────────────────
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console(highlight=True)

# ── Simulated class profiles (fallback if real API fails) ────────────────────
MOCK_GRADE_RESULT = {
    "student_name": "Ahmed Benali",
    "subject": "Programmation Python",
    "total_score": 13.5,
    "max_score": 20,
    "percentage": 67.5,
    "confidence_score": 0.82,
    "detailed_scores": [
        {
            "question": "Q1.1 — Récursivité factoriel",
            "score": 2.0, "max": 2.5,
            "feedback": "Logique correcte mais erreur de syntaxe : '=' au lieu de '==' dans le if.",
            "lacunes": [{"concept": "syntaxe Python", "niveau": 0.55, "type_erreur": "syntaxe"}],
        },
        {
            "question": "Q1.2 — Somme récursive",
            "score": 2.5, "max": 2.5,
            "feedback": "Parfait ! Cas de base et appel récursif corrects.",
            "lacunes": [],
        },
        {
            "question": "Q2.1 — Liste vs Tuple",
            "score": 3.0, "max": 3.5,
            "feedback": "Bonne définition, exemple clair. Manque la notion d'immuabilité des tuples comme clés de dict.",
            "lacunes": [{"concept": "tuples avancés", "niveau": 0.65, "type_erreur": "incomplète"}],
        },
        {
            "question": "Q2.2 — Pile (Stack)",
            "score": 3.5, "max": 3.5,
            "feedback": "Excellente implémentation complète avec gestion des cas limites.",
            "lacunes": [],
        },
        {
            "question": "Q3.2 — Recherche binaire",
            "score": 1.5, "max": 2.0,
            "feedback": "Off-by-one : 'droite = len(liste)' devrait être 'len(liste) - 1'.",
            "lacunes": [{"concept": "recherche binaire", "niveau": 0.45, "type_erreur": "application"}],
        },
        {
            "question": "Q4.1 — POO héritage/polymorphisme",
            "score": 1.0, "max": 2.0,
            "feedback": "Héritage bien défini, mais polymorphisme mal maîtrisé.",
            "lacunes": [{"concept": "polymorphisme", "niveau": 0.3, "type_erreur": "conceptuelle"}],
        },
    ],
    "global_lacunes": [
        {"concept": "polymorphisme OOP", "niveau_maitrise": 0.30, "priorite": "haute", "type_erreur": "conceptuelle"},
        {"concept": "recherche binaire", "niveau_maitrise": 0.45, "priorite": "haute", "type_erreur": "application"},
        {"concept": "syntaxe Python", "niveau_maitrise": 0.55, "priorite": "moyenne", "type_erreur": "syntaxe"},
        {"concept": "tuples avancés", "niveau_maitrise": 0.65, "priorite": "faible", "type_erreur": "incomplète"},
    ],
    "points_forts": [
        "Excellente maîtrise de la récursivité",
        "Implémentation parfaite des structures de données (Pile)",
        "Bonne compréhension de la complexité algorithmique O(log n)",
    ],
    "recommandations": [
        "Revoir le chapitre sur le polymorphisme et les méthodes virtuelles",
        "Pratiquer la recherche binaire avec attention aux indices (off-by-one)",
        "Réviser la syntaxe Python : opérateurs de comparaison vs affectation",
    ],
    "feedback_global": (
        "Très bon travail Ahmed ! Tu maîtrises bien les structures de données "
        "et la récursivité — c'est ton point fort. Concentre-toi sur le polymorphisme "
        "et les détails d'implémentation. Tu es sur la bonne voie ! 💪"
    ),
}

MOCK_CLASS_DATA = [
    {"student_id": "ahmed_01",  "name": "Ahmed Benali",   "score": 13.5, "max": 20,
     "gaps": ["polymorphisme", "recherche binaire", "syntaxe Python"]},
    {"student_id": "sara_02",   "name": "Sara Mansouri",  "score": 16.0, "max": 20,
     "gaps": ["récursivité avancée", "complexité spatiale"]},
    {"student_id": "youssef_03","name": "Youssef Chraibi","score": 9.5,  "max": 20,
     "gaps": ["POO", "récursivité", "pointeurs et références", "complexité algorithmique"]},
]

MOCK_CONVERSATIONS = [
    {
        "q": "Je ne comprends pas le polymorphisme en Python, peux-tu m'expliquer ?",
        "a": (
            "📚 **Rappel du concept** — Le polymorphisme signifie littéralement "
            "\"plusieurs formes\". En POO, c'est la capacité d'objets de *classes différentes* "
            "à répondre au même message (méthode) avec des comportements adaptés.\n\n"
            "💡 **Exemple concret** :\n"
            "```python\n"
            "class Animal:\n"
            "    def parler(self):\n"
            "        pass  # méthode abstraite\n\n"
            "class Chien(Animal):\n"
            "    def parler(self):\n"
            "        return 'Ouaf !'\n\n"
            "class Chat(Animal):\n"
            "    def parler(self):\n"
            "        return 'Miaou !'\n\n"
            "animaux = [Chien(), Chat(), Chien()]\n"
            "for a in animaux:\n"
            "    print(a.parler())  # chaque objet répond à SA façon\n"
            "```\n\n"
            "✏️ **Mini-quiz** : Qu'affiche `Chat().parler()` ? Et pourquoi peut-on "
            "appeler `.parler()` sur tous les animaux sans connaître leur type ?"
        ),
    },
    {
        "q": "Je comprends mieux ! Et l'erreur dans ma recherche binaire ?",
        "a": (
            "🎯 Excellent progrès ! Regardons ton bug ensemble.\n\n"
            "Dans ton code, tu écris `droite = len(liste)` — c'est un **off-by-one error** "
            "classique.\n\n"
            "```python\n"
            "# ❌ Ton code\n"
            "droite = len(liste)      # index hors tableau !\n\n"
            "# ✅ Correction\n"
            "droite = len(liste) - 1  # dernier index valide\n"
            "```\n\n"
            "💡 Si ta liste a 5 éléments `[1,3,5,7,9]`, les indices valides sont "
            "`0,1,2,3,4`. `len(liste)=5` pointe **hors** du tableau.\n\n"
            "✏️ **Exercice** : Trace l'exécution de `recherche_binaire([1,3,5,7,9], 7)` "
            "étape par étape avec ta correction. Combien d'itérations faut-il ?"
        ),
    },
    {
        "q": "Génère-moi un quiz sur la récursivité pour que je m'entraîne.",
        "a": (
            "🧪 **Quiz Récursivité — Niveau Intermédiaire**\n\n"
            "**Q1 (QCM)** — Quel est le cas de base obligatoire dans toute fonction récursive ?\n"
            "- A) Un appel récursif\n"
            "- B) Une condition d'arrêt ✅\n"
            "- C) Une boucle while\n"
            "- D) Un return None\n\n"
            "**Q2 (Vrai/Faux)** — Une fonction récursive sans cas de base provoque "
            "une `RecursionError` en Python.\n"
            "→ **Vrai** ✅ — Python limite la profondeur d'appels (~1000 par défaut)\n\n"
            "**Q3 (Code)** — Que retourne cette fonction ?\n"
            "```python\n"
            "def mystere(n):\n"
            "    if n <= 1: return n\n"
            "    return mystere(n-1) + mystere(n-2)\n"
            "print(mystere(6))\n"
            "```\n"
            "→ Indice : pense à la suite de Fibonacci... 🐇"
        ),
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# DEMO SECTIONS
# ═════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    """Display the opening banner."""
    console.print()
    banner = Text()
    banner.append("🎓  EduMentor OS", style="bold bright_cyan")
    banner.append("  —  ", style="dim")
    banner.append("Démonstration Complète du Système Multi-Agents", style="bold white")
    console.print(Panel(Align.center(banner), style="cyan", padding=(1, 4)))

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column(style="bright_white")
    meta.add_row("🤖 LLM",     f"Groq — llama-3.3-70b-versatile")
    meta.add_row("🧠 Agents",  "GraderAgent · TutorAgent · AnalystAgent")
    meta.add_row("🗄️  VectorDB","ChromaDB (local) + HuggingFace Embeddings")
    meta.add_row("🔗 Orchestration", "LangGraph StateGraph + MemorySaver")
    console.print(Align.center(meta))
    console.print()


def section_header(number: str, title: str, subtitle: str = "") -> None:
    console.print()
    console.print(Rule(style="dim"))
    label = Text()
    label.append(f"  ÉTAPE {number}", style="bold bright_yellow")
    label.append(f"  —  {title}", style="bold white")
    if subtitle:
        label.append(f"\n  {subtitle}", style="dim italic")
    console.print(Panel(label, style="yellow", padding=(0, 2)))
    console.print()


# ── ÉTAPE 1 : Grader ─────────────────────────────────────────────────────────

async def demo_grader() -> dict:
    section_header("1", "Grader Agent", "Correction automatique d'une copie d'examen PDF")

    # Create sample PDF
    pdf_path = ROOT / "data" / "sample_copies" / "exam_ahmed_benali.pdf"
    if not pdf_path.exists():
        console.print("  📄 Génération du PDF de démonstration...", style="dim")
        try:
            from demo._create_sample_pdf import create_sample_pdf
            create_sample_pdf(pdf_path)
            console.print(f"  ✅ PDF créé : [cyan]{pdf_path.name}[/cyan]")
        except Exception as e:
            console.print(f"  ⚠️  Impossible de créer le PDF ({e}) — utilisation des données simulées.", style="yellow")

    console.print(
        Panel(
            f"  [bold]Fichier :[/bold] [cyan]{pdf_path.name}[/cyan]\n"
            f"  [bold]Étudiant :[/bold] Ahmed Benali\n"
            f"  [bold]Matière :[/bold] Programmation Python — Algorithmique\n"
            f"  [bold]Barème :[/bold] 20 points (4 exercices)",
            title="📄 Copie à corriger",
            style="blue",
        )
    )
    console.print()

    grade_result = None

    # Attempt real API call
    with Progress(
        SpinnerColumn("dots", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("  [cyan]GraderAgent analyse la copie…", total=None)

        if pdf_path.exists():
            try:
                from agents.grader_agent import grade_submission
                grade_result = await asyncio.wait_for(
                    grade_submission(
                        pdf_path=str(pdf_path),
                        rubric_text="",
                        student_id="ahmed_benali_01",
                    ),
                    timeout=90,
                )
                progress.update(task, description="  [green]✅ Correction terminée par l'API réelle")
            except asyncio.TimeoutError:
                progress.update(task, description="  [yellow]⏱ Timeout — affichage des données simulées")
            except Exception as exc:
                progress.update(task, description=f"  [yellow]⚡ Mode demo (API : {str(exc)[:40]}…)")
        else:
            await asyncio.sleep(2.5)   # simulate API call duration
            progress.update(task, description="  [green]✅ Correction simulée terminée")

    if grade_result is None:
        grade_result = MOCK_GRADE_RESULT

    _render_grade_result(grade_result)
    return grade_result


def _render_grade_result(r: dict) -> None:
    """Render grading result with rich tables and panels."""
    score = r["total_score"]
    max_s = r["max_score"]
    pct   = r["percentage"]
    color = "green" if pct >= 70 else "yellow" if pct >= 50 else "red"

    # ── Score header ──────────────────────────────────────────────────────
    score_text = Text()
    score_text.append(f"\n   {score}", style=f"bold {color} underline")
    score_text.append(f" / {max_s}", style="bold white")
    score_text.append(f"   ({pct:.1f}%)", style=f"bold {color}")
    score_text.append(f"\n   Confiance de l'agent : {r.get('confidence_score', 0):.0%}", style="dim")
    console.print(Panel(score_text, title="📊 Note finale", style=color, width=50))
    console.print()

    # ── Detailed scores table ─────────────────────────────────────────────
    table = Table(
        title="Détail par question",
        box=box.ROUNDED,
        style="blue",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Question", style="bold", width=28)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Feedback", width=42)
    table.add_column("Type d'erreur", width=15)

    for q in r.get("detailed_scores", []):
        qs = q["score"]
        qm = q["max"]
        ratio = qs / qm if qm else 0
        sc_color = "green" if ratio >= 0.8 else "yellow" if ratio >= 0.5 else "red"
        lacunes = q.get("lacunes", [])
        err_type = lacunes[0]["type_erreur"] if lacunes else "—"
        table.add_row(
            q["question"],
            f"[{sc_color}]{qs}/{qm}[/{sc_color}]",
            q["feedback"],
            f"[dim]{err_type}[/dim]",
        )
    console.print(table)
    console.print()

    # ── Global lacunes ────────────────────────────────────────────────────
    lacune_table = Table(
        title="🔴 Lacunes identifiées",
        box=box.SIMPLE_HEAVY,
        header_style="bold red",
    )
    lacune_table.add_column("Concept", style="bold")
    lacune_table.add_column("Maîtrise", justify="center")
    lacune_table.add_column("Priorité", justify="center")
    lacune_table.add_column("Type d'erreur")

    for lac in r.get("global_lacunes", []):
        niv = lac["niveau_maitrise"]
        bar = "█" * int(niv * 10) + "░" * (10 - int(niv * 10))
        niv_color = "green" if niv > 0.7 else "yellow" if niv > 0.4 else "red"
        prio_color = {"haute": "red", "moyenne": "yellow", "faible": "green"}.get(
            lac["priorite"], "white"
        )
        lacune_table.add_row(
            lac["concept"],
            f"[{niv_color}]{bar} {niv:.0%}[/{niv_color}]",
            f"[{prio_color}]{lac['priorite']}[/{prio_color}]",
            lac["type_erreur"],
        )
    console.print(lacune_table)
    console.print()

    # ── Points forts ──────────────────────────────────────────────────────
    forts_text = "\n".join(f"  ✅ {p}" for p in r.get("points_forts", []))
    reco_text  = "\n".join(f"  📌 {p}" for p in r.get("recommandations", []))

    console.print(
        Columns([
            Panel(forts_text, title="💪 Points forts", style="green", width=55),
            Panel(reco_text,  title="📋 Recommandations", style="yellow", width=55),
        ])
    )
    console.print()

    # ── Feedback global ───────────────────────────────────────────────────
    console.print(
        Panel(
            f"  [italic]{r.get('feedback_global', '')}[/italic]",
            title="💬 Feedback global de l'agent",
            style="bright_cyan",
        )
    )
    console.print()

    # ── JSON complet ──────────────────────────────────────────────────────
    if Confirm.ask("  Afficher le JSON complet de la correction ?", default=False):
        syntax = Syntax(
            json.dumps(r, ensure_ascii=False, indent=2),
            "json",
            theme="monokai",
            line_numbers=True,
        )
        console.print(Panel(syntax, title="📦 GradeResult JSON", style="dim"))


# ── ÉTAPE 2 : Analyst ─────────────────────────────────────────────────────────

async def demo_analyst() -> None:
    section_header("2", "Analyst Agent", "Rapport de classe sur 3 étudiants")

    # Render student cards
    cards = []
    for s in MOCK_CLASS_DATA:
        pct = s["score"] / s["max"] * 100
        color = "green" if pct >= 70 else "yellow" if pct >= 50 else "red"
        gaps_str = "\n".join(f"  • {g}" for g in s["gaps"])
        card = Panel(
            f"[bold]{s['name']}[/bold]\n"
            f"Note : [{color}]{s['score']}/{s['max']} ({pct:.0f}%)[/{color}]\n"
            f"\n[dim]Lacunes :[/dim]\n{gaps_str}",
            style=color,
            width=36,
        )
        cards.append(card)
    console.print(Columns(cards))
    console.print()

    with Progress(
        SpinnerColumn("dots2", style="magenta"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("  [magenta]AnalystAgent génère le rapport de classe…", total=None)
        await asyncio.sleep(2)

    # Class stats table
    all_gaps: dict[str, int] = {}
    for s in MOCK_CLASS_DATA:
        for g in s["gaps"]:
            all_gaps[g] = all_gaps.get(g, 0) + 1

    class_avg = sum(s["score"] / s["max"] for s in MOCK_CLASS_DATA) / len(MOCK_CLASS_DATA)

    stats_table = Table(
        title=f"📊 Rapport de Classe — Moy. {class_avg:.0%}",
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        show_lines=True,
    )
    stats_table.add_column("Lacune commune", style="bold")
    stats_table.add_column("Étudiants concernés", justify="center")
    stats_table.add_column("Fréquence", justify="center")
    stats_table.add_column("Action recommandée")

    actions = {
        "polymorphisme":        "Cours collectif + TP POO",
        "récursivité":          "Exercices guidés en séance",
        "recherche binaire":    "Correction commentée + quiz",
        "complexité algorithmique": "Révision Big-O avec exemples",
        "syntaxe Python":       "Atelier débogage en groupe",
        "POO":                  "Refonte du chapitre POO",
        "complexité spatiale":  "Introduction memory profiling",
        "récursivité avancée":  "Problèmes Leetcode niveau medium",
        "pointeurs et références": "Schémas visuels mémoire",
        "tuples avancés":       "Exercices dict/set avec tuples",
    }

    for gap, count in sorted(all_gaps.items(), key=lambda x: -x[1]):
        freq = count / len(MOCK_CLASS_DATA)
        bar = "█" * int(freq * 5) + "░" * (5 - int(freq * 5))
        color = "red" if freq >= 0.6 else "yellow" if freq >= 0.3 else "green"
        stats_table.add_row(
            gap,
            str(count),
            f"[{color}]{bar} {freq:.0%}[/{color}]",
            actions.get(gap, "Exercices supplémentaires"),
        )

    console.print(stats_table)

    # At-risk students
    at_risk = [s for s in MOCK_CLASS_DATA if s["score"] / s["max"] < 0.55]
    if at_risk:
        risk_text = "\n".join(
            f"  ⚠️  [bold]{s['name']}[/bold] — {s['score']}/{s['max']} "
            f"({s['score']/s['max']:.0%}) — lacunes : {', '.join(s['gaps'][:2])}"
            for s in at_risk
        )
        console.print(
            Panel(risk_text, title="🚨 Étudiants à risque (< 55%)", style="red")
        )
    console.print()


# ── ÉTAPE 3 : Tutor ──────────────────────────────────────────────────────────

async def demo_tutor() -> None:
    section_header(
        "3",
        "Tutor Agent — Session de tutorat adaptatif",
        "Simulation d'une conversation Ahmed ↔ EduMentor",
    )

    console.print(
        Panel(
            "  [bold]Étudiant :[/bold] Ahmed Benali (ahmed_benali_01)\n"
            "  [bold]Profil :[/bold] Niveau intermédiaire — lacunes en polymorphisme et recherche binaire\n"
            "  [bold]Mode :[/bold] Méthode Socratique — questions guidantes avant réponse",
            title="🎯 Contexte de la session",
            style="cyan",
        )
    )
    console.print()

    history: list[dict] = []

    for i, exchange in enumerate(MOCK_CONVERSATIONS, 1):
        console.print(Rule(f"  Tour {i} / {len(MOCK_CONVERSATIONS)}", style="dim"))

        # Human message
        console.print(
            Panel(
                f"  {exchange['q']}",
                title="🧑 Ahmed",
                style="blue",
                title_align="left",
            )
        )

        # Simulate "thinking"
        with Progress(
            SpinnerColumn("aesthetic", style="green"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as prog:
            prog.add_task("  EduMentor réfléchit…", total=None)

            # Try real API for first question only
            ai_response = None
            if i == 1:
                try:
                    from agents.tutor_agent import chat_with_tutor
                    ai_response, history = await asyncio.wait_for(
                        chat_with_tutor(
                            student_id="ahmed_benali_01",
                            message=exchange["q"],
                            conversation_history=history,
                        ),
                        timeout=30,
                    )
                except Exception:
                    pass

            if ai_response is None:
                await asyncio.sleep(1.5)
                ai_response = exchange["a"]
                history.append({"role": "human",  "content": exchange["q"]})
                history.append({"role": "ai",     "content": ai_response})

        # AI response rendered as Markdown
        console.print(
            Panel(
                Markdown(ai_response),
                title="🤖 EduMentor",
                style="green",
                title_align="left",
                padding=(1, 2),
            )
        )
        console.print()

        if i < len(MOCK_CONVERSATIONS):
            await asyncio.sleep(0.8)

    # Conversation summary
    summary_table = Table(box=box.SIMPLE, show_header=False)
    summary_table.add_column(style="dim", width=22)
    summary_table.add_column(style="bold white")
    summary_table.add_row("Tours échangés",    str(len(MOCK_CONVERSATIONS)))
    summary_table.add_row("Concepts couverts", "Polymorphisme, Recherche binaire, Récursivité")
    summary_table.add_row("Quiz générés",       "1 (3 questions, niveau intermédiaire)")
    summary_table.add_row("Méthode",           "Socratique — questions guidantes ✅")

    console.print(Panel(summary_table, title="📋 Bilan de la session", style="cyan"))
    console.print()


# ── OUTRO ─────────────────────────────────────────────────────────────────────

def demo_outro() -> None:
    section_header("4", "Architecture du système", "Graphe LangGraph + Stack technique")

    # LangGraph ASCII tree
    tree = Tree("🔷 [bold cyan]EduMentor OS — LangGraph StateGraph[/bold cyan]")

    entry = tree.add("⬛ [bold]START[/bold]")
    router = entry.add("⚡ [yellow]route_by_task()[/yellow]  (conditional)")
    router.add(
        "📝 [blue]task='grade'[/blue] ──► node_grader"
        "\n                      ──► node_quality_check"
        "\n                          ├─ [green]OK[/green] ──► node_analyst ──► END"
        "\n                          └─ [red]LOW[/red] ──► node_human_review ──► END"
    )
    router.add("💬 [cyan]task='tutor_chat'[/cyan] ──► node_tutor ──► END")
    router.add("📊 [magenta]task='analyze_class'[/magenta] ──► node_analyst ──► END")
    tree.add("💾 [dim]MemorySaver checkpointer (thread_id = student_id)[/dim]")

    console.print(Panel(tree, title="🗺️  Graphe d'orchestration", style="bright_blue"))
    console.print()

    # Tech stack table
    stack = Table(title="🛠️  Stack technique", box=box.ROUNDED, header_style="bold")
    stack.add_column("Couche", style="bold cyan", width=20)
    stack.add_column("Technologie", width=35)
    stack.add_column("Rôle", width=35)
    rows = [
        ("Orchestration",    "LangGraph 0.2+",             "StateGraph multi-agents + mémoire"),
        ("LLM",              "Groq llama-3.3-70b",          "Inférence rapide et gratuite"),
        ("Embeddings (RAG)", "HuggingFace all-MiniLM-L6",  "Local, sans API key, 384 dims"),
        ("Vector Store",     "ChromaDB (local)",            "3 collections persistées sur disque"),
        ("API",              "FastAPI + Uvicorn",           "6 endpoints REST + background tasks"),
        ("PDF",              "PyMuPDF (fitz)",              "Extraction texte + détection code"),
        ("Sécurité code",    "subprocess + blacklist",      "Sandbox étudiant, timeout 10 s"),
        ("UI terminal",      "Rich",                        "Tables, spinners, syntax highlight"),
    ]
    for r in rows:
        stack.add_row(*r)
    console.print(stack)
    console.print()

    # Final message
    outro = Text()
    outro.append("\n  ✅  ", style="bold green")
    outro.append("Démonstration terminée.\n", style="bold white")
    outro.append("  🚀  Backend EduMentor OS prêt pour la Phase Finale !\n", style="bold bright_cyan")
    outro.append("\n  Pour lancer l'API :\n", style="dim")
    outro.append("  uvicorn api.main:app --reload --port 8000\n", style="bold yellow")
    outro.append("  Docs : http://localhost:8000/docs\n", style="dim cyan")
    console.print(Panel(outro, style="green", padding=(1, 2)))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print_banner()

    try:
        # ── Étape 1 : Grader ─────────────────────────────────────────────
        grade_result = await demo_grader()

        if not Confirm.ask("\n  ▶  Continuer vers l'Étape 2 (Analyst) ?", default=True):
            demo_outro()
            return

        # ── Étape 2 : Analyst ─────────────────────────────────────────────
        await demo_analyst()

        if not Confirm.ask("\n  ▶  Continuer vers l'Étape 3 (Tutor) ?", default=True):
            demo_outro()
            return

        # ── Étape 3 : Tutor ───────────────────────────────────────────────
        await demo_tutor()

    except KeyboardInterrupt:
        console.print("\n\n  [yellow]Démonstration interrompue par l'utilisateur.[/yellow]\n")

    finally:
        demo_outro()


if __name__ == "__main__":
    asyncio.run(main())
