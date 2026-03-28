# config.py
# Centralized configuration for EduMentor-OS.
#
# Responsibilities:
#   - Loads all environment variables from .env using python-dotenv.
#   - Exposes typed, validated settings as module-level constants or a
#     Pydantic BaseSettings class (Settings) for easy import across the project.
#
# Configuration groups:
#
#   LLM Settings:
#     OPENAI_API_KEY        — OpenAI API key for LLM and embedding calls.
#     LLM_MODEL             — Default chat model (e.g., "gpt-4o-mini").
#     LLM_TEMPERATURE       — Sampling temperature (default: 0.2 for grading, 0.7 for tutoring).
#     EMBEDDING_MODEL       — Embedding model name (default: "text-embedding-3-small").
#
#   Search / Web:
#     TAVILY_API_KEY        — Tavily Search API key for web-augmented analysis.
#
#   Observability (LangSmith):
#     LANGSMITH_API_KEY     — LangSmith API key for tracing and evaluation.
#     LANGSMITH_PROJECT     — LangSmith project name for trace grouping.
#     LANGCHAIN_TRACING_V2  — Enables LangSmith tracing (set to "true").
#
#   Storage:
#     DATA_DIR              — Root path for data/ directory (default: "./data").
#     CHROMA_PERSIST_DIR    — ChromaDB persistence path (default: "./data/chroma_db").
#     KNOWLEDGE_BASE_DIR    — Path to knowledge base documents for RAG ingestion.
#     RUBRICS_DIR           — Path to rubric JSON/YAML files.
#
#   API:
#     API_HOST              — Uvicorn host (default: "0.0.0.0").
#     API_PORT              — Uvicorn port (default: 8000).
#     ADMIN_API_KEY         — Simple secret key for admin endpoints (e.g., /ingest).
#
#   Code Executor:
#     CODE_EXEC_TIMEOUT     — Timeout in seconds for student code execution (default: 10).
