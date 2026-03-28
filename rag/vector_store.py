# rag/vector_store.py
# VectorStore — ChromaDB-backed vector database interface for educational content.
#
# Responsibilities:
#   - Initializes and manages a persistent ChromaDB collection for the knowledge base.
#   - Ingestion pipeline:
#       * Reads documents (PDF, Markdown, plain text) from data/knowledge_base/.
#       * Chunks documents using a RecursiveCharacterTextSplitter (configurable
#         chunk_size and chunk_overlap from config.py).
#       * Embeds chunks via the Embeddings module and upserts into ChromaDB.
#   - Retrieval:
#       * similarity_search(query, k) — returns top-k relevant document chunks.
#       * as_retriever() — wraps the collection as a LangChain BaseRetriever
#         for use in LCEL chains and LangGraph nodes.
#   - Supports metadata filtering (e.g., filter by subject, grade_level, doc_type).
#   - Provides a rebuild_index() utility to re-ingest all documents from scratch.
#
# Libraries: chromadb, langchain, langchain-openai
# Used by: TutorAgent (via quiz_generator), RubricRetriever
