# rag/embeddings.py
# Embeddings — Centralized embedding model configuration and factory.
#
# Responsibilities:
#   - Provides a get_embeddings() factory function that returns a configured
#     LangChain Embeddings instance based on the EMBEDDING_MODEL setting in config.py.
#   - Default: OpenAIEmbeddings with model "text-embedding-3-small" (cost-efficient,
#     high performance for educational content retrieval).
#   - Designed to be easily swappable: supports HuggingFace sentence-transformers
#     as a local/offline alternative (no API key required).
#   - Embedding dimension and model name are surfaced in config.py to ensure
#     consistency between ingestion and retrieval phases.
#   - Singleton pattern: embeddings instance is cached to avoid redundant
#     initialization across multiple calls within a session.
#
# Libraries: langchain-openai, (optional) langchain-community for HuggingFace
# Used by: VectorStore
