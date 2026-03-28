# tools/rubric_retriever.py
# RubricRetriever tool — Fetches grading rubrics from the data store.
#
# Responsibilities:
#   - Loads structured rubric definitions from the data/rubrics/ directory
#     (JSON or YAML format).
#   - Supports retrieval by rubric_id, subject, or assignment type.
#   - Parses rubrics into a standardized RubricSchema (Pydantic model) with
#     fields: criteria[], max_score, subject, description.
#   - Caches loaded rubrics in memory to avoid redundant disk reads.
#   - Provides a search interface to find the best-matching rubric for a given
#     assignment description using embedding similarity (via RAG module).
#
# Input:  rubric_id (str) or query (str)
# Output: RubricSchema object
# Used by: GraderAgent
