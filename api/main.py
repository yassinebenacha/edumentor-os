# api/main.py
# FastAPI application entry point for EduMentor-OS.
#
# Endpoints:
#
#   POST /grade
#       - Accepts: multipart form data with PDF file (student submission) + rubric_id + student_id.
#       - Triggers the LangGraph graph with task_type="grade".
#       - Returns: GradeResult JSON (score, feedback, criterion breakdown).
#
#   POST /tutor
#       - Accepts: JSON body with student_id, topic, user_message, thread_id (for memory).
#       - Triggers the graph with task_type="tutor".
#       - Returns: TutorResponse JSON (explanation, resources, optional quiz).
#       - Supports streaming responses via Server-Sent Events (SSE).
#
#   GET  /report/{class_id}
#       - Triggers the graph with task_type="report".
#       - Returns: AnalyticsReport JSON for the specified class.
#
#   POST /ingest
#       - Admin endpoint to trigger knowledge base ingestion (rebuilds vector index).
#       - Protected by a simple API key header check.
#
# Application setup:
#   - Loads environment variables from .env via python-dotenv on startup.
#   - Configures CORS middleware for cross-origin frontend access.
#   - Imports the compiled LangGraph `app` from graphs/main_graph.py.
#   - Uses python-multipart for file upload handling.
#
# Run with: uvicorn api.main:app --reload
