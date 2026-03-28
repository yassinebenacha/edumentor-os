# agents/grader_agent.py
# GraderAgent — Automated grading agent.
#
# Responsibilities:
#   - Receives a student's submission (text or code) and a rubric.
#   - Uses a language model to evaluate the submission against each rubric criterion.
#   - Assigns a score and generates detailed, criterion-level feedback.
#   - Supports multi-modal inputs (PDF-parsed text via pdf_parser tool, 
#     code execution results via code_executor tool).
#   - Emits a structured GradeResult object consumed by the main LangGraph state.
#
# LangGraph role: A node in the grading subgraph.
# Tools used: rubric_retriever, pdf_parser, code_executor.
# Input state keys: submission_text, rubric_id, student_id.
# Output state keys: grade_result, feedback_items.
