# agents/tutor_agent.py
# TutorAgent — Personalized tutoring agent.
#
# Responsibilities:
#   - Reads the student's profile (learning history, weak areas, learning style)
#     via the student_profile tool.
#   - Retrieves relevant educational content from the RAG vector store based on
#     the current topic and student needs.
#   - Generates adaptive explanations, examples, and hints tailored to the student.
#   - Recommends and generates quizzes via the quiz_generator tool to reinforce learning.
#   - Maintains a conversational memory to provide coherent multi-turn tutoring sessions.
#
# LangGraph role: A node in the tutoring subgraph.
# Tools used: student_profile, quiz_generator, vector_store (RAG retrieval).
# Input state keys: student_id, topic, conversation_history, grade_result (optional).
# Output state keys: tutor_response, recommended_resources, quiz (optional).
