# tools/quiz_generator.py
# QuizGenerator tool — Generates adaptive quizzes for reinforcement learning.
#
# Responsibilities:
#   - Creates multiple-choice, true/false, and short-answer questions on a given topic.
#   - Adapts difficulty level based on the student's current mastery score for the topic
#     (retrieved from the student profile).
#   - Uses the LLM to generate question-answer pairs, distractors, and explanations.
#   - Optionally seeds generation from relevant RAG-retrieved knowledge base content
#     to ensure factual accuracy.
#   - Returns a Quiz object: {topic, difficulty, questions[], answer_key, explanations[]}.
#   - Supports configurable number of questions and question type distribution.
#
# Input:  topic (str), student_id (str), num_questions (int), difficulty (str)
# Output: Quiz object (Pydantic model)
# Used by: TutorAgent
