# graphs/main_graph.py
# Main LangGraph orchestration graph for EduMentor-OS.
#
# Graph Architecture:
#   - Defines the shared AgentState (TypedDict) passed between all nodes.
#     State fields: student_id, task_type, submission_text, rubric_id,
#     topic, grade_result, tutor_response, analytics_report, messages[].
#
#   - Nodes:
#       * "grader"   → calls GraderAgent to evaluate a submission
#       * "tutor"    → calls TutorAgent to provide personalized feedback/learning
#       * "analyst"  → calls AnalystAgent to generate reports and update profiles
#
#   - Edges & Routing:
#       * Entry point: conditional router that reads task_type from state.
#         - "grade"  → routes to "grader" node
#         - "tutor"  → routes directly to "tutor" node
#         - "report" → routes directly to "analyst" node
#       * After "grader": always routes to "tutor" (grading triggers tutoring feedback).
#       * After "tutor":  routes to "analyst" to update student profile.
#       * After "analyst": END.
#
#   - Compilation:
#       * The graph is compiled with a MemorySaver checkpointer to support
#         multi-turn conversational state persistence (thread_id based).
#       * The compiled `app` object is imported by api/main.py for invocation.
#
# Dependencies: langgraph, agents.*, config.py
