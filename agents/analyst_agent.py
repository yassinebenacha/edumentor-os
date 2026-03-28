# agents/analyst_agent.py
# AnalystAgent — Learning analytics and reporting agent.
#
# Responsibilities:
#   - Aggregates grading results, tutoring interactions, and quiz performance
#     across one or multiple students.
#   - Identifies patterns, trends, and recurring misconceptions at the class level.
#   - Generates actionable insights reports for educators (e.g., class performance
#     heatmaps, concept mastery rates, at-risk student flags).
#   - Updates student profiles with new performance data via the student_profile tool.
#   - Optionally uses Tavily web search to enrich reports with pedagogical best practices.
#
# LangGraph role: A terminal/reporting node in the main graph, triggered after
#                 grading or tutoring subgraphs complete.
# Tools used: student_profile, (optionally) Tavily search.
# Input state keys: grade_results[], tutoring_logs[], class_id.
# Output state keys: analytics_report, updated_profiles[].
