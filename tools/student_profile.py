# tools/student_profile.py
# StudentProfile tool — Reads and updates persistent student learning profiles.
#
# Responsibilities:
#   - Maintains a per-student profile stored as JSON in a persistent data layer
#     (file-based initially, database-ready via an abstract interface).
#   - Profile schema includes: student_id, name, grade_level, learning_style,
#     subject_mastery{}, weak_areas[], session_history[], total_sessions.
#   - Provides get_profile(student_id) and update_profile(student_id, updates) operations.
#   - Tracks cumulative performance metrics updated after each grading/tutoring session.
#   - Used by TutorAgent to personalize content and AnalystAgent to generate reports.
#
# Input:  student_id (str), optional update payload (dict)
# Output: StudentProfileSchema object
# Used by: TutorAgent, AnalystAgent
