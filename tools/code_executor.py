# tools/code_executor.py
# CodeExecutor tool — Safely executes student-submitted code for automated testing.
#
# Responsibilities:
#   - Accepts code snippets (Python initially, extensible to other languages).
#   - Runs code in an isolated subprocess with configurable timeouts and memory limits.
#   - Captures stdout, stderr, return values, and execution time.
#   - Compares output against expected test cases defined in the rubric.
#   - Returns a CodeExecutionResult with: passed_tests, failed_tests, runtime_errors,
#     output_diff, and a score contribution.
#   - SECURITY: Sandboxes execution to prevent file system access and network calls
#     from student code (uses subprocess restrictions and resource limits).
#
# Input:  code (str), test_cases (list), timeout_seconds (int)
# Output: CodeExecutionResult object
# Used by: GraderAgent
