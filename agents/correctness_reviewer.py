"""Correctness reviewer: alignment of code, tests, and user request."""
from typing import Any, Optional

from config.settings import TesterMode
from .review_models import ReviewerResult
from .review_utils import build_review_context, create_review_client, structured_completion


class CorrectnessReviewer:
    """Checks that tests and implementation match the request and test run results."""

    def __init__(self):
        self.client = create_review_client()

    @staticmethod
    def _is_benchmarking(
        tester_mode: Optional[str | TesterMode],
        testing_output: dict[str, Any],
    ) -> bool:
        if testing_output.get("mode") == TesterMode.BENCHMARKING.value:
            return True
        if tester_mode is None:
            return False
        if isinstance(tester_mode, TesterMode):
            return tester_mode == TesterMode.BENCHMARKING
        return tester_mode == TesterMode.BENCHMARKING.value

    def review(
        self,
        query: str,
        coding_output: dict[str, Any],
        test_authoring_output: dict[str, Any],
        testing_output: dict[str, Any],
        workspace_path: str,
        tester_mode: Optional[str | TesterMode] = None,
        architecture_output: dict[str, Any] | None = None,
        review_context: str | None = None,
    ) -> ReviewerResult:
        benchmarking = self._is_benchmarking(tester_mode, testing_output)
        tests_passed = testing_output.get("passed", False)

        if not benchmarking and not tests_passed:
            return ReviewerResult(
                reviewer="correctness",
                passed=False,
                score=0.0,
                findings=[
                    "Test run did not pass.",
                    testing_output.get("response", "No test output provided."),
                ],
                summary="Correctness check failed: tests did not pass.",
            )
        context = review_context or build_review_context(
            query,
            coding_output,
            testing_output,
            workspace_path,
            test_authoring_output=test_authoring_output,
            architecture_output=architecture_output,
            benchmarking=benchmarking,
            tester_mode=tester_mode.value if isinstance(tester_mode, TesterMode) else tester_mode,
        )
        if benchmarking:
            system_prompt = """You are a correctness reviewer for benchmark mode (e.g. SWE-bench).
The repository already contains the official test suite; the agent did NOT author new tests.
In-pipeline test execution is skipped; verification happens via the external benchmark harness.
Review the coder changes using the "Coder patch (captured immediately after coding)" section as the primary source of truth.
If that section and the live git diff are both empty, treat the implementation as missing.
The "OpenHands edited files" list shows which paths the coding agent actually touched.
Evaluate:
1. Implementation addresses the user request/issue.
2. No obvious hardcoded hacks, stubs, or incomplete fixes.
3. Functional completeness relative to the request.
4. The change is coherent end-to-end relative to the issue description.
Return structured JSON with reviewer="correctness"."""
        else:
            system_prompt = """You are a correctness reviewer.
Review the git diff / captured coder patch together with authored tests and workspace context.
Use the captured patch and OpenHands edited-files list as the primary evidence of what the coder changed.
Evaluate:
1. Tests accurately reflect the user request (behavior, edge cases, not trivial assertions).
2. Implementation satisfies the tests without cheating (e.g. hardcoded return values).
3. Functional completeness relative to the user request.
4. Test run passed — confirm the solution is coherent end-to-end.
Return structured JSON with reviewer="correctness"."""

        result = structured_completion(
            self.client,
            system_prompt=system_prompt,
            user_prompt=f"Review correctness (tests + code):\n\n{context}",
            response_model=ReviewerResult,
        )
        passed = result.passed and (tests_passed or benchmarking)
        return result.model_copy(update={"reviewer": "correctness", "passed": passed})
