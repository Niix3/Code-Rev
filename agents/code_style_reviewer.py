"""Code style reviewer against common standards."""
from typing import Any

from .review_models import ReviewerResult
from .review_utils import build_review_context, create_review_client, structured_completion


class CodeStyleReviewer:
    """Checks naming, formatting, structure, and maintainability."""

    def __init__(self):
        self.client = create_review_client()

    def review(
        self,
        query: str,
        coding_output: dict[str, Any],
        testing_output: dict[str, Any],
        workspace_path: str,
        test_authoring_output: dict[str, Any] | None = None,
        tester_mode: str | None = None,
        architecture_output: dict[str, Any] | None = None,
        review_context: str | None = None,
    ) -> ReviewerResult:
        context = review_context or build_review_context(
            query,
            coding_output,
            testing_output,
            workspace_path,
            test_authoring_output=test_authoring_output,
            architecture_output=architecture_output,
            tester_mode=tester_mode,
        )
        system_prompt = """You are a code style reviewer.
Review the captured coder patch and git diff as the primary source of truth, plus workspace context.
The "OpenHands edited files" list indicates paths the coding agent modified via file_editor.
Evaluate against widely accepted standards (PEP 8 for Python, idiomatic patterns, clear naming,
consistent formatting, reasonable module structure, docstrings where appropriate).
Return structured JSON with reviewer="code_style". Minor nits should not fail the review."""

        result = structured_completion(
            self.client,
            system_prompt=system_prompt,
            user_prompt=f"Review code style:\n\n{context}",
            response_model=ReviewerResult,
        )
        return result.model_copy(update={"reviewer": "code_style"})
