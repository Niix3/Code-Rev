"""Security reviewer for generated code."""
from typing import Any

from .review_models import ReviewerResult
from .review_utils import build_review_context, create_review_client, structured_completion


class SecurityReviewer:
    """Checks code for common security issues and unsafe patterns."""

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
        system_prompt = """You are a security code reviewer.
Review the captured coder patch and git diff as the primary source of truth, plus workspace context.
The "OpenHands edited files" list indicates paths the coding agent modified via file_editor.
Look for injection risks, unsafe deserialization, hardcoded secrets, insecure defaults,
missing input validation, path traversal, command injection, and auth flaws.
Return structured JSON with reviewer="security". Set passed=false for any material risk."""

        result = structured_completion(
            self.client,
            system_prompt=system_prompt,
            user_prompt=f"Review security:\n\n{context}",
            response_model=ReviewerResult,
        )
        return result.model_copy(update={"reviewer": "security"})
