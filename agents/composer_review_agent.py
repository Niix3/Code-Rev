"""Composer that merges reviewer outputs into a pass/refine verdict."""
import json
from typing import Any

from .review_models import ComposerReviewVerdict, ReviewVerdict, ReviewerResult
from .review_utils import create_review_client, structured_completion


class ComposerReviewAgent:
    """Aggregates reviewer results and emits a structured pass/refine decision."""

    def __init__(self):
        self.client = create_review_client()

    def compose(
        self,
        query: str,
        correctness: ReviewerResult,
        security: ReviewerResult,
        code_style: ReviewerResult,
    ) -> ComposerReviewVerdict:
        reviews_payload = {
            "correctness": correctness.model_dump(),
            "security": security.model_dump(),
            "code_style": code_style.model_dump(),
        }
        system_prompt = """You are a composer review agent.
Combine correctness, security, and code style reviews into one final decision.
Rules:
- verdict must be "pass" only if all three reviewers passed and there are no blocking issues.
- verdict must be "refine" if any reviewer failed or material issues remain.
- refinement_instructions must be actionable for a coding agent.
- Include the three reviewer results unchanged in the output fields correctness, security, code_style.
- blocking_issues lists issues that must be fixed before pass."""

        user_prompt = (
            f"User request:\n{query}\n\n"
            f"Reviewer outputs:\n{json.dumps(reviews_payload, ensure_ascii=False, indent=2)}\n\n"
            "Produce the final composed verdict."
        )

        verdict = structured_completion(
            self.client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ComposerReviewVerdict,
        )

        if not all([correctness.passed, security.passed, code_style.passed]):
            verdict = verdict.model_copy(
                update={
                    "verdict": ReviewVerdict.REFINE,
                    "correctness": correctness,
                    "security": security,
                    "code_style": code_style,
                }
            )
        else:
            verdict = verdict.model_copy(
                update={
                    "correctness": correctness,
                    "security": security,
                    "code_style": code_style,
                }
            )

        if verdict.verdict == ReviewVerdict.PASS and verdict.blocking_issues:
            verdict = verdict.model_copy(update={"verdict": ReviewVerdict.REFINE})

        return verdict

    @staticmethod
    def to_feedback_dict(verdict: ComposerReviewVerdict) -> dict[str, Any]:
        """Serialize verdict for coding agent refinement prompts."""
        return verdict.model_dump(mode="json")
