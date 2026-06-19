"""Pydantic models for multi-stage code review pipeline."""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ReviewVerdict(str, Enum):
    PASS = "pass"
    REFINE = "refine"


class ReviewerResult(BaseModel):
    """Structured output from a single review dimension."""

    reviewer: Literal["correctness", "security", "code_style"]
    passed: bool
    score: float = Field(ge=0, le=10, description="Quality score from 0 to 10")
    findings: list[str] = Field(default_factory=list)
    summary: str


class ComposerReviewVerdict(BaseModel):
    """Final composed review decision fed back to the coding agent on refine."""

    verdict: ReviewVerdict
    summary: str
    refinement_instructions: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    correctness: ReviewerResult
    security: ReviewerResult
    code_style: ReviewerResult
