"""Final pipeline result models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RecommendationValue = Literal[
    "accept",
    "clarify",
    "simplify",
    "mentor_review",
    "reject",
]
ConfidenceValue = Literal["low", "medium", "high"]


class BriefExtractedFields(BaseModel):
    """Product-facing extracted fields for one customer brief."""

    goal: str | None = None
    expected_result: str | None = None
    tasks: list[str] = Field(default_factory=list)
    domain: str | None = None
    direction: str | None = None
    available_materials: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    complexity_factors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BriefAssessmentSummary(BaseModel):
    """Product-facing assessment summary for one customer brief."""

    recommendation: RecommendationValue
    confidence: ConfidenceValue
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BriefAnalysisResult(BaseModel):
    """Structured JSON returned by the working brief analysis pipeline."""

    summary: str
    extracted_fields: BriefExtractedFields
    assessment: BriefAssessmentSummary
    clarifying_questions: list[str] = Field(default_factory=list)
    mvp_suggestion: str = ""
    customer_response_draft: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("summary", "customer_response_draft")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Reject blank required user-facing fields."""
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value
