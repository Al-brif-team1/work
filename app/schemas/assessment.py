"""Unified assessment result models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.evaluation import CriterionEvaluation
from app.schemas.risk import Risk


class AssessmentRecommendation(str, Enum):
    """Non-binding analytical recommendation consumed by Deterministic Arbiter."""

    ready_for_arbitration = "ready_for_arbitration"
    needs_clarification = "needs_clarification"
    high_risk_review = "high_risk_review"


class AssessmentEvidence(BaseModel):
    """Evidence fragment used to support an assessment conclusion."""

    source: str
    quote: str
    related_criteria: list[str] = Field(default_factory=list)
    related_risks: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("source", "quote")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Reject empty evidence identifiers and fragments."""
        value = value.strip()
        if not value:
            raise ValueError("evidence source and quote must not be empty")
        return value


class AssessmentTechnicalInfo(BaseModel):
    """Technical metadata about a future assessment run."""

    attempts: int = 0
    prompt_name: str | None = None
    trace_enabled: bool = False
    trace_name: str = "assessment.brief"
    model_name: str | None = None
    retriever_used: bool = False
    retrieved_context_count: int = 0
    criteria_count: int = 0
    risk_types_count: int = 0
    raw_response: dict[str, Any] | None = None
    recovered_errors: list[str] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class AssessmentPayload(BaseModel):
    """Raw structured payload expected from the future Assessment LLM stage."""

    criterion_evaluations: list[CriterionEvaluation] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    evidence: list[AssessmentEvidence] = Field(default_factory=list)
    has_risks: bool
    recommendation: AssessmentRecommendation
    summary: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_consistency(self) -> "AssessmentPayload":
        """Keep aggregate assessment flags aligned with detailed findings."""
        if self.has_risks != bool(self.risks):
            raise ValueError("has_risks must match whether risks are present")
        return self


class AssessmentResult(BaseModel):
    """Unified criteria and risk assessment output consumed by arbitration."""

    criterion_evaluations: list[CriterionEvaluation] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    evidence: list[AssessmentEvidence] = Field(default_factory=list)
    has_risks: bool
    recommendation: AssessmentRecommendation
    summary: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    technical_info: AssessmentTechnicalInfo = Field(
        default_factory=AssessmentTechnicalInfo
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_consistency(self) -> "AssessmentResult":
        """Keep aggregate assessment flags aligned with detailed findings."""
        if self.has_risks != bool(self.risks):
            raise ValueError("has_risks must match whether risks are present")
        return self
