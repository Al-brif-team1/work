"""Models for criterion evaluation results."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CriterionEvaluationStatus(str, Enum):
    """Status assigned to an evaluated criterion."""

    met = "met"
    not_met = "not_met"
    insufficient_information = "insufficient_information"
    risk_detected = "risk_detected"


class CriterionEvaluation(BaseModel):
    """Result for a single criterion."""

    criterion: str
    criterion_title: str | None = None
    status: CriterionEvaluationStatus
    evidence: list[str] = Field(default_factory=list)
    explanation: str | None = None
    confidence: float | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")
