"""Models for deterministic arbitration results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionStatus(str, Enum):
    """Supported final decision statuses."""

    accept = "ACCEPT"
    reject = "REJECT"
    clarify = "CLARIFY"
    simplify = "SIMPLIFY"
    mentor_review = "MENTOR_REVIEW"


class ArbitrationRuleHit(BaseModel):
    """A single matched arbitration rule."""

    rule_key: str
    title: str
    status: DecisionStatus
    conditions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    explanation: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ArbitrationResult(BaseModel):
    """Final deterministic decision for a brief."""

    final_status: DecisionStatus
    reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    triggered_rules: list[ArbitrationRuleHit] = Field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
