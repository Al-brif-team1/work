"""Models for risk analysis results."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskSeverity(str, Enum):
    """Severity levels for identified risks."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Risk(BaseModel):
    """A single risk identified by the risk analyst."""

    type: str
    description: str
    severity: RiskSeverity
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")
