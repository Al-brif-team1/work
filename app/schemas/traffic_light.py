"""Structured result of traffic-light skill-fit assessment."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrafficLightStatus(str, Enum):
    """Traffic-light status for student skill fit."""

    green = "green"
    yellow = "yellow"
    red = "red"
    unknown = "unknown"


class TrafficLightMatch(BaseModel):
    """One task-to-rule traffic-light match."""

    task: str
    matched_rule: str
    status: TrafficLightStatus
    reason: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("task", "matched_rule", "reason")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("traffic light match text fields must not be empty")
        return value


class TrafficLightResult(BaseModel):
    """Structured result of matching a brief to traffic-light rules."""

    status: TrafficLightStatus = TrafficLightStatus.unknown
    direction: str | None = None
    specialization: str | None = None
    matches: list[TrafficLightMatch] = Field(default_factory=list)
    reason: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("direction", "specialization", "reason")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None
