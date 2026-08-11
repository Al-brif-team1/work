"""Models for MVP planning results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MVPPlan(BaseModel):
    """Structured MVP plan that preserves the original project goal."""

    core_goal: str
    keep: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    simplify: list[str] = Field(default_factory=list)
    mvp_scope: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("core_goal")
    @classmethod
    def _strip_core_goal(cls, value: str) -> str:
        """Normalize the core goal text."""
        value = value.strip()
        if not value:
            raise ValueError("core_goal must not be empty")

        return value

    @field_validator("keep", "remove", "simplify", "mvp_scope", "rationale")
    @classmethod
    def _normalize_text_list(cls, value: list[str]) -> list[str]:
        """Normalize list-style plan fields."""
        if not isinstance(value, list):
            raise ValueError("must be a list")

        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("list items must be strings")
            item = item.strip()
            if not item:
                raise ValueError("list items must not be empty")
            normalized.append(item)

        return normalized


class MVPPlanningTechnicalInfo(BaseModel):
    """Technical metadata for the MVP planning run."""

    llm_invoked: bool
    attempts: int
    prompt_name: str
    trace_enabled: bool
    trace_name: str
    model_name: str | None = None
    skipped_reason: str | None = None
    raw_response: dict[str, Any] | None = None
    recovered_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class MVPPlanningResult(BaseModel):
    """Structured result produced by the MVP planner."""

    plan: MVPPlan | None = None
    technical_info: MVPPlanningTechnicalInfo

    model_config = ConfigDict(extra="forbid")
