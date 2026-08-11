"""Models for clarification question generation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClarificationQuestion(BaseModel):
    """A single clarification question."""

    question: str
    related_field: str
    reason: str
    priority: int = 1

    model_config = ConfigDict(extra="forbid")

    @field_validator("question", "reason", "related_field")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        """Normalize user-facing text fields."""
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")

        return value

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: int) -> int:
        """Ensure priority is a positive integer."""
        if value <= 0:
            raise ValueError("priority must be greater than zero")

        return value


class QuestionGenerationTechnicalInfo(BaseModel):
    """Technical metadata about question generation."""

    llm_invoked: bool
    attempts: int
    prompt_name: str | None = None
    trace_enabled: bool
    trace_name: str
    model_name: str | None = None
    question_count: int = 0
    missing_template_fields: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    recovered_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class QuestionGenerationResult(BaseModel):
    """Structured result produced by the question generator."""

    questions: list[ClarificationQuestion] = Field(default_factory=list)
    summary: str | None = None
    technical_info: QuestionGenerationTechnicalInfo

    model_config = ConfigDict(extra="forbid")
