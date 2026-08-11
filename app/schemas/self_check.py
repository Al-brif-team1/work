"""Models for final self-check validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.assessment import AssessmentResult
from app.schemas.brief import BriefInput
from app.schemas.completeness import CompletenessResult
from app.schemas.decision import ArbitrationResult
from app.schemas.knowledge import SearchResult
from app.schemas.mvp import MVPPlanningResult
from app.schemas.question import QuestionGenerationResult
from app.schemas.extraction import ExtractedBrief


class SelfCheckContext(BaseModel):
    """Input context for final response validation."""

    response_text: str
    response_payload: dict[str, Any] | None = None
    brief_input: BriefInput
    extracted_brief: ExtractedBrief
    completeness_result: CompletenessResult
    assessment_result: AssessmentResult
    arbitration_result: ArbitrationResult
    clarification_result: QuestionGenerationResult | None = None
    mvp_planning_result: MVPPlanningResult | None = None
    retrieved_context: list[SearchResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("response_text")
    @classmethod
    def _strip_response_text(cls, value: str) -> str:
        """Reject blank response text."""
        value = value.strip()
        if not value:
            raise ValueError("response_text must not be empty")

        return value


class SelfCheckPayload(BaseModel):
    """Raw JSON payload expected from the LLM self-check."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_fields: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("issues", "warnings", "checked_fields")
    @classmethod
    def _normalize_text_list(cls, value: list[str]) -> list[str]:
        """Normalize list-based string fields."""
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

    @model_validator(mode="after")
    def _validate_consistency(self) -> "SelfCheckPayload":
        """Keep the validity flag aligned with recorded issues."""
        if self.is_valid and self.issues:
            raise ValueError("is_valid cannot be true when issues are present")
        if not self.is_valid and not self.issues:
            raise ValueError("is_valid cannot be false without issues")

        return self


class SelfCheckTechnicalInfo(BaseModel):
    """Technical metadata for the self-check run."""

    deterministic_issues_count: int = 0
    deterministic_warning_count: int = 0
    llm_invoked: bool
    attempts: int
    prompt_name: str | None = None
    trace_enabled: bool
    trace_name: str
    model_name: str | None = None
    needs_llm_review: bool = False
    raw_response: dict[str, Any] | None = None
    recovered_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SelfCheckResult(BaseModel):
    """Structured result produced by the final self-check."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_fields: list[str] = Field(default_factory=list)
    technical_info: SelfCheckTechnicalInfo | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("issues", "warnings", "checked_fields")
    @classmethod
    def _normalize_text_list(cls, value: list[str]) -> list[str]:
        """Normalize list-based string fields."""
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

    @model_validator(mode="after")
    def _validate_consistency(self) -> "SelfCheckResult":
        """Keep the validity flag aligned with reported issues."""
        if self.is_valid and self.issues:
            raise ValueError("is_valid cannot be true when issues are present")
        if not self.is_valid and not self.issues:
            raise ValueError("is_valid cannot be false without issues")

        return self
