"""Models for completeness checking of extracted briefs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CompletenessStatus(str, Enum):
    """Status assigned to a field during completeness analysis."""

    present = "present"
    missing = "missing"
    clarification = "clarification"


class CompletenessLevel(str, Enum):
    """Aggregate completeness level used by downstream stages."""

    complete = "complete"
    needs_clarification = "needs_clarification"
    incomplete = "incomplete"


class CompletenessItem(BaseModel):
    """A single completeness assessment for one configured field."""

    field_key: str
    field_path: str
    title: str
    status: CompletenessStatus
    value: Any | None = None
    reason: str | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")


class CompletenessTechnicalInfo(BaseModel):
    """Technical metadata produced by the deterministic completeness stage."""

    checked_fields_count: int = 0
    required_fields_count: int = 0
    optional_fields_count: int = 0
    present_count: int = 0
    missing_count: int = 0
    critical_missing_count: int = 0
    clarification_count: int = 0

    model_config = ConfigDict(extra="forbid")


class CompletenessResult(BaseModel):
    """Result of a completeness check over an extracted brief."""

    is_complete: bool
    level: CompletenessLevel = CompletenessLevel.complete
    missing_information: list[CompletenessItem] = Field(default_factory=list)
    critical_missing_information: list[CompletenessItem] = Field(default_factory=list)
    present_information: list[CompletenessItem] = Field(default_factory=list)
    clarification_information: list[CompletenessItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    technical_info: CompletenessTechnicalInfo = Field(
        default_factory=CompletenessTechnicalInfo
    )

    model_config = ConfigDict(extra="forbid")
