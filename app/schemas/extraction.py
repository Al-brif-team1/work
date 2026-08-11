"""Extractor output models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FactStatus(str, Enum):
    """Extraction confidence marker for a fact."""

    explicit = "explicit"
    missing = "missing"
    uncertain = "uncertain"


class ExtractedFact(BaseModel):
    """A single extracted fact with status and evidence."""

    status: FactStatus
    value: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("value")
    @classmethod
    def _strip_empty_value(cls, value: str | None) -> str | None:
        """Treat blank values as missing."""
        if value is not None and not value.strip():
            return None

        return value


class ExtractedBrief(BaseModel):
    """Structured facts extracted from a project brief."""

    project_goal: ExtractedFact
    tasks: list[ExtractedFact] = Field(default_factory=list)
    project_type: ExtractedFact
    project_direction: ExtractedFact
    technologies: list[ExtractedFact] = Field(default_factory=list)
    stack: list[ExtractedFact] = Field(default_factory=list)
    materials: list[ExtractedFact] = Field(default_factory=list)
    expected_result: ExtractedFact
    constraints: list[ExtractedFact] = Field(default_factory=list)
    deadlines: list[ExtractedFact] = Field(default_factory=list)
    existing_resources: list[ExtractedFact] = Field(default_factory=list)
    integrations: list[ExtractedFact] = Field(default_factory=list)
    other_facts: list[ExtractedFact] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ExtractorTechnicalInfo(BaseModel):
    """Technical data about extraction and tracing."""

    attempts: int
    prompt_name: str
    trace_enabled: bool
    trace_name: str
    model_name: str | None = None
    raw_response: dict[str, Any] | None = None
    recovered_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ExtractionResult(BaseModel):
    """Complete result produced by the extractor."""

    extracted_brief: ExtractedBrief
    technical_info: ExtractorTechnicalInfo

    model_config = ConfigDict(extra="forbid")
