"""Brief input and request metadata models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BriefInputMetadata(BaseModel):
    """Metadata describing how a brief was received."""

    source: str = "cli"
    input_type: str = "text"
    file_path: str | None = None
    file_name: str | None = None
    encoding: str = "utf-8"
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class BriefInput(BaseModel):
    """Validated brief input used by the AI pipeline."""

    original_text: str
    normalized_text: str
    metadata: BriefInputMetadata = Field(default_factory=BriefInputMetadata)

    model_config = ConfigDict(extra="forbid")

    @field_validator("original_text", "normalized_text")
    @classmethod
    def _ensure_non_empty(cls, value: str) -> str:
        """Reject blank text values."""
        if not value or not value.strip():
            raise ValueError("text must not be empty")

        return value
