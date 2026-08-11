"""Knowledge base data models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Metadata attached to a knowledge base document."""

    source: str | None = None
    title: str | None = None
    document_type: str | None = None
    category: str | None = None
    version: str | None = None
    date: str | None = None

    model_config = ConfigDict(extra="allow")


class Document(BaseModel):
    """A document stored in or retrieved from a vector store."""

    id: str
    text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    model_config = ConfigDict(extra="forbid")


class SearchResult(BaseModel):
    """A ranked similarity-search result."""

    document: Document
    score: float
    rank: int | None = None
    distance: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
