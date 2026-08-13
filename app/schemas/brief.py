"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BriefInputMetadata(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    source: str = "cli"
    input_type: str = "text"
    file_path: str | None = None
    file_name: str | None = None
    encoding: str = "utf-8"
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class BriefInput(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    original_text: str
    normalized_text: str
    metadata: BriefInputMetadata = Field(default_factory=BriefInputMetadata)

    model_config = ConfigDict(extra="forbid")

    @field_validator("original_text", "normalized_text")
    @classmethod
    def _ensure_non_empty(cls, value: str) -> str:
        """Выполняет шаг «ensure non empty». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if not value or not value.strip():
            raise ValueError("text must not be empty")

        return value
