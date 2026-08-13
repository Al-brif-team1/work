"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClarificationQuestion(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    question: str
    related_field: str
    reason: str
    priority: int = 1

    model_config = ConfigDict(extra="forbid")

    @field_validator("question", "reason", "related_field")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        """Выполняет шаг «strip text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")

        return value

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: int) -> int:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if value <= 0:
            raise ValueError("priority must be greater than zero")

        return value


class QuestionGenerationTechnicalInfo(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

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
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    questions: list[ClarificationQuestion] = Field(default_factory=list)
    summary: str | None = None
    technical_info: QuestionGenerationTechnicalInfo

    model_config = ConfigDict(extra="forbid")
