"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CompletenessStatus(str, Enum):
    """Класс «CompletenessStatus» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    present = "present"
    missing = "missing"
    clarification = "clarification"


class CompletenessLevel(str, Enum):
    """Класс «CompletenessLevel» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    complete = "complete"
    needs_clarification = "needs_clarification"
    incomplete = "incomplete"


class CompletenessItem(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    field_key: str
    field_path: str
    title: str
    status: CompletenessStatus
    value: Any | None = None
    reason: str | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")


class CompletenessTechnicalInfo(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    checked_fields_count: int = 0
    required_fields_count: int = 0
    optional_fields_count: int = 0
    present_count: int = 0
    missing_count: int = 0
    critical_missing_count: int = 0
    clarification_count: int = 0

    model_config = ConfigDict(extra="forbid")


class CompletenessResult(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

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
