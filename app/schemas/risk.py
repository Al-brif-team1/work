"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskSeverity(str, Enum):
    """Класс «RiskSeverity» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Risk(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    type: str
    description: str
    severity: RiskSeverity
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")
