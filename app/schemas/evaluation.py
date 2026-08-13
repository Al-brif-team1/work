"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CriterionEvaluationStatus(str, Enum):
    """Класс «CriterionEvaluationStatus» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    met = "met"
    not_met = "not_met"
    insufficient_information = "insufficient_information"
    risk_detected = "risk_detected"


class CriterionEvaluation(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    criterion: str
    criterion_title: str | None = None
    status: CriterionEvaluationStatus
    evidence: list[str] = Field(default_factory=list)
    explanation: str | None = None
    confidence: float | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")
