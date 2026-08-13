"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionStatus(str, Enum):
    """Класс «DecisionStatus» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    accept = "ACCEPT"
    reject = "REJECT"
    clarify = "CLARIFY"
    simplify = "SIMPLIFY"
    mentor_review = "MENTOR_REVIEW"


class ArbitrationRuleHit(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    rule_key: str
    title: str
    status: DecisionStatus
    conditions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    explanation: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ArbitrationResult(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    final_status: DecisionStatus
    reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    triggered_rules: list[ArbitrationRuleHit] = Field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
