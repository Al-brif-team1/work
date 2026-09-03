"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.evaluation import CriterionEvaluation
from app.schemas.risk import Risk
from app.schemas.traffic_light import TrafficLightResult


class AssessmentRecommendation(str, Enum):
    """Класс «AssessmentRecommendation» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    ready_for_arbitration = "ready_for_arbitration"
    needs_clarification = "needs_clarification"
    high_risk_review = "high_risk_review"


class AssessmentEvidence(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    source: str
    quote: str
    related_criteria: list[str] = Field(default_factory=list)
    related_risks: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("source", "quote")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Выполняет шаг «strip required text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        value = value.strip()
        if not value:
            raise ValueError("evidence source and quote must not be empty")
        return value


class AssessmentTechnicalInfo(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    attempts: int = 0
    prompt_name: str | None = None
    trace_enabled: bool = False
    trace_name: str = "assessment.brief"
    model_name: str | None = None
    retriever_used: bool = False
    retrieved_context_count: int = 0
    criteria_count: int = 0
    risk_types_count: int = 0
    raw_response: dict[str, Any] | None = None
    recovered_errors: list[str] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class AssessmentPayload(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    criterion_evaluations: list[CriterionEvaluation] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    evidence: list[AssessmentEvidence] = Field(default_factory=list)
    has_risks: bool
    recommendation: AssessmentRecommendation
    summary: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    traffic_light: TrafficLightResult = Field(default_factory=TrafficLightResult)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_consistency(self) -> "AssessmentPayload":
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if self.has_risks != bool(self.risks):
            raise ValueError("has_risks must match whether risks are present")
        return self


class AssessmentResult(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    criterion_evaluations: list[CriterionEvaluation] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    evidence: list[AssessmentEvidence] = Field(default_factory=list)
    has_risks: bool
    recommendation: AssessmentRecommendation
    summary: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    traffic_light: TrafficLightResult = Field(default_factory=TrafficLightResult)
    technical_info: AssessmentTechnicalInfo = Field(
        default_factory=AssessmentTechnicalInfo
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_consistency(self) -> "AssessmentResult":
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if self.has_risks != bool(self.risks):
            raise ValueError("has_risks must match whether risks are present")
        return self
