"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RecommendationValue = Literal[
    "accept",
    "clarify",
    "simplify",
    "mentor_review",
    "reject",
]
ConfidenceValue = Literal["low", "medium", "high"]


class BriefExtractedFields(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    goal: str = ""
    expected_result: str = ""
    tasks: list[str] = Field(default_factory=list)
    domain: str = ""
    direction: str = ""
    available_materials: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    complexity_factors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BriefAssessmentSummary(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    recommendation: RecommendationValue
    confidence: ConfidenceValue
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BriefAnalysisResult(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    summary: str
    extracted_fields: BriefExtractedFields
    assessment: BriefAssessmentSummary
    clarifying_questions: list[str] = Field(default_factory=list)
    mvp_suggestion: str = ""
    customer_response_draft: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("summary", "customer_response_draft")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Выполняет шаг «strip required text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value
