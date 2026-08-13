"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.assessment import AssessmentResult
from app.schemas.brief import BriefInput
from app.schemas.completeness import CompletenessResult
from app.schemas.decision import ArbitrationResult
from app.schemas.knowledge import SearchResult
from app.schemas.mvp import MVPPlanningResult
from app.schemas.question import QuestionGenerationResult
from app.schemas.extraction import ExtractedBrief


class SelfCheckContext(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    response_text: str
    response_payload: dict[str, Any] | None = None
    brief_input: BriefInput
    extracted_brief: ExtractedBrief
    completeness_result: CompletenessResult
    assessment_result: AssessmentResult
    arbitration_result: ArbitrationResult
    clarification_result: QuestionGenerationResult | None = None
    mvp_planning_result: MVPPlanningResult | None = None
    retrieved_context: list[SearchResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("response_text")
    @classmethod
    def _strip_response_text(cls, value: str) -> str:
        """Выполняет шаг «strip response text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        value = value.strip()
        if not value:
            raise ValueError("response_text must not be empty")

        return value


class SelfCheckPayload(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_fields: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("issues", "warnings", "checked_fields")
    @classmethod
    def _normalize_text_list(cls, value: list[str]) -> list[str]:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        if not isinstance(value, list):
            raise ValueError("must be a list")

        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("list items must be strings")
            item = item.strip()
            if not item:
                raise ValueError("list items must not be empty")
            normalized.append(item)

        return normalized

    @model_validator(mode="after")
    def _validate_consistency(self) -> "SelfCheckPayload":
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if self.is_valid and self.issues:
            raise ValueError("is_valid cannot be true when issues are present")
        if not self.is_valid and not self.issues:
            raise ValueError("is_valid cannot be false without issues")

        return self


class SelfCheckTechnicalInfo(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    deterministic_issues_count: int = 0
    deterministic_warning_count: int = 0
    llm_invoked: bool
    attempts: int
    prompt_name: str | None = None
    trace_enabled: bool
    trace_name: str
    model_name: str | None = None
    needs_llm_review: bool = False
    raw_response: dict[str, Any] | None = None
    recovered_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SelfCheckResult(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_fields: list[str] = Field(default_factory=list)
    technical_info: SelfCheckTechnicalInfo | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("issues", "warnings", "checked_fields")
    @classmethod
    def _normalize_text_list(cls, value: list[str]) -> list[str]:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        if not isinstance(value, list):
            raise ValueError("must be a list")

        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("list items must be strings")
            item = item.strip()
            if not item:
                raise ValueError("list items must not be empty")
            normalized.append(item)

        return normalized

    @model_validator(mode="after")
    def _validate_consistency(self) -> "SelfCheckResult":
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if self.is_valid and self.issues:
            raise ValueError("is_valid cannot be true when issues are present")
        if not self.is_valid and not self.issues:
            raise ValueError("is_valid cannot be false without issues")

        return self
