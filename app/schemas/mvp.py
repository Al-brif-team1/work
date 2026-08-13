"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MVPPlan(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    core_goal: str
    keep: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    simplify: list[str] = Field(default_factory=list)
    mvp_scope: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("core_goal")
    @classmethod
    def _strip_core_goal(cls, value: str) -> str:
        """Выполняет шаг «strip core goal». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        value = value.strip()
        if not value:
            raise ValueError("core_goal must not be empty")

        return value

    @field_validator("keep", "remove", "simplify", "mvp_scope", "rationale")
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


class MVPPlanningTechnicalInfo(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    llm_invoked: bool
    attempts: int
    prompt_name: str
    trace_enabled: bool
    trace_name: str
    model_name: str | None = None
    skipped_reason: str | None = None
    raw_response: dict[str, Any] | None = None
    recovered_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class MVPPlanningResult(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    plan: MVPPlan | None = None
    technical_info: MVPPlanningTechnicalInfo

    model_config = ConfigDict(extra="forbid")
