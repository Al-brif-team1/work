"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas import (
    AIContext,
    BriefAnalysisResult,
    BriefAssessmentSummary,
    BriefExtractedFields,
    CompletenessItem,
    DecisionStatus,
    ExtractedFact,
    RiskSeverity,
)


class BriefAnalysisResultError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


def deduplicate(values: list[str]) -> list[str]:
    """Убирает пустые строки и повторы, сохраняя исходный порядок. Нужна и билдеру публичного результата, и сборщику текста письма."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


class BriefAnalysisResultBuilder:
    """Класс «BriefAnalysisResultBuilder» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    _STATUS_MAP = {
        DecisionStatus.accept: "accept",
        DecisionStatus.clarify: "clarify",
        DecisionStatus.simplify: "simplify",
        DecisionStatus.mentor_review: "mentor_review",
        DecisionStatus.reject: "reject",
    }

    def __init__(self, *, field_titles_path: str | Path | None = None) -> None:
        """Подготавливает объект к работе: загружает русские названия полей брифа, потому что title в criteria.yaml английские и в результат их отдавать нельзя."""
        self._field_titles = self._load_field_titles(
            field_titles_path or self._default_field_titles_path()
        )

    def build(self, context: AIContext) -> BriefAnalysisResult:
        """Выполняет шаг «build». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        self._validate_context(context)
        extracted = context.extracted_brief
        assessment = context.assessment_result
        arbitration = context.arbitration_result

        assert extracted is not None
        assert assessment is not None
        assert arbitration is not None

        return BriefAnalysisResult(
            summary=self._build_summary(context),
            extracted_fields=BriefExtractedFields(
                goal=self._fact_value(extracted.project_goal),
                expected_result=self._fact_value(extracted.expected_result),
                tasks=self._fact_values(extracted.tasks),
                domain=self._fact_value(extracted.project_type),
                direction=self._normalize_direction(
                    self._fact_value(extracted.project_direction)
                ),
                available_materials=deduplicate(
                    [
                        *self._fact_values(extracted.materials),
                        *self._fact_values(extracted.existing_resources),
                        *self._fact_values(extracted.integrations),
                    ]
                ),
                missing_information=[
                    self._field_title(item)
                    for item in context.completeness_result.missing_information
                ]
                if context.completeness_result is not None
                else [],
                complexity_factors=deduplicate(
                    [
                        *self._fact_values(extracted.constraints),
                        *[
                            risk.description
                            for risk in assessment.risks
                            if risk.severity in {RiskSeverity.high, RiskSeverity.critical}
                        ],
                    ]
                ),
            ),
            assessment=BriefAssessmentSummary(
                recommendation=self._STATUS_MAP[arbitration.final_status],
                confidence=self._confidence_label(
                    arbitration.confidence or assessment.confidence
                ),
                reasons=deduplicate(
                    [

                        *[
                            item.explanation
                            for item in assessment.criterion_evaluations
                            if item.explanation
                        ],

                    ]
                ),
                risks=[risk.description for risk in assessment.risks],
            ),
            clarifying_questions=[
                item.question
                for item in (
                    context.clarification_result.questions
                    if context.clarification_result is not None
                    else []
                )
            ],
            mvp_suggestion=self._format_mvp_suggestion(context),
            customer_response_draft=context.final_response_text or "",
        )

    @staticmethod
    def _validate_context(context: AIContext) -> None:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if context.extracted_brief is None:
            raise BriefAnalysisResultError("Final result requires extracted_brief")
        if context.completeness_result is None:
            raise BriefAnalysisResultError("Final result requires completeness_result")
        if context.assessment_result is None:
            raise BriefAnalysisResultError("Final result requires assessment_result")
        if context.arbitration_result is None:
            raise BriefAnalysisResultError("Final result requires arbitration_result")
        if context.final_response_text is None:
            raise BriefAnalysisResultError("Final result requires final_response_text")

        status = context.arbitration_result.final_status
        if status is DecisionStatus.clarify and not (
            context.clarification_result
            and context.clarification_result.questions
        ):
            raise BriefAnalysisResultError(
                "CLARIFY final result requires clarification questions"
            )
        if status is DecisionStatus.simplify and not (
            context.mvp_planning_result and context.mvp_planning_result.plan
        ):
            raise BriefAnalysisResultError(
                "SIMPLIFY final result requires an MVP plan"
            )

    def _field_title(self, item: CompletenessItem) -> str:
        """Возвращает русское название поля брифа. Если перевода нет, отдаем машинный ключ: он хотя бы не выглядит английской фразой в русском отчете."""
        return self._field_titles.get(item.field_key, item.field_key)

    @staticmethod
    def _load_field_titles(path: str | Path) -> dict[str, str]:
        """Читает русские названия обязательных полей брифа из JSON-ресурса."""
        titles_path = Path(path)
        try:
            raw = json.loads(titles_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BriefAnalysisResultError(
                f"Unable to load field titles: {titles_path}"
            ) from exc

        if not isinstance(raw, dict):
            raise BriefAnalysisResultError("field titles must be a mapping")

        titles: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise BriefAnalysisResultError("field title keys must be strings")
            if not isinstance(value, str) or not value.strip():
                raise BriefAnalysisResultError(
                    f"field title for {key!r} must be a non-empty string"
                )
            titles[key.strip()] = value.strip()
        return titles

    @staticmethod
    def _default_field_titles_path() -> Path:
        """Возвращает значение по умолчанию, чтобы этап мог работать без ручной настройки."""
        return Path(__file__).resolve().parents[2] / "config" / "field_titles.json"

    @staticmethod
    def _fact_value(fact: ExtractedFact) -> str:
        """Выполняет шаг «fact value». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if fact.value is None:
            return ""
        value = fact.value.strip()
        return value

    @classmethod
    def _fact_values(cls, facts: list[ExtractedFact]) -> list[str]:
        """Выполняет шаг «fact values». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return deduplicate(
            [value for fact in facts if (value := cls._fact_value(fact))]
        )

    @staticmethod
    def _normalize_direction(value: str) -> str:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        normalized = " ".join(value.lower().split())
        aliases = {
            "разработка": "development",
            "development": "development",
            "дизайн": "design",
            "design": "design",
            "аналитика": "analytics",
            "analytics": "analytics",
            "маркетинг": "marketing",
            "marketing": "marketing",
            "ии": "ai",
            "ai": "ai",
            "искусственный интеллект": "ai",
            "образование": "education",
            "education": "education",
            "смешанный проект": "mixed",
            "mixed": "mixed",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _confidence_label(value: float | None) -> str:
        """Выполняет шаг «confidence label». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if value is None:
            return "medium"
        if value < 0.45:
            return "low"
        if value < 0.75:
            return "medium"
        return "high"

    def _build_summary(self, context: AIContext) -> str:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        if context.assessment_result and context.assessment_result.summary:
            return context.assessment_result.summary

        assert context.extracted_brief is not None
        goal = self._fact_value(context.extracted_brief.project_goal)
        result = self._fact_value(context.extracted_brief.expected_result)
        if goal and result:
            return f"{goal}. Ожидаемый результат: {result}."
        if goal:
            return goal
        if result:
            return f"Ожидаемый результат: {result}."
        return "Краткое описание проекта не удалось извлечь из брифа."

    @staticmethod
    def _format_mvp_suggestion(context: AIContext) -> str:
        """Выполняет шаг «format mvp suggestion». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if (
            context.arbitration_result is None
            or context.arbitration_result.final_status is not DecisionStatus.simplify
        ):
            return ""

        planning = context.mvp_planning_result
        if planning is None or planning.plan is None:
            return ""

        plan = planning.plan
        parts = [f"Цель MVP: {plan.core_goal}"]
        if plan.keep:
            parts.append("Оставить: " + "; ".join(plan.keep))
        if plan.simplify:
            parts.append("Упростить: " + "; ".join(plan.simplify))
        if plan.remove:
            parts.append("Исключить из первой версии: " + "; ".join(plan.remove))
        if plan.mvp_scope:
            parts.append("Состав первой версии: " + "; ".join(plan.mvp_scope))
        return "\n".join(parts)
