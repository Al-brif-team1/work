"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import CriteriaConfig
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
from app.schemas.final_result import DirectionValue


class BriefAnalysisResultError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


_DIRECTION_VALUES = frozenset(
    {
        "development",
        "design",
        "analytics",
        "marketing",
        "ai",
        "education",
        "mixed",
    }
)

_FALLBACK_DIRECTION_SIGNALS = {
    "development": (
        "web",
        "website",
        "mobile",
        "backend",
        "frontend",
        "api",
        "app",
        "application",
        "service",
        "bot",
        "сайт",
        "веб-сервис",
        "приложение",
        "мобильное приложение",
        "бот",
        "портал",
        "автоматизация",
        "интеграция",
    ),
    "design": (
        "ux",
        "ui",
        "redesign",
        "interface",
        "mockup",
        "редизайн",
        "интерфейс",
        "макет",
        "прототип",
        "дизайн",
    ),
    "analytics": (
        "analytics",
        "analysis",
        "research",
        "bi",
        "dashboard",
        "metrics",
        "аналитика",
        "анализ данных",
        "исследование",
        "дашборд",
        "метрики",
    ),
    "marketing": (
        "marketing",
        "smm",
        "promotion",
        "advertising",
        "маркетинг",
        "реклама",
        "продвижение",
    ),
    "ai": (
        "ai",
        "ml",
        "llm",
        "nlp",
        "computer vision",
        "ии",
        "машинное обучение",
        "нейросеть",
        "нейросети",
        "компьютерное зрение",
    ),
    "education": (
        "education",
        "course",
        "learning materials",
        "methodology",
        "образование",
        "курс",
        "обучение",
        "учебные материалы",
        "методология",
    ),
}


def classify_direction(
    *,
    project_direction: ExtractedFact,
    project_type: ExtractedFact,
    project_goal: ExtractedFact,
    tasks: list[ExtractedFact],
    expected_result: ExtractedFact,
    criteria_config: CriteriaConfig | None = None,
) -> DirectionValue:
    """Classify the public direction enum without changing raw extracted facts."""
    alias_map = _build_direction_alias_map(criteria_config)
    explicit = _classify_exact_direction(project_direction.value, alias_map)
    if explicit is not None:
        return explicit

    signal_matches = _find_direction_signals([project_direction.value])
    if len(signal_matches) == 1:
        return next(iter(signal_matches))
    if len(signal_matches) > 1:
        return "mixed"

    context_values = [
        project_type.value,
        project_goal.value,
        *[task.value for task in tasks],
        expected_result.value,
    ]
    signal_matches = _find_direction_signals(context_values)
    if len(signal_matches) == 1:
        return next(iter(signal_matches))
    if len(signal_matches) > 1:
        return "mixed"

    raw = project_direction.value.strip() if project_direction.value else ""
    raise BriefAnalysisResultError(
        f"Unable to classify public project direction: {raw or '<empty>'}"
    )


def _build_direction_alias_map(
    criteria_config: CriteriaConfig | None,
) -> dict[str, DirectionValue]:
    aliases: dict[str, DirectionValue] = {
        value: value for value in _DIRECTION_VALUES
    }
    if criteria_config is None:
        return aliases

    for project_type in criteria_config.evaluation.project_types:
        if project_type.key not in _DIRECTION_VALUES:
            continue
        direction = project_type.key
        aliases[normalize_lookup_text(project_type.key)] = direction
        aliases[normalize_lookup_text(project_type.title)] = direction
        aliases[normalize_lookup_text(project_type.description)] = direction
        for alias in project_type.aliases:
            aliases[normalize_lookup_text(alias)] = direction
    return aliases


def _classify_exact_direction(
    value: str | None,
    aliases: dict[str, DirectionValue],
) -> DirectionValue | None:
    normalized = normalize_lookup_text(value)
    if not normalized:
        return None
    return aliases.get(normalized)


def _find_direction_signals(values: list[str | None]) -> set[DirectionValue]:
    matches: set[DirectionValue] = set()
    for value in values:
        normalized = normalize_lookup_text(value)
        if not normalized:
            continue
        for direction, signals in _FALLBACK_DIRECTION_SIGNALS.items():
            if any(contains_signal(normalized, signal) for signal in signals):
                matches.add(direction)

    if "ai" in matches:
        matches.discard("development")
    if "development" in matches:
        matches.discard("education")
    return matches


def contains_signal(value: str, signal: str) -> bool:
    """Ищет сигнал в уже нормализованном тексте. Короткие латинские сигналы вроде «ai» или «nft» проверяются по границам слова, иначе они всплывали бы внутри посторонних слов; длинные и кириллические ищутся подстрокой, чтобы ловить любые окончания. Нужна и классификатору направления, и матчеру запрещённых тем."""
    normalized_signal = normalize_lookup_text(signal)
    if not normalized_signal:
        return False
    if len(normalized_signal) <= 3 and normalized_signal.isascii():
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_signal)}(?![a-z0-9])"
        return re.search(pattern, value) is not None
    return normalized_signal in value


def normalize_lookup_text(value: str | None) -> str:
    """Приводит текст к виду, пригодному для поиска сигналов: нижний регистр, пунктуация в пробелы, схлопнутые пробелы. Смысл не меняется, убирается только то, что мешает сравнению."""
    if value is None:
        return ""
    value = value.strip().lower()
    value = re.sub(r"[\"'`.,;:!?()\[\]{}<>/\\|]+", " ", value)
    return " ".join(value.split())


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
        DecisionStatus.accept_with_clarifications: "accept_with_clarifications",
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
                direction=classify_direction(
                    project_direction=extracted.project_direction,
                    project_type=extracted.project_type,
                    project_goal=extracted.project_goal,
                    tasks=extracted.tasks,
                    expected_result=extracted.expected_result,
                    criteria_config=context.configuration,
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

        status = context.arbitration_result.final_status
        if status in {
            DecisionStatus.clarify,
            DecisionStatus.accept_with_clarifications,
        } and not (
            context.clarification_result
            and context.clarification_result.questions
        ):
            raise BriefAnalysisResultError(
                f"{status.value} final result requires clarification questions"
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
