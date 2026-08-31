"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.config import (
    CriteriaConfig,
    CriteriaConfigError,
    CriteriaLoader,
    Criterion,
    RestrictedTopic,
    RiskType,
    get_criteria_config,
)
from app.llm.runner import LLMRunResult, LLMRunner
from app.pipeline.base import BaseLLMStage
from app.pipeline.result_builder import contains_signal, normalize_lookup_text
from app.prompts import PromptManager, RenderedPrompt
from app.schemas.ai_context import AIContext
from app.schemas.assessment import (
    AssessmentEvidence,
    AssessmentPayload,
    AssessmentRecommendation,
    AssessmentResult,
    AssessmentTechnicalInfo,
)
from app.schemas.evaluation import CriterionEvaluation, CriterionEvaluationStatus
from app.schemas.knowledge import SearchResult
from app.schemas.risk import Risk, RiskSeverity
from app.tracing.tracing import TracingClient

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class AssessmentError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class AssessmentConfigError(AssessmentError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class AssessmentRetriever(Protocol):
    """Класс «AssessmentRetriever» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[SearchResult]:
        """Выполняет шаг «retrieve». Документация описывает назначение метода, а сама логика остается в коде ниже."""


class AssessmentPreparedInput(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    context: AIContext
    criteria_config: CriteriaConfig
    criteria: list[Criterion]
    risk_types: list[RiskType]
    restricted_topics: list[RestrictedTopic] = Field(default_factory=list)
    retrieved_context: list[SearchResult] = Field(default_factory=list)
    retrieval_query: str | None = None
    metadata_filters: dict[str, object] | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @property
    def criteria_count(self) -> int:
        """Выполняет шаг «criteria count». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return len(self.criteria)

    @property
    def risk_types_count(self) -> int:
        """Выполняет шаг «risk types count». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return len(self.risk_types)

    def to_prompt_context(self) -> dict[str, Any]:
        """Преобразует данные в другой формат. Это нужно, чтобы соседний слой системы получил информацию в удобном для себя виде."""
        return {
            "brief": self.context.brief_input.model_dump(mode="json"),
            "extracted_brief": (
                self.context.extracted_brief.model_dump(mode="json")
                if self.context.extracted_brief is not None
                else None
            ),
            "completeness_result": (
                self.context.completeness_result.model_dump(mode="json")
                if self.context.completeness_result is not None
                else None
            ),
            "criteria_config": self.criteria_config.model_dump(mode="json"),
            "retrieved_context": [
                item.model_dump(mode="json") for item in self.retrieved_context
            ],
            "metadata": self.context.metadata,
        }


class RestrictedTopicHit(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    topic: RestrictedTopic
    keyword: str
    source_title: str
    fragment: str

    model_config = ConfigDict(extra="forbid")


class RestrictedTopicMatcher:
    """[РОЛЬ В КОНВЕЙЕРЕ] Сверяет бриф со списком тем, которые Мастерская не берет. Работает обычным кодом, без ИИ: решение о запрете не должно зависеть от модели, температуры или попытки переубедить ее текстом брифа. Матчер намеренно грубый - лишнее срабатывание стоит менеджеру нескольких секунд, а пропущенная тема уезжает дальше как обычный проект."""

    # Порог, после которого цитата в письме менеджеру обрезается до окна вокруг совпадения.
    _MAX_FRAGMENT_LENGTH = 160
    _FRAGMENT_MARGIN = 60

    def __init__(self, criteria_config: CriteriaConfig) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        configuration = criteria_config.evaluation.restricted_topics
        self._topics = list(configuration.topics) if configuration is not None else []

    def match(self, context: AIContext) -> RestrictedTopicHit | None:
        """Возвращает первую сработавшую тему или None. Порядок тем в конфиге задает приоритет, а источники перебираются от точных извлеченных фактов к тексту брифа: так в цитату попадает короткая формулировка, а не кусок всего письма."""
        if not self._topics:
            return None

        sources = self._build_sources(context)
        for topic in self._topics:
            for source_title, source_text in sources:
                for keyword in topic.keywords:
                    if not contains_signal(source_text, keyword):
                        continue
                    return RestrictedTopicHit(
                        topic=topic,
                        keyword=keyword,
                        source_title=source_title,
                        fragment=self._fragment(source_text, keyword),
                    )
        return None

    @staticmethod
    def _build_sources(context: AIContext) -> list[tuple[str, str]]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        raw: list[tuple[str, str | None]] = []
        extracted = context.extracted_brief
        if extracted is not None:
            raw.append(("цель проекта", extracted.project_goal.value))
            raw.append(("ожидаемый результат", extracted.expected_result.value))
            for task in extracted.tasks:
                raw.append(("задача", task.value))
            raw.append(("тип проекта", extracted.project_type.value))
            raw.append(("направление", extracted.project_direction.value))
        # Текст брифа идет последним: он самый полный, но и самый длинный.
        raw.append(("текст брифа", context.normalized_text))

        sources: list[tuple[str, str]] = []
        for title, value in raw:
            normalized = normalize_lookup_text(value)
            if normalized:
                sources.append((title, normalized))
        return sources

    @classmethod
    def _fragment(cls, source_text: str, keyword: str) -> str:
        """Готовит человекочитаемый текст из внутренних данных. Это нужно для промптов, объяснений или финального ответа."""
        if len(source_text) <= cls._MAX_FRAGMENT_LENGTH:
            return source_text

        # Ищем подстрокой: совпадение по границам слова его тоже содержит,
        # а точная позиция нужна только для того, чтобы вырезать окно цитаты.
        position = source_text.find(normalize_lookup_text(keyword))
        if position < 0:
            return source_text[: cls._MAX_FRAGMENT_LENGTH].rstrip() + "…"

        start = max(0, position - cls._FRAGMENT_MARGIN)
        end = min(len(source_text), position + len(keyword) + cls._FRAGMENT_MARGIN)
        fragment = source_text[start:end].strip()
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(source_text) else ""
        return f"{prefix}{fragment}{suffix}"


def build_restricted_topic_assessment(hit: RestrictedTopicHit) -> AssessmentResult:
    """Собирает тот же AssessmentResult, что вернул бы LLM-этап, но детерминированно и без вызова модели. Дальше по конвейеру разницы нет: арбитр читает типы рисков одинаково, откуда бы результат ни пришел."""
    evidence = f"{hit.source_title}: {hit.fragment}"
    description = (
        f"{hit.topic.customer_reason} "
        f"Тема «{hit.topic.title}» определена по формулировке — {evidence}."
    )
    return AssessmentResult(
        criterion_evaluations=[
            CriterionEvaluation(
                criterion="topic_eligibility",
                criterion_title=hit.topic.title,
                status=CriterionEvaluationStatus.not_met,
                evidence=[evidence],
                explanation=hit.topic.customer_reason,
                confidence=1.0,
            )
        ],
        risks=[
            Risk(
                type="restricted_topic",
                description=description,
                severity=RiskSeverity.critical,
                evidence=[evidence],
                confidence=1.0,
                notes=f"restricted_topic={hit.topic.key}; keyword={hit.keyword}",
            )
        ],
        has_risks=True,
        recommendation=AssessmentRecommendation.high_risk_review,
        summary=hit.topic.customer_reason,
        confidence=1.0,
        technical_info=AssessmentTechnicalInfo(
            attempts=0,
            prompt_name=None,
            trace_name="assessment.restricted_topic",
            retriever_used=False,
        ),
    )


class AssessmentStage(
    BaseLLMStage[AssessmentPreparedInput, AssessmentPayload, AssessmentResult]
):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Эксперт-аналитик. Он через ИИ ищет риски, сложности и соответствие критериям проекта. Этот этап обращается к LLM, поэтому внутри работает ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseLLMStage, поэтому он умеет работать в нашем конвейере."""

    output_model: ClassVar[type[AssessmentPayload]] = AssessmentPayload

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tracing_client: TracingClient | None = None,
        prompt_path: str | Path | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        prompt_manager: PromptManager | None = None,
        max_retries: int = 2,
        model_name: str | None = None,
        llm_runner: LLMRunner | None = None,
        retriever: AssessmentRetriever | None = None,
        criteria_config: CriteriaConfig | None = None,
        criteria_path: str | Path | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        super().__init__(
            llm_client=llm_client,
            tracing_client=tracing_client,
            prompt_path=prompt_path,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_manager=prompt_manager,
            max_retries=max_retries,
            model_name=model_name,
            llm_runner=llm_runner,
        )
        self._preparation = AssessmentPreparation(
            retriever=retriever,
            criteria_config=criteria_config,
            criteria_path=criteria_path,
        )
        self._restricted_topics = RestrictedTopicMatcher(
            self._preparation.criteria_config
        )
        self._last_run_metadata: dict[str, Any] = {}

    def assess(
        self,
        context: AIContext,
        *,
        top_k: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> AIContext:
        """Выполняет шаг «assess». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        # Запрещенная тема - решение политики, а не оценка качества брифа, поэтому
        # оно принимается обычным кодом. Замыкание стоит до prepare: так пропускается
        # и вызов модели, и обращение к базе знаний.
        self._preparation._validate_context(context)
        hit = self._restricted_topics.match(context)
        if hit is not None:
            return context.with_assessment_result(
                build_restricted_topic_assessment(hit)
            )

        prepared = self._preparation.prepare(
            context,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        return prepared.context.with_assessment_result(self.run(prepared))

    def run_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
        return self.assess(context)

    @property
    def trace_name(self) -> str:
        """Выполняет шаг «trace name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return "assessment.brief"

    @property
    def span_name(self) -> str:
        """Выполняет шаг «span name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return "assessment.llm"

    def build_prompt(self, stage_input: AssessmentPreparedInput) -> str:
        """Выполняет шаг «build prompt». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        rendered = self._render_assessment_prompt(stage_input)
        return rendered.user or rendered.system

    def build_system_prompt(self, stage_input: AssessmentPreparedInput) -> str | None:
        """Выполняет шаг «build system prompt». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        rendered = self._render_assessment_prompt(stage_input)
        return rendered.system if rendered.user is not None else None

    def build_context(self, stage_input: AssessmentPreparedInput) -> None:
        """Выполняет шаг «build context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return None

    def build_trace_input(self, stage_input: AssessmentPreparedInput) -> dict[str, Any]:
        """Выполняет шаг «build trace input». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        metadata = {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "criteria_count": stage_input.criteria_count,
            "risk_types_count": stage_input.risk_types_count,
            "retriever_used": self._preparation.retriever_used,
            "retrieved_context_count": len(stage_input.retrieved_context),
            "is_complete": stage_input.context.completeness_result.is_complete,
            "completeness_level": stage_input.context.completeness_result.level.value,
        }
        self._last_run_metadata = metadata
        return metadata

    def postprocess(
        self,
        result: LLMRunResult[AssessmentPayload],
    ) -> AssessmentResult:
        """Выполняет шаг «postprocess». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        payload = self._normalize_payload(result.payload)
        return AssessmentResult(
            criterion_evaluations=payload.criterion_evaluations,
            risks=payload.risks,
            evidence=payload.evidence,
            has_risks=payload.has_risks,
            recommendation=payload.recommendation,
            summary=self._strip_optional_text(payload.summary),
            confidence=payload.confidence,
            technical_info=AssessmentTechnicalInfo(
                attempts=result.attempts,
                prompt_name=self.prompt_name,
                trace_enabled=result.trace_enabled,
                trace_name=result.trace_name or self.trace_name,
                model_name=result.model_name,
                retriever_used=bool(self._last_run_metadata.get("retriever_used")),
                retrieved_context_count=int(
                    self._last_run_metadata.get("retrieved_context_count", 0)
                ),
                criteria_count=int(self._last_run_metadata.get("criteria_count", 0)),
                risk_types_count=int(
                    self._last_run_metadata.get("risk_types_count", 0)
                ),
                raw_response=result.raw_response,
                recovered_errors=list(result.recovered_errors),
                provider_metadata=dict(result.provider_metadata),
            ),
        )

    @staticmethod
    def _default_prompt_path() -> Path:
        """Возвращает значение по умолчанию, чтобы этап мог работать без ручной настройки."""
        return Path(__file__).resolve().parents[2] / "prompts" / "assessment.md"

    def _build_failure_exception(
        self,
        attempts: int,
        last_error: Exception | None,
    ) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return AssessmentError(f"Unable to assess brief after {attempts} attempts")

    def _render_assessment_prompt(
        self,
        stage_input: AssessmentPreparedInput,
    ) -> RenderedPrompt:
        """Готовит человекочитаемый текст из внутренних данных. Это нужно для промптов, объяснений или финального ответа."""
        return self._render_prompt(
            {
                "normalized_brief": stage_input.context.normalized_text,
                "extracted_brief": stage_input.context.extracted_brief,
                "completeness_result": stage_input.context.completeness_result,
                "criteria": [
                    item.model_dump(mode="json") for item in stage_input.criteria
                ],
                "risk_types": [
                    item.model_dump(mode="json") for item in stage_input.risk_types
                ],
                "restricted_topics": [
                    item.model_dump(mode="json")
                    for item in stage_input.restricted_topics
                ],
                "retrieved_context": [
                    item.model_dump(mode="json")
                    for item in stage_input.retrieved_context
                ],
            }
        )

    def _normalize_payload(self, payload: AssessmentPayload) -> AssessmentPayload:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        return payload.model_copy(
            update={
                "criterion_evaluations": [
                    self._normalize_criterion(item)
                    for item in payload.criterion_evaluations
                ],
                "risks": [self._normalize_risk(item) for item in payload.risks],
                "evidence": [
                    self._normalize_evidence(item)
                    for item in payload.evidence
                    if item.quote.strip()
                ],
                "summary": self._strip_optional_text(payload.summary),
            }
        )

    @classmethod
    def _normalize_criterion(
        cls,
        item: CriterionEvaluation,
    ) -> CriterionEvaluation:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        return item.model_copy(
            update={
                "criterion": item.criterion.strip(),
                "criterion_title": cls._strip_optional_text(item.criterion_title),
                "evidence": cls._strip_text_list(item.evidence),
                "explanation": cls._strip_optional_text(item.explanation),
                "notes": cls._strip_optional_text(item.notes),
            }
        )

    @classmethod
    def _normalize_risk(cls, item: Risk) -> Risk:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        return item.model_copy(
            update={
                "type": item.type.strip(),
                "description": item.description.strip(),
                "evidence": cls._strip_text_list(item.evidence),
                "notes": cls._strip_optional_text(item.notes),
            }
        )

    @classmethod
    def _normalize_evidence(cls, item: AssessmentEvidence) -> AssessmentEvidence:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        return item.model_copy(
            update={
                "source": item.source.strip(),
                "quote": item.quote.strip(),
                "related_criteria": cls._strip_text_list(item.related_criteria),
                "related_risks": cls._strip_text_list(item.related_risks),
            }
        )

    @staticmethod
    def _strip_optional_text(value: str | None) -> str | None:
        """Выполняет шаг «strip optional text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if value is None:
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _strip_text_list(values: list[str]) -> list[str]:
        """Выполняет шаг «strip text list». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return [value.strip() for value in values if value and value.strip()]


class AssessmentPreparation:
    """Класс «AssessmentPreparation» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(
        self,
        retriever: AssessmentRetriever | None = None,
        criteria_config: CriteriaConfig | None = None,
        criteria_path: str | Path | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        if criteria_config is not None and criteria_path is not None:
            raise ValueError("Pass either criteria_config or criteria_path, not both")

        self._retriever = retriever
        self._config = self._load_config(criteria_config, criteria_path)
        self._criteria = list(self._config.evaluation.criteria)
        self._risk_types = list(self._config.evaluation.risk_analysis.risk_types)
        # Секции может не быть: урезанные конфиги в тестах описывают только правила.
        restricted_topics = self._config.evaluation.restricted_topics
        self._restricted_topics = (
            list(restricted_topics.topics) if restricted_topics is not None else []
        )

    @property
    def criteria_config(self) -> CriteriaConfig:
        """Выполняет шаг «criteria config». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._config

    @property
    def criteria(self) -> list[Criterion]:
        """Выполняет шаг «criteria». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return list(self._criteria)

    @property
    def risk_types(self) -> list[RiskType]:
        """Выполняет шаг «risk types». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return list(self._risk_types)

    @property
    def restricted_topics(self) -> list[RestrictedTopic]:
        """Выполняет шаг «restricted topics». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return list(self._restricted_topics)

    @property
    def retriever_used(self) -> bool:
        """Выполняет шаг «retriever used». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._retriever is not None

    def prepare(
        self,
        context: AIContext,
        *,
        top_k: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> AssessmentPreparedInput:
        """Выполняет шаг «prepare». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        self._validate_context(context)
        retrieval_query = self._build_retrieval_query(context)
        retrieved_context = self._retrieve_context(
            context=context,
            retrieval_query=retrieval_query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

        return AssessmentPreparedInput(
            context=context.with_retrieved_context(retrieved_context),
            criteria_config=self._config,
            criteria=self._criteria,
            risk_types=self._risk_types,
            restricted_topics=self._restricted_topics,
            retrieved_context=retrieved_context,
            retrieval_query=retrieval_query,
            metadata_filters=(
                dict(metadata_filters) if metadata_filters is not None else None
            ),
        )

    def _retrieve_context(
        self,
        context: AIContext,
        retrieval_query: str,
        top_k: int | None,
        metadata_filters: Mapping[str, object] | None,
    ) -> list[SearchResult]:
        """Выполняет шаг «retrieve context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if self._retriever is None:
            return list(context.retrieved_context)

        return self._retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

    @staticmethod
    def _build_retrieval_query(context: AIContext) -> str:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        parts: list[str] = [context.normalized_text]

        if context.extracted_brief is not None:
            extracted = context.extracted_brief
            if extracted.project_goal.value:
                parts.append(extracted.project_goal.value)
            parts.extend(item.value for item in extracted.tasks if item.value)
            parts.extend(item.value for item in extracted.technologies if item.value)
            parts.extend(item.value for item in extracted.integrations if item.value)

        if context.completeness_result is not None:
            parts.extend(
                item.field_key
                for item in context.completeness_result.missing_information
            )

        return "\n".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _validate_context(context: AIContext) -> None:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if context.extracted_brief is None:
            raise AssessmentError("Assessment requires extracted_brief in AIContext")
        if context.completeness_result is None:
            raise AssessmentError("Assessment requires completeness_result in AIContext")

    @staticmethod
    def _load_config(
        criteria_config: CriteriaConfig | None,
        criteria_path: str | Path | None,
    ) -> CriteriaConfig:
        """Выполняет шаг «load config». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if criteria_config is not None:
            config = criteria_config
        elif criteria_path is not None:
            try:
                config = CriteriaLoader.load(Path(criteria_path))
            except CriteriaConfigError as exc:
                raise AssessmentConfigError(str(exc)) from exc
        else:
            try:
                config = get_criteria_config()
            except CriteriaConfigError as exc:
                raise AssessmentConfigError(str(exc)) from exc

        if not config.evaluation.criteria:
            raise AssessmentConfigError("criteria configuration has no criteria")

        risk_analysis = config.evaluation.risk_analysis
        if risk_analysis is None:
            raise AssessmentConfigError(
                "criteria configuration is missing risk_analysis section"
            )
        if not risk_analysis.risk_types:
            raise AssessmentConfigError(
                "risk_analysis configuration has no risk_types"
            )

        return config
