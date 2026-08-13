"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.criteria import CriteriaConfig
from app.schemas.assessment import AssessmentResult
from app.schemas.brief import BriefInput
from app.schemas.completeness import CompletenessResult
from app.schemas.decision import ArbitrationResult
from app.schemas.extraction import ExtractionResult, ExtractedBrief
from app.schemas.knowledge import SearchResult
from app.schemas.mvp import MVPPlanningResult
from app.schemas.question import QuestionGenerationResult
from app.schemas.self_check import SelfCheckResult


class PipelineInputState(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    brief_input: BriefInput

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def original_text(self) -> str:
        """Выполняет шаг «original text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.brief_input.original_text

    @property
    def normalized_text(self) -> str:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        return self.brief_input.normalized_text


class RetrievalState(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    results: list[SearchResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class PipelineResults(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    extracted_brief: ExtractedBrief | None = None
    extraction_result: ExtractionResult | None = None
    completeness_result: CompletenessResult | None = None
    assessment_result: AssessmentResult | None = None
    arbitration_result: ArbitrationResult | None = None
    clarification_result: QuestionGenerationResult | None = None
    mvp_planning_result: MVPPlanningResult | None = None
    self_check_result: SelfCheckResult | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ResponseState(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    text: str | None = None
    payload: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        """Выполняет шаг «strip text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("final response text must not be empty")
        return value


class PipelineTechnicalState(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: CriteriaConfig | None = None
    stage_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


class AIContext(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    inputs: PipelineInputState
    retrieval: RetrievalState = Field(default_factory=RetrievalState)
    results: PipelineResults = Field(default_factory=PipelineResults)
    response: ResponseState = Field(default_factory=ResponseState)
    technical: PipelineTechnicalState = Field(default_factory=PipelineTechnicalState)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def brief_input(self) -> BriefInput:
        """Выполняет шаг «brief input». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.inputs.brief_input

    @property
    def original_text(self) -> str:
        """Выполняет шаг «original text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.inputs.original_text

    @property
    def normalized_text(self) -> str:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        return self.inputs.normalized_text

    @property
    def extracted_brief(self) -> ExtractedBrief | None:
        """Выполняет шаг «extracted brief». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.results.extracted_brief

    @property
    def extraction_result(self) -> ExtractionResult | None:
        """Выполняет шаг «extraction result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.results.extraction_result

    @property
    def completeness_result(self) -> CompletenessResult | None:
        """Выполняет шаг «completeness result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.results.completeness_result

    @property
    def assessment_result(self) -> AssessmentResult | None:
        """Выполняет шаг «assessment result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.results.assessment_result

    @property
    def arbitration_result(self) -> ArbitrationResult | None:
        """Выполняет шаг «arbitration result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.results.arbitration_result

    @property
    def clarification_result(self) -> QuestionGenerationResult | None:
        """Выполняет шаг «clarification result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.results.clarification_result

    @property
    def mvp_planning_result(self) -> MVPPlanningResult | None:
        """Выполняет шаг «mvp planning result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.results.mvp_planning_result

    @property
    def final_response_text(self) -> str | None:
        """Выполняет шаг «final response text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.response.text

    @property
    def final_response_payload(self) -> dict[str, Any] | None:
        """Выполняет шаг «final response payload». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.response.payload

    @property
    def self_check_result(self) -> SelfCheckResult | None:
        """Выполняет шаг «self check result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.results.self_check_result

    @property
    def retrieved_context(self) -> list[SearchResult]:
        """Выполняет шаг «retrieved context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.retrieval.results

    @property
    def metadata(self) -> dict[str, Any]:
        """Выполняет шаг «metadata». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.technical.metadata

    @property
    def configuration(self) -> CriteriaConfig | None:
        """Выполняет шаг «configuration». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.technical.configuration

    @property
    def stage_metadata(self) -> dict[str, dict[str, Any]]:
        """Выполняет шаг «stage metadata». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.technical.stage_metadata

    @classmethod
    def from_brief(
        cls,
        brief_input: BriefInput,
        *,
        metadata: dict[str, Any] | None = None,
        configuration: CriteriaConfig | None = None,
    ) -> "AIContext":
        """Выполняет шаг «from brief». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return cls(
            inputs=PipelineInputState(brief_input=brief_input),
            technical=PipelineTechnicalState(
                metadata=metadata or {},
                configuration=configuration,
            ),
        )

    def with_extraction_result(self, result: ExtractionResult) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self._with_results(
            update={
                "extraction_result": result,
                "extracted_brief": result.extracted_brief,
            }
        )

    def with_extracted_brief(self, extracted_brief: ExtractedBrief) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self._with_results(update={"extracted_brief": extracted_brief})

    def with_completeness_result(self, result: CompletenessResult) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self._with_results(update={"completeness_result": result})

    def with_assessment_result(self, result: AssessmentResult) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self._with_results(update={"assessment_result": result})

    def with_assessment(self, result: AssessmentResult) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self.with_assessment_result(result)

    def with_arbitration_result(self, result: ArbitrationResult) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self._with_results(update={"arbitration_result": result})

    def with_clarification_result(
        self,
        result: QuestionGenerationResult,
    ) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self._with_results(update={"clarification_result": result})

    def with_mvp_planning_result(self, result: MVPPlanningResult) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self._with_results(update={"mvp_planning_result": result})

    def with_retrieved_context(self, results: list[SearchResult]) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self.model_copy(
            update={"retrieval": RetrievalState(results=list(results))}
        )

    def append_retrieved_context(self, results: list[SearchResult]) -> "AIContext":
        """Выполняет шаг «append retrieved context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.model_copy(
            update={
                "retrieval": RetrievalState(
                    results=[*self.retrieved_context, *results]
                )
            }
        )

    def with_final_response(
        self,
        response_text: str,
        response_payload: dict[str, Any] | None = None,
    ) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self.model_copy(
            update={
                "response": ResponseState(
                    text=response_text,
                    payload=response_payload,
                )
            }
        )

    def with_self_check_result(self, result: SelfCheckResult) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self._with_results(update={"self_check_result": result})

    def with_metadata(self, **metadata: Any) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        return self.model_copy(
            update={
                "technical": self.technical.model_copy(
                    update={"metadata": {**self.metadata, **metadata}}
                )
            }
        )

    def with_stage_metadata(self, stage_name: str, **metadata: Any) -> "AIContext":
        """Возвращает новую версию структуры данных с добавленным результатом. Так конвейер не теряет предыдущие детали и аккуратно дополняет контекст."""
        stage_name = stage_name.strip()
        if not stage_name:
            raise ValueError("stage_name must not be empty")

        existing = self.stage_metadata.get(stage_name, {})
        return self.model_copy(
            update={
                "technical": self.technical.model_copy(
                    update={
                        "stage_metadata": {
                            **self.stage_metadata,
                            stage_name: {**existing, **metadata},
                        }
                    }
                )
            }
        )

    def _with_results(self, update: dict[str, Any]) -> "AIContext":
        """Выполняет шаг «with results». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.model_copy(
            update={"results": self.results.model_copy(update=update)}
        )
