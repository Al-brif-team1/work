"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.config import CriteriaConfig, get_criteria_config
from app.input import BriefInputFactory
from app.llm.client import LLMClient
from app.llm.runner import LLMRunner
from app.pipeline.arbiter import DeterministicArbiterStage
from app.pipeline.assessment import AssessmentRetriever, AssessmentStage
from app.pipeline.completeness import CompletenessCheckStage
from app.pipeline.extractor import Extractor
from app.pipeline.mvp_planner import MVPPlannerStage
from app.pipeline.question_generator import TemplateQuestionGeneratorStage
from app.pipeline.response_writer import ResponseWriterStage
from app.prompts import PromptManager
from app.schemas import AIContext, BriefAnalysisResult, BriefInput
from app.tracing.tracing import TracingClient, get_tracing_client


class ContextStage(Protocol):
    """Класс «ContextStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def run_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""


class BriefAnalysisPipelineError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class BriefAnalysisPipeline:
    """Класс «BriefAnalysisPipeline» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(
        self,
        *,
        stages: Sequence[ContextStage] | None = None,
        criteria_config: CriteriaConfig | None = None,
        input_factory: BriefInputFactory | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        self._criteria_config = criteria_config or get_criteria_config()
        self._input_factory = input_factory or BriefInputFactory()
        self._stages = list(stages) if stages is not None else []

    @classmethod
    def from_llm_client(
        cls,
        llm_client: LLMClient,
        *,
        retriever: AssessmentRetriever | None = None,
        criteria_config: CriteriaConfig | None = None,
        tracing_client: TracingClient | None = None,
        prompt_manager: PromptManager | None = None,
        max_retries: int = 2,
        timeout_seconds: float | None = 60.0,
        model_name: str | None = None,
        input_factory: BriefInputFactory | None = None,
    ) -> "BriefAnalysisPipeline":
        """Выполняет шаг «from llm client». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        tracing = tracing_client or get_tracing_client()
        config = criteria_config or get_criteria_config()
        prompts = prompt_manager or PromptManager()
        llm_runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=tracing,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
        )
        return cls(
            stages=[
                Extractor(
                    llm_runner=llm_runner,
                    tracing_client=tracing,
                    prompt_manager=prompts,
                    max_retries=max_retries,
                    model_name=model_name,
                ),
                CompletenessCheckStage(
                    criteria_config=config,
                    tracing_client=tracing,
                ),
                AssessmentStage(
                    llm_runner=llm_runner,
                    tracing_client=tracing,
                    prompt_manager=prompts,
                    max_retries=max_retries,
                    model_name=model_name,
                    retriever=retriever,
                    criteria_config=config,
                ),
                DeterministicArbiterStage(
                    criteria_config=config,
                    tracing_client=tracing,
                ),
                # Этот этап детерминированный: он работает без ИИ и берет шаблоны вопросов из question_templates.json.
                TemplateQuestionGeneratorStage(tracing_client=tracing),
                MVPPlannerStage(
                    llm_runner=llm_runner,
                    tracing_client=tracing,
                    prompt_manager=prompts,
                    max_retries=max_retries,
                    model_name=model_name,
                ),
                ResponseWriterStage(tracing_client=tracing),
            ],
            criteria_config=config,
            input_factory=input_factory,
        )

    def analyze_text(self, text: str) -> BriefAnalysisResult:
        """Выполняет шаг «analyze text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.analyze(self._input_factory.from_text(text))

    def analyze(self, brief_input: BriefInput) -> BriefAnalysisResult:
        """Выполняет шаг «analyze». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        context = self.run_context(brief_input)
        if context.final_response_payload is None:
            raise BriefAnalysisPipelineError("Pipeline did not produce final payload")
        return BriefAnalysisResult.model_validate(context.final_response_payload)

    def run_context(self, brief_input: BriefInput) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
        context = AIContext.from_brief(
            brief_input,
            configuration=self._criteria_config,
        )
        for stage in self._stages:
            context = stage.run_context(context)
        return context
