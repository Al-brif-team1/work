"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from app.llm.runner import LLMRunResult
from app.llm.runner import LLMRunner
from app.pipeline.base import BaseLLMStage
from app.prompts import PromptManager, RenderedPrompt
from app.schemas import (
    AIContext,
    BriefInput,
    ExtractionResult,
    ExtractedBrief,
    ExtractedFact,
    ExtractorTechnicalInfo,
)
from app.tracing.tracing import TracingClient

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class ExtractorError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class Extractor(BaseLLMStage[BriefInput, ExtractedBrief, ExtractionResult]):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Переводчик-Экстрактор. Он переводит бриф с человеческого языка на язык структурированных данных и извлекает факты без искажения смысла. Этот этап обращается к LLM, поэтому внутри работает ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseLLMStage, поэтому он умеет работать в нашем конвейере."""

    output_model: ClassVar[type[ExtractedBrief]] = ExtractedBrief

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

    def extract(self, brief_input: BriefInput) -> ExtractionResult:
        """Выполняет шаг «extract». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self.run(brief_input)

    def extract_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
        return context.with_extraction_result(self.extract(context.brief_input))

    def run_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
        return self.extract_context(context)

    @property
    def trace_name(self) -> str:
        """Выполняет шаг «trace name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return "extractor.brief"

    @property
    def span_name(self) -> str:
        """Выполняет шаг «span name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return "extractor.llm"

    def build_prompt(self, stage_input: BriefInput) -> str:
        """Выполняет шаг «build prompt». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        rendered = self._render_extraction_prompt(stage_input)
        return rendered.user or rendered.system

    def build_system_prompt(self, stage_input: BriefInput) -> str | None:
        """Выполняет шаг «build system prompt». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        rendered = self._render_extraction_prompt(stage_input)
        return rendered.system if rendered.user is not None else None

    def build_context(self, stage_input: BriefInput) -> None:
        """Выполняет шаг «build context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return None

    def build_trace_input(self, stage_input: BriefInput) -> dict[str, str | None]:
        """Выполняет шаг «build trace input». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return {
            "source": stage_input.metadata.source,
            "input_type": stage_input.metadata.input_type,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
        }

    def postprocess(
        self,
        result: LLMRunResult[ExtractedBrief],
    ) -> ExtractionResult:
        """Выполняет шаг «postprocess». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        extracted_brief = self._normalize_extracted_brief(result.payload)
        return ExtractionResult(
            extracted_brief=extracted_brief,
            technical_info=ExtractorTechnicalInfo(
                attempts=result.attempts,
                prompt_name=self.prompt_name,
                trace_enabled=result.trace_enabled,
                trace_name=result.trace_name or self.trace_name,
                model_name=result.model_name,
                raw_response=result.raw_response,
                recovered_errors=list(result.recovered_errors),
            ),
        )

    @staticmethod
    def _default_prompt_path() -> Path:
        """Возвращает значение по умолчанию, чтобы этап мог работать без ручной настройки."""
        return Path(__file__).resolve().parents[2] / "prompts" / "extractor.md"

    def _build_failure_exception(
        self,
        attempts: int,
        last_error: Exception | None,
    ) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return ExtractorError(
            f"Unable to extract brief structure after {attempts} attempts"
        )

    def _render_extraction_prompt(self, brief_input: BriefInput) -> RenderedPrompt:
        """Готовит человекочитаемый текст из внутренних данных. Это нужно для промптов, объяснений или финального ответа."""
        return self._render_prompt(
            {
                "brief_text": brief_input.normalized_text,
            }
        )

    def _normalize_extracted_brief(self, extracted_brief: ExtractedBrief) -> ExtractedBrief:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        return extracted_brief.model_copy(
            update={
                "project_goal": self._normalize_fact(extracted_brief.project_goal),
                "tasks": self._normalize_facts(extracted_brief.tasks),
                "project_type": self._normalize_fact(extracted_brief.project_type),
                "project_direction": self._normalize_fact(
                    extracted_brief.project_direction
                ),
                "technologies": self._normalize_facts(extracted_brief.technologies),
                "stack": self._normalize_facts(extracted_brief.stack),
                "materials": self._normalize_facts(extracted_brief.materials),
                "expected_result": self._normalize_fact(
                    extracted_brief.expected_result
                ),
                "constraints": self._normalize_facts(extracted_brief.constraints),
                "deadlines": self._normalize_facts(extracted_brief.deadlines),
                "existing_resources": self._normalize_facts(
                    extracted_brief.existing_resources
                ),
                "integrations": self._normalize_facts(extracted_brief.integrations),
                "other_facts": self._normalize_facts(extracted_brief.other_facts),
            }
        )

    @classmethod
    def _normalize_facts(cls, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        return [cls._normalize_fact(fact) for fact in facts]

    @staticmethod
    def _normalize_fact(fact: ExtractedFact) -> ExtractedFact:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        value = fact.value.strip() if fact.value is not None else None
        notes = fact.notes.strip() if fact.notes is not None else None
        return fact.model_copy(
            update={
                "value": value or None,
                "evidence": [
                    item.strip()
                    for item in fact.evidence
                    if item and item.strip()
                ],
                "notes": notes or None,
            }
        )
