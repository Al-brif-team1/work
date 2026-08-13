"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, ClassVar, Generic, Sequence, TypeVar, cast

from pydantic import BaseModel

from app.llm.client import LLMClient, Message
from app.llm.runner import LLMRunResult, LLMRunner, LLMRunnerError, LLMTokenUsage
from app.pipeline.contracts import BaseStage
from app.prompts import PromptManager, RenderedPrompt, get_prompt_manager
from app.tracing.tracing import TracingClient, get_tracing_client

TPayload = TypeVar("TPayload", bound=BaseModel)
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass(frozen=True)
class LLMStageRunResult(Generic[TPayload]):
    """Класс «LLMStageRunResult» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    payload: TPayload
    raw_response: dict[str, Any]
    attempts: int
    recovered_errors: list[str]
    prompt_name: str
    trace_name: str
    trace_enabled: bool
    model_name: str | None
    latency_seconds: float
    token_usage: LLMTokenUsage
    provider_metadata: dict[str, Any]
    trace_id: str | None = None
    llm_invoked: bool = True

    def technical_kwargs(self) -> dict[str, Any]:
        """Выполняет шаг «technical kwargs». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return {
            "attempts": self.attempts,
            "prompt_name": self.prompt_name,
            "trace_enabled": self.trace_enabled,
            "trace_name": self.trace_name,
            "model_name": self.model_name,
            "raw_response": self.raw_response,
            "recovered_errors": list(self.recovered_errors),
        }


class BaseLLMStage(BaseStage[TInput, TOutput], ABC, Generic[TInput, TPayload, TOutput]):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Робот этапа. Он выполняет участок конвейера «base l l m stage». Этот этап работает как детерминированный робот: обычный код, без творческих догадок ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseStage, поэтому он умеет работать в нашем конвейере."""

    output_model: ClassVar[type[BaseModel] | None] = None

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
        timeout_seconds: float | None = 60.0,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than zero")
        if llm_client is None and llm_runner is None:
            raise ValueError("Pass either llm_client or llm_runner")

        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or get_tracing_client(),
        )
        default_prompt_path = self._default_prompt_path()
        effective_prompt_path = Path(prompt_path) if prompt_path is not None else default_prompt_path
        effective_prompt_name = prompt_name or effective_prompt_path.name
        effective_prompt_directories = (
            (effective_prompt_path.parent,)
            if prompt_path is not None
            else (default_prompt_path.parent,)
        )
        self._prompt_manager = prompt_manager or get_prompt_manager(
            tuple(effective_prompt_directories)
        )
        self._prompt = self._prompt_manager.load(
            effective_prompt_name,
            version=prompt_version,
        )
        self._max_retries = max_retries
        self._model_name = model_name
        if llm_runner is not None:
            self._llm_runner = llm_runner
        else:
            self._llm_runner = LLMRunner(
                llm_client=cast(LLMClient, llm_client),
                tracing_client=self._tracing_client,
                logger=self._logger,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                model_name=model_name,
            )

    def _run(self, stage_input: TInput) -> TOutput:
        """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""
        output_model = self._resolve_output_model()
        try:
            llm_result = self._llm_runner.run(
                prompt=self.build_prompt(stage_input),
                output_model=output_model,
                context=self.build_context(stage_input),
                system_prompt=self.build_system_prompt(stage_input),
                trace_name=self.trace_name,
                span_name=self.span_name,
                trace_input=self.build_trace_input(stage_input),
                payload_validator=self.validate_payload,
                temperature=self.temperature,
                request_kwargs=self.build_request_kwargs(stage_input),
            )
        except LLMRunnerError as exc:
            raise self._build_failure_exception(self._max_retries, exc) from exc

        try:
            return self.postprocess(cast(LLMRunResult[TPayload], llm_result))
        except Exception as exc:
            raise self._build_failure_exception(
                llm_result.attempts,
                exc,
            ) from exc

    @property
    def trace_name(self) -> str:
        """Выполняет шаг «trace name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return f"{self.__class__.__name__}.run"

    @property
    def span_name(self) -> str:
        """Выполняет шаг «span name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return f"{self.__class__.__name__}.llm"

    @property
    def temperature(self) -> float:
        """Выполняет шаг «temperature». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return 0

    def build_prompt(self, stage_input: TInput) -> str:
        """Выполняет шаг «build prompt». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_prompt()"
        )

    def build_context(
        self,
        stage_input: TInput,
    ) -> str | Mapping[str, Any] | BaseModel | None:
        """Выполняет шаг «build context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return stage_input if isinstance(stage_input, (str, BaseModel)) else None

    def build_system_prompt(self, stage_input: TInput) -> str | None:
        """Выполняет шаг «build system prompt». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return None

    def build_trace_input(self, stage_input: TInput) -> dict[str, Any]:
        """Выполняет шаг «build trace input». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "input_type": type(stage_input).__name__,
        }

    def build_request_kwargs(self, stage_input: TInput) -> dict[str, Any] | None:
        """Выполняет шаг «build request kwargs». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return None

    def validate_payload(self, payload: TPayload) -> None:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        return None

    def postprocess(self, result: LLMRunResult[TPayload]) -> TOutput:
        """Выполняет шаг «postprocess». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return cast(TOutput, result.payload)

    def _resolve_output_model(self) -> type[TPayload]:
        """Находит нужное поле внутри вложенной структуры данных. Это похоже на движение по адресу: шаг за шагом до конкретного значения."""
        if self.output_model is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define output_model"
            )
        return cast(type[TPayload], self.output_model)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return exc

    @property
    def prompt_path(self) -> Path:
        """Выполняет шаг «prompt path». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._prompt.path

    @property
    def prompt_name(self) -> str:
        """Выполняет шаг «prompt name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._prompt.name

    @property
    def prompt_version(self) -> str | None:
        """Выполняет шаг «prompt version». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._prompt.version

    @property
    def prompt_template(self) -> str:
        """Выполняет шаг «prompt template». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._prompt.content

    def _render_prompt(
        self,
        variables: dict[str, Any] | None = None,
    ) -> RenderedPrompt:
        """Готовит человекочитаемый текст из внутренних данных. Это нужно для промптов, объяснений или финального ответа."""
        return self._prompt_manager.render(
            self.prompt_name,
            version=self.prompt_version,
            variables=variables or {},
        )

    @property
    def model_name(self) -> str | None:
        """Выполняет шаг «model name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._model_name

    @property
    def max_retries(self) -> int:
        """Выполняет шаг «max retries». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._max_retries

    def _execute_structured_stage(
        self,
        *,
        trace_name: str,
        span_name: str,
        trace_input: dict[str, Any],
        messages: Sequence[Message],
        response_model: type[TPayload],
        payload_validator: Callable[[TPayload], None] | None = None,
    ) -> LLMStageRunResult[TPayload]:
        """Выполняет шаг «execute structured stage». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        try:
            runner_result = self._llm_runner.run_json(
                trace_name=trace_name,
                span_name=span_name,
                trace_input={
                    **trace_input,
                    "prompt_name": self.prompt_name,
                    "prompt_version": self.prompt_version,
                },
                messages=messages,
                response_model=response_model,
                payload_validator=payload_validator,
                temperature=0,
            )
        except Exception as exc:
            raise self._build_failure_exception(
                self._max_retries,
                exc,
            ) from exc

        return LLMStageRunResult(
            payload=runner_result.payload,
            raw_response=runner_result.raw_response,
            attempts=runner_result.attempts,
            recovered_errors=runner_result.recovered_errors,
            prompt_name=self.prompt_name,
            trace_name=runner_result.trace_name or trace_name,
            trace_enabled=runner_result.trace_enabled,
            model_name=runner_result.model_name,
            latency_seconds=runner_result.latency_seconds,
            token_usage=runner_result.token_usage,
            provider_metadata=runner_result.provider_metadata,
            trace_id=runner_result.trace_id,
        )

    @staticmethod
    @abstractmethod
    def _default_prompt_path() -> Path:
        """Возвращает значение по умолчанию, чтобы этап мог работать без ручной настройки."""

    @abstractmethod
    def _build_failure_exception(
        self,
        attempts: int,
        last_error: Exception | None,
    ) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
