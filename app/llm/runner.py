"""Модуль инфраструктуры LLM. Он отделяет работу с ИИ-моделью от бизнес-логики, чтобы роботы конвейера получали ответы единым способом."""

from __future__ import annotations

import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections.abc import Mapping
from typing import Any, Callable, Generic, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.client import LLMClient, Message
from app.tracing.tracing import NoOpTracingClient, TracingClient, get_tracing_client

TPayload = TypeVar("TPayload", bound=BaseModel)

_JSON_OBJECT_INSTRUCTION = "Return only a valid JSON object."
_RUSSIAN_LANGUAGE_INSTRUCTION = (
    "All human-readable string values in the JSON response must be written in Russian."
)


class LLMTokenUsage(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_estimate: int | None = None
    completion_tokens_estimate: int | None = None
    total_tokens_estimate: int | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMRunResult(BaseModel, Generic[TPayload]):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    payload: TPayload
    raw_response: dict[str, Any]
    raw_text: str | None = None
    attempts: int
    latency_seconds: float
    token_usage: LLMTokenUsage
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    trace_name: str | None = None
    trace_enabled: bool = False
    model_name: str | None = None
    recovered_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMRunnerError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class LLMRunnerTimeoutError(LLMRunnerError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class LLMRunnerProviderError(LLMRunnerError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class LLMRunnerStructuredOutputError(LLMRunnerError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class LLMRunner:
    """Класс «LLMRunner» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(
        self,
        llm_client: LLMClient,
        tracing_client: TracingClient | None = None,
        logger: logging.Logger | None = None,
        max_retries: int = 2,
        timeout_seconds: float | None = 60.0,
        model_name: str | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than zero")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._llm_client = llm_client
        self._tracing_client = tracing_client or get_tracing_client()
        self._logger = logger or logging.getLogger(__name__)
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._model_name = model_name

    def run_json(
        self,
        *,
        messages: Sequence[Message],
        response_model: type[TPayload],
        trace_name: str,
        span_name: str,
        trace_input: dict[str, Any] | None = None,
        payload_validator: Callable[[TPayload], None] | None = None,
        temperature: float = 0,
        request_kwargs: dict[str, Any] | None = None,
    ) -> LLMRunResult[TPayload]:
        """Выполняет шаг «run json». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        trace_input = trace_input or {}
        request_kwargs = request_kwargs or {}
        normalized_messages = self._ensure_json_instruction(
            tuple(messages),
            response_model=response_model,
        )
        trace_enabled = not isinstance(self._tracing_client, NoOpTracingClient)
        recovered_errors: list[str] = []
        last_error: Exception | None = None
        started_at = time.perf_counter()
        trace_id: str | None = None

        with self._tracing_client.create_trace(
            trace_name,
            input={
                **trace_input,
                "model_name": self._model_name,
                "timeout_seconds": self._timeout_seconds,
            },
        ) as trace:
            if trace is not None:
                trace_id = self._extract_trace_id(trace)
                trace.update(
                    input={
                        **trace_input,
                        "model_name": self._model_name,
                        "timeout_seconds": self._timeout_seconds,
                    }
                )

            for attempt in range(1, self._max_retries + 1):
                try:
                    with self._tracing_client.create_span(
                        span_name,
                        input={
                            "attempt": attempt,
                            "model_name": self._model_name,
                            "timeout_seconds": self._timeout_seconds,
                        },
                    ) as span:
                        raw_response = self._call_generate_json(
                            messages=normalized_messages,
                            temperature=temperature,
                            request_kwargs=self._with_json_schema_response_format(
                                request_kwargs,
                                response_model,
                            ),
                        )
                        payload = self._validate_payload(
                            raw_response=raw_response,
                            response_model=response_model,
                            payload_validator=payload_validator,
                        )
                        latency_seconds = time.perf_counter() - started_at
                        token_usage = self._build_token_usage(
                            messages=normalized_messages,
                            raw_response=raw_response,
                        )
                        provider_metadata = self._build_provider_metadata(raw_response)

                        if span is not None:
                            span.update(
                                output={
                                    "status": "success",
                                    "attempt": attempt,
                                    "latency_seconds": latency_seconds,
                                    "token_usage": token_usage.model_dump(
                                        mode="json"
                                    ),
                                }
                            )
                        if trace is not None:
                            trace.update(
                                output={
                                    "attempts": attempt,
                                    "status": "success",
                                    "latency_seconds": latency_seconds,
                                }
                            )

                        self._logger.info(
                            "%s succeeded on attempt %s",
                            trace_name,
                            attempt,
                        )
                        return LLMRunResult(
                            payload=payload,
                            raw_response=raw_response,
                            attempts=attempt,
                            latency_seconds=latency_seconds,
                            token_usage=token_usage,
                            provider_metadata=provider_metadata,
                            trace_id=trace_id,
                            trace_name=trace_name,
                            trace_enabled=trace_enabled,
                            model_name=self._model_name,
                            recovered_errors=recovered_errors,
                        )
                except LLMRunnerError as exc:
                    last_error = exc
                    recovered_errors.append(str(exc))
                    self._logger.warning(
                        "%s attempt %s failed: %s",
                        trace_name,
                        attempt,
                        exc,
                    )
                    if trace is not None:
                        trace.update(
                            output={
                                "attempt": attempt,
                                "status": (
                                    "retrying"
                                    if attempt < self._max_retries
                                    else "failed"
                                ),
                                "error": str(exc),
                            }
                        )

        raise LLMRunnerProviderError(
            f"LLM request failed after {self._max_retries} attempts"
        ) from last_error

    def run(
        self,
        *,
        prompt: str,
        output_model: type[TPayload],
        context: str | Mapping[str, Any] | BaseModel | None = None,
        system_prompt: str | None = None,
        trace_name: str = "llm.run",
        span_name: str = "llm.call",
        trace_input: dict[str, Any] | None = None,
        payload_validator: Callable[[TPayload], None] | None = None,
        temperature: float = 0,
        request_kwargs: dict[str, Any] | None = None,
    ) -> LLMRunResult[TPayload]:
        """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""
        messages = self._build_messages(
            prompt=prompt,
            context=context,
            system_prompt=system_prompt,
        )
        return self.run_json(
            messages=messages,
            response_model=output_model,
            trace_name=trace_name,
            span_name=span_name,
            trace_input=trace_input,
            payload_validator=payload_validator,
            temperature=temperature,
            request_kwargs=request_kwargs,
        )

    def _call_generate_json(
        self,
        *,
        messages: Sequence[Message],
        temperature: float,
        request_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Выполняет шаг «call generate json». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        kwargs = {"temperature": temperature, **request_kwargs}
        if self._timeout_seconds is not None and "timeout" not in kwargs:
            kwargs["timeout"] = self._timeout_seconds

        try:
            if self._timeout_seconds is None:
                return self._llm_client.generate_json(messages, **kwargs)

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                self._llm_client.generate_json,
                messages,
                **kwargs,
            )
            try:
                return future.result(timeout=self._timeout_seconds)
            finally:
                executor.shutdown(wait=future.done(), cancel_futures=not future.done())
        except FutureTimeoutError as exc:
            raise LLMRunnerTimeoutError(
                f"LLM request timed out after {self._timeout_seconds} seconds"
            ) from exc
        except Exception as exc:
            if isinstance(exc, LLMRunnerError):
                raise
            raise LLMRunnerProviderError(
                f"LLM provider request failed: {exc}"
            ) from exc

    @staticmethod
    def _with_json_schema_response_format(
        request_kwargs: dict[str, Any],
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        """Выполняет шаг «with json schema response format». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if "response_format" in request_kwargs:
            return dict(request_kwargs)

        schema = response_model.model_json_schema()
        return {
            **request_kwargs,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    @classmethod
    def _build_messages(
        cls,
        *,
        prompt: str,
        context: str | Mapping[str, Any] | BaseModel | None,
        system_prompt: str | None,
    ) -> tuple[Message, ...]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        if system_prompt is not None:
            system_prompt = system_prompt.strip()
            if not system_prompt:
                raise ValueError("system_prompt must not be empty")

        if context is None:
            if system_prompt is None:
                return ({"role": "user", "content": prompt},)
            return (
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            )

        serialized_context = cls._serialize_context(context)
        if system_prompt is None:
            return (
                {"role": "system", "content": prompt},
                {"role": "user", "content": serialized_context},
            )
        return (
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{prompt}\n\n{serialized_context}",
            },
        )

    @staticmethod
    def _serialize_context(context: str | Mapping[str, Any] | BaseModel) -> str:
        """Выполняет шаг «serialize context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if isinstance(context, str):
            serialized = context.strip()
            if not serialized:
                raise ValueError("context must not be empty")
            return serialized

        if isinstance(context, BaseModel):
            return context.model_dump_json()

        return json.dumps(dict(context), ensure_ascii=False, indent=2)

    @staticmethod
    def _ensure_json_instruction(
        messages: Sequence[Message],
        *,
        response_model: type[BaseModel],
    ) -> tuple[Message, ...]:
        """Выполняет шаг «ensure json instruction». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        schema_instruction = (
            f"{_JSON_OBJECT_INSTRUCTION}\n"
            f"{_RUSSIAN_LANGUAGE_INSTRUCTION}\n"
            "The JSON object must match this JSON Schema exactly:\n"
            f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False)}"
        )
        if any(
            "json schema" in message.get("content", "").lower()
            for message in messages
        ):
            return tuple(messages)

        if not messages:
            return ({"role": "system", "content": schema_instruction},)

        first, *rest = messages
        first_content = first.get("content", "").strip()
        merged_first = {
            **dict(first),
            "content": (
                f"{first_content}\n\n{schema_instruction}"
                if first_content
                else schema_instruction
            ),
        }
        return (merged_first, *rest)

    @staticmethod
    def _validate_payload(
        *,
        raw_response: dict[str, Any],
        response_model: type[TPayload],
        payload_validator: Callable[[TPayload], None] | None,
    ) -> TPayload:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        try:
            payload = response_model.model_validate(raw_response)
            if payload_validator is not None:
                payload_validator(payload)
            return payload
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            raise LLMRunnerStructuredOutputError(
                f"LLM structured output validation failed: {exc}"
            ) from exc

    @staticmethod
    def _build_token_usage(
        *,
        messages: Sequence[Message],
        raw_response: dict[str, Any],
    ) -> LLMTokenUsage:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        usage = raw_response.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = _optional_int(usage.get("prompt_tokens"))
            completion_tokens = _optional_int(usage.get("completion_tokens"))
            total_tokens = _optional_int(usage.get("total_tokens"))
            return LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                prompt_tokens_estimate=_estimate_text_tokens(
                    "\n".join(message.get("content", "") for message in messages)
                ),
                completion_tokens_estimate=_estimate_text_tokens(str(raw_response)),
                total_tokens_estimate=None,
            )

        prompt_estimate = _estimate_text_tokens(
            "\n".join(message.get("content", "") for message in messages)
        )
        completion_estimate = _estimate_text_tokens(str(raw_response))
        return LLMTokenUsage(
            prompt_tokens_estimate=prompt_estimate,
            completion_tokens_estimate=completion_estimate,
            total_tokens_estimate=prompt_estimate + completion_estimate,
        )

    @staticmethod
    def _build_provider_metadata(raw_response: dict[str, Any]) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        metadata: dict[str, Any] = {}
        for key in ("id", "model", "provider", "created", "system_fingerprint"):
            if key in raw_response:
                metadata[key] = raw_response[key]
        return metadata

    @staticmethod
    def _extract_trace_id(trace: object) -> str | None:
        """Выполняет шаг «extract trace id». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        for attribute in ("id", "trace_id", "observation_id"):
            value = getattr(trace, attribute, None)
            if value:
                return str(value)
        return None


def _estimate_text_tokens(text: str) -> int:
    """Выполняет шаг «estimate text tokens». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _optional_int(value: object) -> int | None:
    """Выполняет шаг «optional int». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
