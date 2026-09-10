import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import Settings
from app.llm.client import (
    LLMClient,
    LLMProviderError,
    LLMStructuredOutputError,
    Message,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 402, 403, 404, 422}


class OpenRouterLLMClient(LLMClient):
    """Класс «OpenRouterLLMClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, settings: Settings) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        self._model = settings.llm_model
        # Параметры генерации нельзя задать в конструкторе OpenAI: он принимает только
        # транспортные настройки. Поэтому держим их здесь и подмешиваем в каждый запрос.
        self._generation_defaults = self._build_generation_defaults(settings)
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_transport_retries,
        )

    @staticmethod
    def _build_generation_defaults(settings: Settings) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        defaults: dict[str, Any] = {"temperature": settings.llm_temperature}
        if settings.llm_max_tokens is not None:
            defaults["max_tokens"] = settings.llm_max_tokens
        if settings.llm_top_p is not None:
            defaults["top_p"] = settings.llm_top_p

        return defaults

    def _merge_generation_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        # Настройки задают базу, аргументы конкретного вызова ее перекрывают.
        return {**self._generation_defaults, **kwargs}

    def generate(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> str:
        """Выполняет шаг «generate». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        response = self._create_completion(messages, **kwargs)
        content = response.choices[0].message.content
        return content or ""

    def _create_completion(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> Any:
        try:
            return self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                **self._merge_generation_kwargs(kwargs),
            )
        except APITimeoutError as exc:
            raise LLMProviderError(
                f"LLM provider request timed out: {exc}",
                error_code=type(exc).__name__,
                retryable=True,
            ) from exc
        except APIConnectionError as exc:
            raise LLMProviderError(
                f"LLM provider connection failed: {exc}",
                error_code=type(exc).__name__,
                retryable=True,
            ) from exc
        except APIStatusError as exc:
            status_code = exc.status_code
            error_code = self._extract_error_code(exc)
            retryable = self._is_retryable_status(status_code)
            raise LLMProviderError(
                f"LLM provider returned status {status_code}: {exc}",
                status_code=status_code,
                error_code=error_code,
                retryable=retryable,
            ) from exc

    def generate_json(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Выполняет шаг «generate json». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        response_format = kwargs.pop("response_format", {"type": "json_object"})
        response = self._create_completion(
            messages,
            response_format=response_format,
            **kwargs,
        )
        choice = response.choices[0]
        response_text = choice.message.content or ""
        provider_metadata = self._extract_response_metadata(response, choice)
        content_length = len(response_text)

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            error_kind = self._classify_json_failure(
                response_text=response_text,
                decode_error=exc,
                finish_reason=provider_metadata.get("finish_reason"),
            )
            logger.exception(
                "LLM response JSON parsing failed: error_type=%s error=%s error_kind=%s response_length=%s finish_reason=%s response_model=%s response_id=%s usage=%r raw_response_text=%r",
                type(exc).__name__,
                str(exc),
                error_kind,
                content_length,
                provider_metadata.get("finish_reason"),
                provider_metadata.get("model"),
                provider_metadata.get("id"),
                provider_metadata.get("usage"),
                response_text,
            )
            raise LLMStructuredOutputError(
                "LLM response is not valid JSON",
                error_kind=error_kind,
                provider_metadata=provider_metadata,
                content_length=content_length,
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMStructuredOutputError(
                "LLM JSON response must be an object",
                error_kind="malformed_json",
                provider_metadata=provider_metadata,
                content_length=content_length,
            )

        return parsed

    def stream(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Выполняет шаг «stream». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            stream=True,
            **self._merge_generation_kwargs(kwargs),
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    @staticmethod
    def _is_retryable_status(status_code: int | None) -> bool:
        if status_code in _NON_RETRYABLE_STATUS_CODES:
            return False
        if status_code in _RETRYABLE_STATUS_CODES:
            return True
        return True

    @staticmethod
    def _extract_error_code(exc: APIStatusError) -> str | None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                if code is not None:
                    return str(code)
            code = body.get("code")
            if code is not None:
                return str(code)
        return type(exc).__name__

    @classmethod
    def _classify_json_failure(
        cls,
        *,
        response_text: str,
        decode_error: json.JSONDecodeError,
        finish_reason: object,
    ) -> str:
        if finish_reason == "length":
            return "length_finish"
        if not response_text:
            return "empty_content"
        if cls._looks_truncated_json(response_text, decode_error):
            return "truncated_json"
        return "malformed_json"

    @staticmethod
    def _looks_truncated_json(
        response_text: str,
        decode_error: json.JSONDecodeError,
    ) -> bool:
        message = decode_error.msg.lower()
        stripped = response_text.rstrip()
        if "unterminated string" in message:
            return True
        if not stripped:
            return False
        if stripped[0] in ("{", "[") and stripped[-1] not in ("}", "]"):
            return True
        return False

    @staticmethod
    def _extract_response_metadata(response: Any, choice: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "finish_reason": getattr(choice, "finish_reason", None),
        }
        for key in ("id", "model", "created", "system_fingerprint"):
            value = getattr(response, key, None)
            if value is not None:
                metadata[key] = value

        usage = getattr(response, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                metadata["usage"] = usage.model_dump(mode="json")
            elif isinstance(usage, dict):
                metadata["usage"] = dict(usage)
            else:
                metadata["usage"] = repr(usage)

        return metadata
