import json
from collections.abc import Iterable, Sequence
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.llm.client import LLMClient, Message


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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            **self._merge_generation_kwargs(kwargs),
        )

        content = response.choices[0].message.content
        return content or ""

    def generate_json(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Выполняет шаг «generate json». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        response_format = kwargs.pop("response_format", {"type": "json_object"})
        response_text = self.generate(
            messages=messages,
            response_format=response_format,
            **kwargs,
        )

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM response is not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("LLM JSON response must be an object")

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
