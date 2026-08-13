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
        self._model = settings.openrouter_model
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def generate(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> str:
        """Выполняет шаг «generate». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            **kwargs,
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
            **kwargs,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
