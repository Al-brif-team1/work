"""Модуль наблюдаемости. Он собирает технические следы выполнения, чтобы было проще понять, какой робот что сделал."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, ContextManager, Protocol

from langfuse import Langfuse

from app.config import Settings, get_settings


class TraceContext(Protocol):
    """Класс «TraceContext» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def update(self, **kwargs: Any) -> None:
        """Выполняет шаг «update». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...


class TracingClient(Protocol):
    """Класс «TracingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def create_trace(
        self,
        name: str,
        **kwargs: Any,
    ) -> ContextManager[TraceContext | None]:
        """Выполняет шаг «create trace». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...

    def create_span(
        self,
        name: str,
        **kwargs: Any,
    ) -> ContextManager[TraceContext | None]:
        """Выполняет шаг «create span». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...

    def flush(self) -> None:
        """Выполняет шаг «flush». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        ...


class LangfuseTracingClient:
    """Класс «LangfuseTracingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, settings: Settings) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_host,
        )

    @contextmanager
    def create_trace(
        self,
        name: str,
        **kwargs: Any,
    ) -> Iterator[TraceContext | None]:
        """Выполняет шаг «create trace». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        with self._client.start_as_current_observation(
            as_type="chain",
            name=name,
            **kwargs,
        ) as trace:
            yield trace

    @contextmanager
    def create_span(
        self,
        name: str,
        **kwargs: Any,
    ) -> Iterator[TraceContext | None]:
        """Выполняет шаг «create span». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        with self._client.start_as_current_observation(
            as_type="span",
            name=name,
            **kwargs,
        ) as span:
            yield span

    def flush(self) -> None:
        """Выполняет шаг «flush». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        self._client.flush()


class NoOpTracingClient:
    """Класс «NoOpTracingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    @contextmanager
    def create_trace(
        self,
        name: str,
        **kwargs: Any,
    ) -> Iterator[TraceContext | None]:
        """Выполняет шаг «create trace». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        yield None

    @contextmanager
    def create_span(
        self,
        name: str,
        **kwargs: Any,
    ) -> Iterator[TraceContext | None]:
        """Выполняет шаг «create span». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        yield None

    def flush(self) -> None:
        """Выполняет шаг «flush». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return None


class TracingClientFactory:
    """Класс «TracingClientFactory» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    @staticmethod
    def create(settings: Settings) -> TracingClient:
        """Выполняет шаг «create». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            return NoOpTracingClient()

        return LangfuseTracingClient(settings=settings)


@lru_cache(maxsize=1)
def get_tracing_client() -> TracingClient:
    """Возвращает уже подготовленный объект или настройку, чтобы остальные части проекта использовали единый источник."""
    return TracingClientFactory.create(settings=get_settings())
