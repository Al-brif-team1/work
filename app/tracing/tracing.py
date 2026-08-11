"""Tracing abstractions and Langfuse integration."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, ContextManager, Protocol

from langfuse import Langfuse

from app.config import Settings, get_settings


class TraceContext(Protocol):
    """Minimal trace/span context exposed to application code."""

    def update(self, **kwargs: Any) -> None:
        """Update trace or span attributes."""
        ...


class TracingClient(Protocol):
    """Provider-independent tracing interface."""

    def create_trace(
        self,
        name: str,
        **kwargs: Any,
    ) -> ContextManager[TraceContext | None]:
        """Create a trace context."""
        ...

    def create_span(
        self,
        name: str,
        **kwargs: Any,
    ) -> ContextManager[TraceContext | None]:
        """Create a span context."""
        ...

    def flush(self) -> None:
        """Flush buffered tracing events."""
        ...


class LangfuseTracingClient:
    """Langfuse-backed implementation of the tracing interface."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the Langfuse client from application settings."""
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
        """Create a top-level Langfuse observation context."""
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
        """Create a nested Langfuse span context."""
        with self._client.start_as_current_observation(
            as_type="span",
            name=name,
            **kwargs,
        ) as span:
            yield span

    def flush(self) -> None:
        """Flush buffered Langfuse events."""
        self._client.flush()


class NoOpTracingClient:
    """Tracing client used when Langfuse credentials are not configured."""

    @contextmanager
    def create_trace(
        self,
        name: str,
        **kwargs: Any,
    ) -> Iterator[TraceContext | None]:
        """Create an inert trace context."""
        yield None

    @contextmanager
    def create_span(
        self,
        name: str,
        **kwargs: Any,
    ) -> Iterator[TraceContext | None]:
        """Create an inert span context."""
        yield None

    def flush(self) -> None:
        """No-op flush for disabled tracing."""
        return None


class TracingClientFactory:
    """Factory for creating the configured tracing client."""

    @staticmethod
    def create(settings: Settings) -> TracingClient:
        """Create Langfuse tracing or a no-op fallback."""
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            return NoOpTracingClient()

        return LangfuseTracingClient(settings=settings)


@lru_cache(maxsize=1)
def get_tracing_client() -> TracingClient:
    """Return the cached tracing client."""
    return TracingClientFactory.create(settings=get_settings())
