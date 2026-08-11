"""Common pipeline contracts and lifecycle helpers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar

from app.schemas.ai_context import AIContext
from app.tracing.tracing import TracingClient, get_tracing_client

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class StageExecutionError(RuntimeError):
    """Raised when a stage fails at the shared lifecycle boundary."""


class BaseStage(ABC, Generic[TInput, TOutput]):
    """Base lifecycle for all pipeline stages, including non-LLM stages.

    ``run`` is the single public lifecycle method because pipeline orchestration
    reads naturally as a sequence of stage runs, while ``_run`` keeps concrete
    business logic isolated in subclasses.
    """

    def __init__(
        self,
        *,
        stage_name: str | None = None,
        tracing_client: TracingClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize shared logging and tracing infrastructure."""
        self._stage_name = stage_name or self.__class__.__name__
        self._tracing_client = tracing_client or get_tracing_client()
        self._logger = logger or logging.getLogger(self.__class__.__module__)

    @property
    def stage_name(self) -> str:
        """Return the stable stage name used in logs and traces."""
        return self._stage_name

    def run(self, stage_input: TInput) -> TOutput:
        """Execute the stage with shared logging, tracing and error handling."""
        trace_input = self._build_trace_input(stage_input)
        self._logger.info("%s started", self._stage_name)

        with self._tracing_client.create_trace(
            self._stage_name,
            input=trace_input,
        ) as trace:
            try:
                result = self._run(stage_input)
            except Exception as exc:
                if trace is not None:
                    trace.update(
                        output={
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                self._logger.exception("%s failed", self._stage_name)
                stage_exception = self._build_stage_exception(exc)
                if stage_exception is exc:
                    raise exc
                raise stage_exception from exc

            if trace is not None:
                trace.update(output=self._build_trace_output(result))
            self._logger.info("%s completed", self._stage_name)
            return result

    @abstractmethod
    def _run(self, stage_input: TInput) -> TOutput:
        """Execute concrete stage logic without cross-cutting concerns."""

    def _build_trace_input(self, stage_input: TInput) -> dict[str, Any]:
        """Build safe trace metadata without serializing user payloads by default."""
        return {
            "stage": self._stage_name,
            "input_type": type(stage_input).__name__,
        }

    def _build_trace_output(self, stage_output: TOutput) -> dict[str, Any]:
        """Build safe trace output metadata without serializing full results."""
        return {
            "status": "success",
            "stage": self._stage_name,
            "output_type": type(stage_output).__name__,
        }

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Convert implementation errors into a stage-level exception."""
        if isinstance(exc, StageExecutionError):
            return exc
        return StageExecutionError(f"{self._stage_name} failed: {exc}")


class PipelineStage(Protocol):
    """Common contract for stages that transform AIContext by copy."""

    def run_context(self, context: AIContext) -> AIContext:
        """Return context enriched with this stage output."""
