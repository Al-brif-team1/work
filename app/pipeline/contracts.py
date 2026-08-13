"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar

from app.schemas.ai_context import AIContext
from app.tracing.tracing import TracingClient, get_tracing_client

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class StageExecutionError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class BaseStage(ABC, Generic[TInput, TOutput]):
    """Класс «BaseStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(
        self,
        *,
        stage_name: str | None = None,
        tracing_client: TracingClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        self._stage_name = stage_name or self.__class__.__name__
        self._tracing_client = tracing_client or get_tracing_client()
        self._logger = logger or logging.getLogger(self.__class__.__module__)

    @property
    def stage_name(self) -> str:
        """Выполняет шаг «stage name». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._stage_name

    def run(self, stage_input: TInput) -> TOutput:
        """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""
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
        """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""

    def _build_trace_input(self, stage_input: TInput) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return {
            "stage": self._stage_name,
            "input_type": type(stage_input).__name__,
        }

    def _build_trace_output(self, stage_output: TOutput) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return {
            "status": "success",
            "stage": self._stage_name,
            "output_type": type(stage_output).__name__,
        }

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        if isinstance(exc, StageExecutionError):
            return exc
        return StageExecutionError(f"{self._stage_name} failed: {exc}")


class PipelineStage(Protocol):
    """Класс «PipelineStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def run_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
