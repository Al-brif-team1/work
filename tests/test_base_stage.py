"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import logging
import unittest
from contextlib import contextmanager
from typing import Any

from app.pipeline import BaseStage, StageExecutionError
from app.tracing.tracing import NoOpTracingClient


class RecordingTraceContext:
    """Класс «RecordingTraceContext» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class RecordingTracingClient:
    """Класс «RecordingTracingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self) -> None:
        self.trace = RecordingTraceContext()
        self.trace_calls: list[dict[str, Any]] = []

    @contextmanager
    def create_trace(self, name: str, **kwargs: Any):
        self.trace_calls.append({"name": name, **kwargs})
        yield self.trace

    @contextmanager
    def create_span(self, name: str, **kwargs: Any):
        yield None

    def flush(self) -> None:
        return None


class UppercaseStage(BaseStage[str, str]):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Робот этапа. Он выполняет участок конвейера «uppercase stage». Этот этап работает как детерминированный робот: обычный код, без творческих догадок ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseStage, поэтому он умеет работать в нашем конвейере."""

    def _run(self, stage_input: str) -> str:
        return stage_input.upper()


class CustomStageError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class FailingStage(BaseStage[str, str]):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Робот этапа. Он выполняет участок конвейера «failing stage». Этот этап работает как детерминированный робот: обычный код, без творческих догадок ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseStage, поэтому он умеет работать в нашем конвейере."""

    def _run(self, stage_input: str) -> str:
        raise ValueError(f"bad input: {stage_input}")

    def _build_stage_exception(self, exc: Exception) -> Exception:
        return CustomStageError(f"custom failure: {exc}")


class TestBaseStage(unittest.TestCase):
    """Класс «TestBaseStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_run_executes_stage_with_logging_and_tracing(self) -> None:
        tracing_client = RecordingTracingClient()
        logger = logging.getLogger("tests.base_stage.success")
        stage = UppercaseStage(
            stage_name="uppercase",
            tracing_client=tracing_client,
            logger=logger,
        )

        with self.assertLogs("tests.base_stage.success", level="INFO") as logs:
            result = stage.run("hello")

        self.assertEqual(result, "HELLO")
        self.assertEqual(stage.stage_name, "uppercase")
        self.assertEqual(tracing_client.trace_calls[0]["name"], "uppercase")
        self.assertEqual(
            tracing_client.trace_calls[0]["input"],
            {"stage": "uppercase", "input_type": "str"},
        )
        self.assertEqual(
            tracing_client.trace.updates[0]["output"],
            {"status": "success", "stage": "uppercase", "output_type": "str"},
        )
        self.assertTrue(any("uppercase started" in line for line in logs.output))
        self.assertTrue(any("uppercase completed" in line for line in logs.output))

    def test_run_wraps_errors_and_preserves_cause(self) -> None:
        stage = FailingStage(
            stage_name="failing",
            tracing_client=NoOpTracingClient(),
            logger=logging.getLogger("tests.base_stage.failure"),
        )

        with self.assertRaises(CustomStageError) as context:
            stage.run("payload")

        self.assertIsInstance(context.exception.__cause__, ValueError)
        self.assertIn("bad input: payload", str(context.exception))

    def test_default_error_type_is_stage_execution_error(self) -> None:
        class DefaultFailingStage(BaseStage[str, str]):
            def _run(self, stage_input: str) -> str:
                raise RuntimeError("boom")

        stage = DefaultFailingStage(
            stage_name="default-failing",
            tracing_client=NoOpTracingClient(),
        )

        with self.assertRaises(StageExecutionError) as context:
            stage.run("payload")

        self.assertIsInstance(context.exception.__cause__, RuntimeError)
        self.assertIn("default-failing failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
