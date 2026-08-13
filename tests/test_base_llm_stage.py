"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.pipeline import BaseLLMStage, LLMStageRunResult
from app.llm.client import Message
from app.llm import LLMRunResult, LLMRunnerProviderError, LLMTokenUsage


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
        self.span = RecordingTraceContext()
        self.trace_calls: list[dict[str, Any]] = []
        self.span_calls: list[dict[str, Any]] = []

    @contextmanager
    def create_trace(self, name: str, **kwargs: Any):
        self.trace_calls.append({"name": name, **kwargs})
        yield self.trace

    @contextmanager
    def create_span(self, name: str, **kwargs: Any):
        self.span_calls.append({"name": name, **kwargs})
        yield self.span

    def flush(self) -> None:
        return None


class FakeLLMClient:
    """Класс «FakeLLMClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate_json(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, **kwargs})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def stream(self, messages: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


class DummyPayload(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    answer: str


class DummyStageError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class DummyStage(BaseLLMStage):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Робот этапа. Он выполняет участок конвейера «dummy stage». Этот этап работает как детерминированный робот: обычный код, без творческих догадок ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseLLMStage, поэтому он умеет работать в нашем конвейере."""

    def run(self, context: str) -> LLMStageRunResult[DummyPayload]:
        messages: list[Message] = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": context},
        ]
        return self._execute_structured_stage(
            trace_name="dummy.stage",
            span_name="dummy.stage.llm",
            trace_input={"context": context},
            messages=messages,
            response_model=DummyPayload,
        )

    @staticmethod
    def _default_prompt_path() -> Path:
        return Path(__file__).resolve()

    def _build_failure_exception(
        self,
        attempts: int,
        last_error: Exception | None,
    ) -> Exception:
        return DummyStageError(f"dummy stage failed after {attempts} attempts")


class FakeLLMRunner:
    """Класс «FakeLLMRunner» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, payload: DummyPayload) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> LLMRunResult[DummyPayload]:
        self.calls.append(kwargs)
        validator = kwargs.get("payload_validator")
        if validator is not None:
            try:
                validator(self.payload)
            except Exception as exc:
                raise LLMRunnerProviderError(f"validation failed: {exc}") from exc

        return LLMRunResult(
            payload=self.payload,
            raw_response=self.payload.model_dump(mode="json"),
            attempts=1,
            latency_seconds=0.01,
            token_usage=LLMTokenUsage(total_tokens_estimate=1),
            provider_metadata={},
            trace_name=kwargs["trace_name"],
            trace_enabled=False,
            model_name="fake-model",
            recovered_errors=[],
        )


class FailingLLMRunner:
    """Класс «FailingLLMRunner» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def run(self, **kwargs: Any) -> LLMRunResult[DummyPayload]:
        raise LLMRunnerProviderError("provider failed")


class TemplateStage(BaseLLMStage[str, DummyPayload, str]):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Робот этапа. Он выполняет участок конвейера «template stage». Этот этап работает как детерминированный робот: обычный код, без творческих догадок ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseLLMStage, поэтому он умеет работать в нашем конвейере."""

    output_model = DummyPayload

    def build_prompt(self, stage_input: str) -> str:
        return f"{self.prompt_template}: {stage_input}"

    def build_context(self, stage_input: str) -> dict[str, Any]:
        return {"input": stage_input}

    def postprocess(self, result: LLMRunResult[DummyPayload]) -> str:
        return result.payload.answer.upper()

    @staticmethod
    def _default_prompt_path() -> Path:
        return Path(__file__).resolve()

    def _build_failure_exception(
        self,
        attempts: int,
        last_error: Exception | None,
    ) -> Exception:
        return DummyStageError(f"template stage failed after {attempts} attempts")


class PostprocessFailingStage(TemplateStage):
    """Класс «PostprocessFailingStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def postprocess(self, result: LLMRunResult[DummyPayload]) -> str:
        raise ValueError("postprocess failed")


class ValidatingStage(TemplateStage):
    """Класс «ValidatingStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def validate_payload(self, payload: DummyPayload) -> None:
        if payload.answer != "valid":
            raise ValueError("answer must be valid")


class TestBaseLLMStage(unittest.TestCase):
    """Класс «TestBaseLLMStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_executes_with_retry_and_returns_structured_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "dummy.md"
            prompt_path.write_text("Prompt template", encoding="utf-8")

            llm_client = FakeLLMClient([RuntimeError("temporary"), {"answer": "ok"}])
            tracing_client = RecordingTracingClient()
            stage = DummyStage(
                llm_client=llm_client,
                tracing_client=tracing_client,
                prompt_path=prompt_path,
                max_retries=2,
                model_name="test-model",
            )

            result = stage.run("hello")

        self.assertEqual(result.payload.answer, "ok")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.prompt_name, "dummy.md")
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(len(result.recovered_errors), 1)
        self.assertEqual(tracing_client.trace_calls[0]["name"], "dummy.stage")
        self.assertEqual(tracing_client.span_calls[0]["name"], "dummy.stage.llm")
        self.assertEqual(len(llm_client.calls), 2)

    def test_raises_stage_specific_error_after_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "dummy.md"
            prompt_path.write_text("Prompt template", encoding="utf-8")

            llm_client = FakeLLMClient([RuntimeError("temporary"), RuntimeError("again")])
            stage = DummyStage(
                llm_client=llm_client,
                tracing_client=RecordingTracingClient(),
                prompt_path=prompt_path,
                max_retries=2,
            )

            with self.assertRaises(DummyStageError):
                stage.run("hello")

    def test_template_run_calls_llm_runner_and_postprocesses_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "template.md"
            prompt_path.write_text("Prompt template", encoding="utf-8")
            runner = FakeLLMRunner(DummyPayload(answer="ok"))
            stage = TemplateStage(
                llm_runner=runner,
                tracing_client=RecordingTracingClient(),
                prompt_path=prompt_path,
            )

            result = stage.run("hello")

        self.assertEqual(result, "OK")
        self.assertEqual(runner.calls[0]["prompt"], "Prompt template: hello")
        self.assertEqual(runner.calls[0]["context"], {"input": "hello"})
        self.assertIs(runner.calls[0]["output_model"], DummyPayload)
        self.assertEqual(runner.calls[0]["trace_name"], "TemplateStage.run")
        self.assertEqual(runner.calls[0]["span_name"], "TemplateStage.llm")

    def test_template_run_wraps_llm_runner_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "template.md"
            prompt_path.write_text("Prompt template", encoding="utf-8")
            stage = TemplateStage(
                llm_runner=FailingLLMRunner(),
                tracing_client=RecordingTracingClient(),
                prompt_path=prompt_path,
            )

            with self.assertRaises(DummyStageError) as context:
                stage.run("hello")

        self.assertIsInstance(context.exception.__cause__, LLMRunnerProviderError)

    def test_template_run_wraps_postprocess_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "template.md"
            prompt_path.write_text("Prompt template", encoding="utf-8")
            stage = PostprocessFailingStage(
                llm_runner=FakeLLMRunner(DummyPayload(answer="ok")),
                tracing_client=RecordingTracingClient(),
                prompt_path=prompt_path,
            )

            with self.assertRaises(DummyStageError) as context:
                stage.run("hello")

        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_template_run_wires_payload_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "template.md"
            prompt_path.write_text("Prompt template", encoding="utf-8")
            stage = ValidatingStage(
                llm_runner=FakeLLMRunner(DummyPayload(answer="invalid")),
                tracing_client=RecordingTracingClient(),
                prompt_path=prompt_path,
            )

            with self.assertRaises(DummyStageError) as context:
                stage.run("hello")

        self.assertIsInstance(context.exception.__cause__, LLMRunnerProviderError)


if __name__ == "__main__":
    unittest.main()
