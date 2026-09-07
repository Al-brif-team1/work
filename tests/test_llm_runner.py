"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import time
import unittest
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from app.llm import (
    LLMProviderError,
    LLMRunResult,
    LLMRunner,
    LLMRunnerProviderError,
    LLMRunnerStructuredOutputError,
    LLMStructuredOutputError,
)
from app.llm.openrouter import OpenRouterLLMClient


class RecordingTraceContext:
    """Класс «RecordingTraceContext» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self) -> None:
        self.id = "trace-1"
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

    def __init__(
        self,
        responses: list[dict[str, Any] | Exception],
        delay_seconds: float = 0,
    ) -> None:
        self._responses = responses
        self._delay_seconds = delay_seconds
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate_json(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, **kwargs})
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def stream(self, messages: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


class RunnerPayload(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    value: str


class TestLLMRunner(unittest.TestCase):
    """Класс «TestLLMRunner» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_retries_provider_errors_and_returns_telemetry(self) -> None:
        llm_client = FakeLLMClient(
            [
                LLMProviderError(
                    "temporary",
                    status_code=503,
                    retryable=True,
                ),
                {
                    "value": "ok",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    },
                    "model": "provider-model",
                },
            ]
        )
        tracing_client = RecordingTracingClient()
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=tracing_client,
            max_retries=2,
            timeout_seconds=2,
            model_name="test-model",
        )

        result = runner.run_json(
            messages=[{"role": "user", "content": "hello"}],
            response_model=RunnerPayload,
            trace_name="runner.test",
            span_name="runner.test.llm",
        )

        self.assertEqual(result.payload.value, "ok")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.recovered_errors), 1)
        self.assertGreaterEqual(result.latency_seconds, 0)
        self.assertEqual(result.trace_id, "trace-1")
        self.assertEqual(result.model_name, "test-model")
        self.assertIsInstance(result, LLMRunResult)
        self.assertEqual(result.token_usage.prompt_tokens, 10)
        self.assertEqual(result.token_usage.completion_tokens, 4)
        self.assertEqual(result.token_usage.total_tokens, 14)
        self.assertEqual(result.provider_metadata["model"], "provider-model")
        self.assertEqual(tracing_client.trace_calls[0]["name"], "runner.test")
        self.assertEqual(tracing_client.span_calls[0]["name"], "runner.test.llm")
        self.assertEqual(result.model_dump()["payload"]["value"], "ok")

    def test_429_provider_error_is_retried(self) -> None:
        self._assert_provider_status_is_retried(429)

    def test_503_provider_error_is_retried(self) -> None:
        self._assert_provider_status_is_retried(503)

    def test_401_provider_error_is_not_retried(self) -> None:
        self._assert_provider_status_is_not_retried(401)

    def test_402_provider_error_is_not_retried(self) -> None:
        self._assert_provider_status_is_not_retried(402)

    def test_400_provider_error_is_not_retried(self) -> None:
        self._assert_provider_status_is_not_retried(400)

    def test_run_builds_messages_from_prompt_and_context(self) -> None:
        llm_client = FakeLLMClient([{"value": "ok"}])
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=2,
        )

        result = runner.run(
            prompt="Return JSON.",
            output_model=RunnerPayload,
            context={"brief": "Build a portal"},
            trace_name="runner.simple",
            span_name="runner.simple.llm",
        )

        self.assertEqual(result.payload.value, "ok")
        self.assertEqual(llm_client.calls[0]["messages"][0]["role"], "system")
        self.assertIn("Return JSON.", llm_client.calls[0]["messages"][0]["content"])
        self.assertIn("JSON Schema", llm_client.calls[0]["messages"][0]["content"])
        self.assertIn("Russian", llm_client.calls[0]["messages"][0]["content"])
        self.assertEqual(llm_client.calls[0]["messages"][1]["role"], "user")
        self.assertIn("Build a portal", llm_client.calls[0]["messages"][1]["content"])

    def test_provider_exception_is_wrapped(self) -> None:
        llm_client = FakeLLMClient([RuntimeError("provider down")])
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=2,
        )

        with self.assertRaises(LLMRunnerProviderError) as context:
            runner.run_json(
                messages=[{"role": "user", "content": "hello"}],
                response_model=RunnerPayload,
                trace_name="runner.test",
                span_name="runner.test.llm",
            )

        self.assertIsInstance(context.exception.__cause__, LLMRunnerProviderError)
        self.assertIn("provider down", str(context.exception.__cause__))

    def test_run_json_adds_json_instruction_when_missing(self) -> None:
        llm_client = FakeLLMClient([{"value": "ok"}])
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=2,
        )

        runner.run_json(
            messages=[{"role": "user", "content": "Return the structured result."}],
            response_model=RunnerPayload,
            trace_name="runner.test",
            span_name="runner.test.llm",
        )

        messages = llm_client.calls[0]["messages"]
        self.assertIn("JSON", messages[0]["content"])
        self.assertIn("JSON Schema", messages[0]["content"])
        self.assertIn("Russian", messages[0]["content"])
        self.assertEqual(
            llm_client.calls[0]["response_format"]["type"],
            "json_schema",
        )
        self.assertEqual(
            llm_client.calls[0]["response_format"]["json_schema"]["name"],
            "RunnerPayload",
        )

    def test_run_json_preserves_existing_json_schema_instruction(self) -> None:
        llm_client = FakeLLMClient([{"value": "ok"}])
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=2,
        )

        runner.run_json(
            messages=[{"role": "user", "content": "Return JSON Schema."}],
            response_model=RunnerPayload,
            trace_name="runner.test",
            span_name="runner.test.llm",
        )

        messages = llm_client.calls[0]["messages"]
        self.assertEqual(messages[0]["content"], "Return JSON Schema.")

    def test_invalid_json_from_client_is_wrapped_as_structured_output_error(self) -> None:
        llm_client = FakeLLMClient(
            [
                LLMStructuredOutputError(
                    "LLM response is not valid JSON",
                    error_kind="malformed_json",
                    provider_metadata={"finish_reason": "stop", "model": "m"},
                    content_length=8,
                )
            ]
        )
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=2,
        )

        with self.assertRaises(LLMRunnerStructuredOutputError) as context:
            runner.run_json(
                messages=[{"role": "user", "content": "hello"}],
                response_model=RunnerPayload,
                trace_name="runner.test",
                span_name="runner.test.llm",
            )

        self.assertIsInstance(
            context.exception.__cause__,
            LLMRunnerStructuredOutputError,
        )
        self.assertEqual(context.exception.error_kind, "malformed_json")
        self.assertEqual(context.exception.content_length, 8)
        self.assertEqual(context.exception.provider_metadata["model"], "m")
        self.assertIn("not valid JSON", str(context.exception.__cause__))

    def test_invalid_json_logs_raw_response_for_each_attempt(self) -> None:
        llm_client = object.__new__(OpenRouterLLMClient)
        llm_client._create_completion = lambda messages, **kwargs: type(
            "Response",
            (),
            {
                "id": "response-1",
                "model": "provider-model",
                "usage": {"completion_tokens": 4},
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "finish_reason": "stop",
                            "message": type(
                                "Message",
                                (),
                                {"content": "```json\n{}\n```"},
                            )(),
                        },
                    )()
                ],
            },
        )()
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=2,
            timeout_seconds=None,
        )

        with self.assertLogs("app.llm.openrouter", level="ERROR") as logs:
            with self.assertRaises(LLMRunnerStructuredOutputError):
                runner.run_json(
                    messages=[{"role": "user", "content": "hello"}],
                    response_model=RunnerPayload,
                    trace_name="runner.test",
                    span_name="runner.test.llm",
                )

        self.assertEqual(len(logs.output), 2)
        for entry in logs.output:
            self.assertIn("JSONDecodeError", entry)
            self.assertIn("error_kind=malformed_json", entry)
            self.assertIn("response_length=14", entry)
            self.assertIn("finish_reason=stop", entry)
            self.assertIn("response_model=provider-model", entry)
            self.assertIn("response_id=response-1", entry)
            self.assertIn("usage={'completion_tokens': 4}", entry)
            self.assertIn("raw_response_text='```json\\n{}\\n```'", entry)

    def test_structured_validation_errors_are_centralized(self) -> None:
        llm_client = FakeLLMClient([{"unexpected": "shape"}])
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=2,
        )

        with self.assertRaises(LLMRunnerStructuredOutputError) as context:
            runner.run_json(
                messages=[{"role": "user", "content": "hello"}],
                response_model=RunnerPayload,
                trace_name="runner.test",
                span_name="runner.test.llm",
            )

        self.assertIsInstance(
            context.exception.__cause__,
            LLMRunnerStructuredOutputError,
        )

    def test_structured_output_error_is_retried(self) -> None:
        llm_client = FakeLLMClient(
            [
                {"unexpected": "shape"},
                {"value": "ok"},
            ]
        )
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=2,
            timeout_seconds=2,
        )

        result = runner.run_json(
            messages=[{"role": "user", "content": "hello"}],
            response_model=RunnerPayload,
            trace_name="runner.test",
            span_name="runner.test.llm",
        )

        self.assertEqual(result.payload.value, "ok")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(llm_client.calls), 2)

    def test_validation_error_logs_payload_for_each_attempt(self) -> None:
        llm_client = FakeLLMClient(
            [
                {"unexpected": "shape"},
                {"unexpected": "shape"},
            ]
        )
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=2,
            timeout_seconds=2,
        )

        with self.assertLogs("app.llm.runner", level="ERROR") as logs:
            with self.assertRaises(LLMRunnerStructuredOutputError):
                runner.run_json(
                    messages=[{"role": "user", "content": "hello"}],
                    response_model=RunnerPayload,
                    trace_name="runner.test",
                    span_name="runner.test.llm",
                )

        self.assertEqual(len(logs.output), 2)
        for entry in logs.output:
            self.assertIn("response_model=RunnerPayload", entry)
            self.assertIn("'unexpected': 'shape'", entry)
            self.assertIn("Field required", entry)

    def test_attempt_diagnostics_log_shape_without_message_content(self) -> None:
        llm_client = FakeLLMClient([{"value": "ok"}])
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=2,
            model_name="test-model",
        )

        with self.assertLogs("app.llm.runner", level="INFO") as logs:
            runner.run_json(
                messages=[{"role": "user", "content": "secret brief contents"}],
                response_model=RunnerPayload,
                trace_name="runner.test",
                span_name="runner.test.llm",
            )

        joined_logs = "\n".join(logs.output)
        actual_messages = llm_client.calls[0]["messages"]
        expected_lengths = [
            len(message.get("content", "")) for message in actual_messages
        ]
        self.assertIn("LLM attempt starting: trace_name=runner.test", joined_logs)
        self.assertIn("attempt=1", joined_logs)
        self.assertIn("model=test-model", joined_logs)
        self.assertIn("timeout=2", joined_logs)
        self.assertIn("response_model=RunnerPayload", joined_logs)
        self.assertIn("message_count=1", joined_logs)
        self.assertIn(f"message_lengths={expected_lengths}", joined_logs)
        self.assertIn(
            f"total_message_length={sum(expected_lengths)}",
            joined_logs,
        )
        self.assertIn("response_format_present=True", joined_logs)
        self.assertIn("status=success", joined_logs)
        self.assertIn("latency_seconds=", joined_logs)
        self.assertNotIn("secret brief contents", joined_logs)

    def test_timeout_is_reported_as_runner_error(self) -> None:
        llm_client = FakeLLMClient([TimeoutError("timed out")])
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=0.01,
        )

        with self.assertRaises(LLMRunnerProviderError) as context:
            runner.run_json(
                messages=[{"role": "user", "content": "hello"}],
                response_model=RunnerPayload,
                trace_name="runner.test",
                span_name="runner.test.llm",
            )

        self.assertEqual(len(llm_client.calls), 1)
        self.assertIn("timed out", str(context.exception.__cause__))

    def test_timeout_attempt_diagnostics_include_status_and_latency(self) -> None:
        llm_client = FakeLLMClient([TimeoutError("timed out")])
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=0.01,
            model_name="test-model",
        )

        with self.assertLogs("app.llm.runner", level="INFO") as logs:
            with self.assertRaises(LLMRunnerProviderError):
                runner.run_json(
                    messages=[{"role": "user", "content": "hello"}],
                    response_model=RunnerPayload,
                    trace_name="runner.test",
                    span_name="runner.test.llm",
                )

        joined_logs = "\n".join(logs.output)
        self.assertIn("LLM attempt starting: trace_name=runner.test", joined_logs)
        self.assertIn("status=timeout", joined_logs)
        self.assertIn("latency_seconds=", joined_logs)

    def test_structured_output_error_is_retried_without_becoming_provider_error(self) -> None:
        llm_client = FakeLLMClient(
            [
                LLMStructuredOutputError(
                    "empty",
                    error_kind="empty_content",
                    provider_metadata={"finish_reason": "stop"},
                    content_length=0,
                ),
                {"value": "ok"},
            ]
        )
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=2,
            timeout_seconds=2,
        )

        result = runner.run_json(
            messages=[{"role": "user", "content": "hello"}],
            response_model=RunnerPayload,
            trace_name="runner.test",
            span_name="runner.test.llm",
        )

        self.assertEqual(result.payload.value, "ok")
        self.assertEqual(len(llm_client.calls), 2)

    def test_length_finish_survives_after_retries_are_exhausted(self) -> None:
        llm_client = FakeLLMClient(
            [
                LLMStructuredOutputError(
                    "truncated",
                    error_kind="length_finish",
                    provider_metadata={"finish_reason": "length", "id": "r1"},
                    content_length=21,
                ),
                LLMStructuredOutputError(
                    "truncated",
                    error_kind="length_finish",
                    provider_metadata={"finish_reason": "length", "id": "r2"},
                    content_length=22,
                ),
            ]
        )
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=2,
            timeout_seconds=2,
        )

        with self.assertRaises(LLMRunnerStructuredOutputError) as context:
            runner.run_json(
                messages=[{"role": "user", "content": "hello"}],
                response_model=RunnerPayload,
                trace_name="runner.test",
                span_name="runner.test.llm",
            )

        self.assertEqual(len(llm_client.calls), 2)
        self.assertEqual(context.exception.error_kind, "length_finish")
        self.assertEqual(context.exception.provider_metadata["id"], "r2")
        self.assertEqual(context.exception.attempts_executed, 2)

    def _assert_provider_status_is_retried(self, status_code: int) -> None:
        llm_client = FakeLLMClient(
            [
                LLMProviderError(
                    f"status {status_code}",
                    status_code=status_code,
                    retryable=True,
                ),
                {"value": "ok"},
            ]
        )
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=2,
            timeout_seconds=2,
        )

        result = runner.run_json(
            messages=[{"role": "user", "content": "hello"}],
            response_model=RunnerPayload,
            trace_name="runner.test",
            span_name="runner.test.llm",
        )

        self.assertEqual(result.payload.value, "ok")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(llm_client.calls), 2)

    def _assert_provider_status_is_not_retried(self, status_code: int) -> None:
        llm_client = FakeLLMClient(
            [
                LLMProviderError(
                    f"status {status_code}",
                    status_code=status_code,
                    retryable=False,
                ),
                {"value": "should not be used"},
            ]
        )
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=2,
            timeout_seconds=2,
        )

        with self.assertRaises(LLMRunnerProviderError) as context:
            runner.run_json(
                messages=[{"role": "user", "content": "hello"}],
                response_model=RunnerPayload,
                trace_name="runner.test",
                span_name="runner.test.llm",
            )

        self.assertEqual(len(llm_client.calls), 1)
        self.assertEqual(context.exception.status_code, status_code)
        self.assertFalse(context.exception.retryable)


if __name__ == "__main__":
    unittest.main()
