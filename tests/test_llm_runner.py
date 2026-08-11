"""Tests for the centralized LLM runner."""

from __future__ import annotations

import time
import unittest
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from app.llm import (
    LLMRunResult,
    LLMRunner,
    LLMRunnerProviderError,
    LLMRunnerStructuredOutputError,
)


class RecordingTraceContext:
    """Simple trace/span context used by unit tests."""

    def __init__(self) -> None:
        self.id = "trace-1"
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class RecordingTracingClient:
    """Tracing double that records trace and span updates."""

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
    """Deterministic LLM client used for runner tests."""

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
    """Minimal structured response for runner tests."""

    value: str


class TestLLMRunner(unittest.TestCase):
    """Unit tests for centralized LLM execution."""

    def test_retries_provider_errors_and_returns_telemetry(self) -> None:
        llm_client = FakeLLMClient(
            [
                RuntimeError("temporary"),
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

    def test_invalid_json_from_client_is_wrapped_as_provider_error(self) -> None:
        llm_client = FakeLLMClient([RuntimeError("LLM response is not valid JSON")])
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
        self.assertIn("not valid JSON", str(context.exception.__cause__))

    def test_structured_validation_errors_are_centralized(self) -> None:
        llm_client = FakeLLMClient([{"unexpected": "shape"}])
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

        self.assertIsInstance(
            context.exception.__cause__,
            LLMRunnerStructuredOutputError,
        )

    def test_timeout_is_reported_as_runner_error(self) -> None:
        llm_client = FakeLLMClient([{"value": "late"}], delay_seconds=0.2)
        runner = LLMRunner(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            max_retries=1,
            timeout_seconds=0.01,
        )

        started_at = time.perf_counter()
        with self.assertRaises(LLMRunnerProviderError):
            runner.run_json(
                messages=[{"role": "user", "content": "hello"}],
                response_model=RunnerPayload,
                trace_name="runner.test",
                span_name="runner.test.llm",
            )
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.1)


if __name__ == "__main__":
    unittest.main()
