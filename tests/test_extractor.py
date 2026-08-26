"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import os
import inspect
import unittest
from contextlib import contextmanager
from typing import Any, Iterable

from app.config import Settings
from app.input import BriefInputFactory
from app.llm import LLMClientFactory
from app.llm.runner import (
    LLMRunResult,
    LLMRunnerProviderError,
    LLMRunnerStructuredOutputError,
    LLMTokenUsage,
)
from app.pipeline import Extractor, ExtractorError
from app.schemas import AIContext, ExtractedBrief, FactStatus
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
        self.span = RecordingTraceContext()
        self.trace_calls: list[dict[str, Any]] = []
        self.span_calls: list[dict[str, Any]] = []

    @contextmanager
    def create_trace(self, name: str, **kwargs: Any) -> Iterable[RecordingTraceContext]:
        self.trace_calls.append({"name": name, **kwargs})
        yield self.trace

    @contextmanager
    def create_span(self, name: str, **kwargs: Any) -> Iterable[RecordingTraceContext]:
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

    def stream(self, messages: Any, **kwargs: Any) -> Iterable[str]:  # pragma: no cover
        raise NotImplementedError


class FakeLLMRunner:
    """Класс «FakeLLMRunner» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, payload: ExtractedBrief | Exception) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> LLMRunResult[ExtractedBrief]:
        self.calls.append(kwargs)
        if isinstance(self.payload, Exception):
            raise self.payload

        return LLMRunResult(
            payload=self.payload,
            raw_response={"ok": True},
            attempts=1,
            latency_seconds=0.01,
            token_usage=LLMTokenUsage(total_tokens_estimate=1),
            provider_metadata={},
            trace_id=None,
            trace_name=kwargs["trace_name"],
            trace_enabled=False,
            model_name="fake-model",
            recovered_errors=[],
        )


def make_valid_extraction() -> dict[str, Any]:
    """Выполняет шаг «make valid extraction». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return {
        "project_goal": {
            "status": "explicit",
            "value": "Создать Telegram-бот для поддержки клиентов",
            "evidence": ["Создать Telegram-бот для поддержки клиентов."],
            "confidence": 0.96,
            "notes": None,
        },
        "tasks": [
            {
                "status": "explicit",
                "value": "Разработать Telegram-бот",
                "evidence": ["Создать Telegram-бот для поддержки клиентов."],
                "confidence": 0.9,
                "notes": None,
            }
        ],
        "project_type": {
            "status": "explicit",
            "value": "software",
            "evidence": ["Telegram-бот"],
            "confidence": 0.8,
            "notes": None,
        },
        "project_direction": {
            "status": "explicit",
            "value": "support automation",
            "evidence": ["поддержки клиентов"],
            "confidence": 0.82,
            "notes": None,
        },
        "technologies": [
            {
                "status": "explicit",
                "value": "Python",
                "evidence": ["Python"],
                "confidence": 0.99,
                "notes": None,
            }
        ],
        "stack": [
            {
                "status": "explicit",
                "value": "FastAPI",
                "evidence": ["FastAPI"],
                "confidence": 0.95,
                "notes": None,
            }
        ],
        "materials": [],
        "expected_result": {
            "status": "explicit",
            "value": "Работающий Telegram-бот",
            "evidence": ["Telegram-бот"],
            "confidence": 0.92,
            "notes": None,
        },
        "constraints": [
            {
                "status": "explicit",
                "value": "Срок до конца Q4 2026",
                "evidence": ["до конца Q4 2026"],
                "confidence": 0.98,
                "notes": None,
            }
        ],
        "deadlines": [
            {
                "status": "explicit",
                "value": "Q4 2026",
                "evidence": ["Q4 2026"],
                "confidence": 0.99,
                "notes": None,
            }
        ],
        "existing_resources": [],
        "integrations": [
            {
                "status": "explicit",
                "value": "GitHub",
                "evidence": ["GitHub"],
                "confidence": 0.97,
                "notes": None,
            }
        ],
        "other_facts": [],
    }


class TestExtractor(unittest.TestCase):
    """Класс «TestExtractor» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def setUp(self) -> None:
        self.brief_input = BriefInputFactory().from_text(
            "Создать Telegram-бот для поддержки клиентов. "
            "Использовать Python и FastAPI. "
            "Нужна интеграция с GitHub. "
            "Срок - до конца Q4 2026."
        )

    def test_extracts_structured_brief(self) -> None:
        llm_client = FakeLLMClient([make_valid_extraction()])
        tracing_client = RecordingTracingClient()
        extractor = Extractor(
            llm_client=llm_client,
            tracing_client=tracing_client,
            model_name="test-model",
        )

        result = extractor.extract(self.brief_input)

        self.assertEqual(
            result.extracted_brief.project_goal.status,
            FactStatus.explicit,
        )
        self.assertEqual(
            result.extracted_brief.project_goal.value,
            "Создать Telegram-бот для поддержки клиентов",
        )
        self.assertEqual(result.extracted_brief.stack[0].value, "FastAPI")
        self.assertEqual(result.technical_info.attempts, 1)
        self.assertTrue(result.technical_info.trace_enabled)
        self.assertEqual(result.technical_info.prompt_name, "extractor.md")
        self.assertEqual(result.technical_info.model_name, "test-model")
        self.assertIn(
            "extractor.brief",
            [call["name"] for call in tracing_client.trace_calls],
        )
        self.assertIn(
            "extractor.llm",
            [call["name"] for call in tracing_client.span_calls],
        )

    def test_uses_base_llm_stage_runner_contract(self) -> None:
        payload = ExtractedBrief.model_validate(make_valid_extraction())
        runner = FakeLLMRunner(payload)
        extractor = Extractor(
            llm_runner=runner,
            tracing_client=NoOpTracingClient(),
        )

        result = extractor.extract(self.brief_input)

        self.assertEqual(result.extracted_brief, payload)
        self.assertEqual(Extractor.output_model, ExtractedBrief)
        self.assertEqual(runner.calls[0]["output_model"], ExtractedBrief)
        self.assertEqual(runner.calls[0]["trace_name"], "extractor.brief")
        self.assertEqual(runner.calls[0]["span_name"], "extractor.llm")
        self.assertIn(self.brief_input.normalized_text, runner.calls[0]["prompt"])
        self.assertNotIn(
            self.brief_input.normalized_text,
            runner.calls[0]["system_prompt"],
        )
        self.assertIn("factual extractor", runner.calls[0]["system_prompt"])

    def test_extractor_uses_prompt_manager_render_without_manual_replace(self) -> None:
        source = inspect.getsource(Extractor)

        self.assertIn("_render_prompt", source)
        self.assertNotIn(".replace(", source)

    def test_run_context_updates_ai_context_with_extraction_result(self) -> None:
        payload = ExtractedBrief.model_validate(make_valid_extraction())
        runner = FakeLLMRunner(payload)
        extractor = Extractor(
            llm_runner=runner,
            tracing_client=NoOpTracingClient(),
        )
        brief_input = BriefInputFactory().from_text("Build a bot")

        updated = extractor.run_context(AIContext.from_brief(brief_input))

        self.assertEqual(updated.extraction_result.extracted_brief, payload)

    def test_postprocess_trims_strings_without_business_inference(self) -> None:
        payload_data = make_valid_extraction()
        payload_data["project_goal"]["value"] = "  Build a bot  "
        payload_data["project_goal"]["evidence"] = ["  Build a bot  ", "   "]
        payload_data["project_goal"]["notes"] = "  explicit goal  "
        runner = FakeLLMRunner(ExtractedBrief.model_validate(payload_data))
        extractor = Extractor(
            llm_runner=runner,
            tracing_client=NoOpTracingClient(),
        )

        result = extractor.extract(self.brief_input)

        self.assertEqual(result.extracted_brief.project_goal.value, "Build a bot")
        self.assertEqual(result.extracted_brief.project_goal.evidence, ["Build a bot"])
        self.assertEqual(result.extracted_brief.project_goal.notes, "explicit goal")

    def test_structured_output_error_uses_stage_error_contract(self) -> None:
        extractor = Extractor(
            llm_runner=FakeLLMRunner(
                LLMRunnerStructuredOutputError("invalid structured output")
            ),
            tracing_client=NoOpTracingClient(),
        )

        with self.assertRaises(ExtractorError):
            extractor.extract(self.brief_input)

    def test_llm_runner_exception_uses_stage_error_contract(self) -> None:
        extractor = Extractor(
            llm_runner=FakeLLMRunner(LLMRunnerProviderError("provider failed")),
            tracing_client=NoOpTracingClient(),
        )

        with self.assertRaises(ExtractorError):
            extractor.extract(self.brief_input)

    def test_retries_on_invalid_json_error(self) -> None:
        llm_client = FakeLLMClient(
            [
                ValueError("invalid JSON"),
                make_valid_extraction(),
            ]
        )
        extractor = Extractor(
            llm_client=llm_client,
            tracing_client=NoOpTracingClient(),
            model_name="test-model",
            max_retries=2,
        )

        result = extractor.extract(self.brief_input)

        self.assertEqual(result.technical_info.attempts, 2)
        self.assertEqual(len(result.technical_info.recovered_errors), 1)
        self.assertEqual(
            result.extracted_brief.project_goal.status,
            FactStatus.explicit,
        )

    def test_retries_on_llm_runtime_error(self) -> None:
        llm_client = FakeLLMClient(
            [
                RuntimeError("temporary failure"),
                make_valid_extraction(),
            ]
        )
        extractor = Extractor(
            llm_client=llm_client,
            tracing_client=NoOpTracingClient(),
            model_name="test-model",
            max_retries=2,
        )

        result = extractor.extract(self.brief_input)

        self.assertEqual(result.technical_info.attempts, 2)
        self.assertEqual(len(result.technical_info.recovered_errors), 1)
        self.assertEqual(
            result.extracted_brief.project_goal.status,
            FactStatus.explicit,
        )

    def test_retries_on_missing_required_structure(self) -> None:
        llm_client = FakeLLMClient(
            [
                {
                    "project_goal": {
                        "status": "missing",
                        "value": None,
                        "evidence": [],
                        "confidence": None,
                        "notes": None,
                    }
                },
                make_valid_extraction(),
            ]
        )
        extractor = Extractor(
            llm_client=llm_client,
            tracing_client=NoOpTracingClient(),
            model_name="test-model",
            max_retries=2,
        )

        result = extractor.extract(self.brief_input)

        self.assertEqual(result.technical_info.attempts, 2)
        self.assertEqual(len(result.technical_info.recovered_errors), 1)
        self.assertEqual(
            result.extracted_brief.expected_result.status,
            FactStatus.explicit,
        )

    def test_raises_after_retry_exhaustion(self) -> None:
        llm_client = FakeLLMClient(
            [RuntimeError("temporary failure"), RuntimeError("temporary failure")]
        )
        extractor = Extractor(
            llm_client=llm_client,
            tracing_client=NoOpTracingClient(),
            model_name="test-model",
            max_retries=2,
        )

        with self.assertRaises(ExtractorError):
            extractor.extract(self.brief_input)


@unittest.skipUnless(
    os.getenv("LLM_API_KEY") and os.getenv("LLM_MODEL") and os.getenv("LLM_BASE_URL"),
    "LLM_API_KEY, LLM_MODEL and LLM_BASE_URL are required for the integration test",
)
class TestExtractorIntegration(unittest.TestCase):
    """Класс «TestExtractorIntegration» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_real_llm_extraction_produces_structured_output(self) -> None:
        settings = Settings(
            LLM_API_KEY=os.environ["LLM_API_KEY"],
            LLM_MODEL=os.environ["LLM_MODEL"],
            LLM_BASE_URL=os.environ["LLM_BASE_URL"],
        )
        llm_client = LLMClientFactory.create(settings)
        extractor = Extractor(
            llm_client=llm_client,
            tracing_client=NoOpTracingClient(),
            model_name=settings.llm_model,
        )

        result = extractor.extract(
            BriefInputFactory().from_text(
                "Создать веб-приложение для обработки заявок. "
                "Использовать Python и FastAPI. "
                "Интеграция с GitHub обязательна."
            )
        )

        self.assertIsNotNone(result.extracted_brief.project_goal.status)
        self.assertGreaterEqual(result.technical_info.attempts, 1)
        self.assertEqual(result.technical_info.model_name, settings.llm_model)


if __name__ == "__main__":
    unittest.main()
