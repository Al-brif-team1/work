import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.config import Config, Settings
from app.llm import LLMClientFactory, OpenRouterLLMClient
from app.tracing.logger import LoggerFactory
from app.tracing.tracing import NoOpTracingClient, TracingClientFactory


def build_settings(**overrides: Any) -> Settings:
    """Собирает Settings с обязательными полями, чтобы тесты задавали только то, что проверяют."""
    values: dict[str, Any] = {
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "test-model",
        "LLM_BASE_URL": "https://example.test/api/v1",
    }
    values.update(overrides)
    return Settings(**values)


class FakeCompletions:
    """Класс «FakeCompletions» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"))]
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]
                ),
            ]

        content = '{"ok": true}' if kwargs.get("response_format") else "hello"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAIClient:
    """Класс «FakeOpenAIClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class TestInfrastructure(unittest.TestCase):
    """Класс «TestInfrastructure» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_settings_loads_from_environment(self) -> None:
        env = {
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_BASE_URL": "https://example.test/api/v1",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Config.load()

        self.assertEqual(settings.llm_api_key, "test-key")
        self.assertEqual(settings.llm_model, "test-model")
        self.assertEqual(settings.llm_base_url, "https://example.test/api/v1")
        self.assertTrue(settings.debug)
        self.assertEqual(settings.log_level, "DEBUG")

    def test_settings_missing_required_values_raises_readable_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            Config,
            "_env_file",
            return_value=Path("__missing_test_env_file__.env"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing required"):
                Config.load()

    def test_logger_writes_to_console_and_file_handlers(self) -> None:
        settings = build_settings(LOG_LEVEL="DEBUG")
        logger = logging.getLogger(LoggerFactory.LOGGER_NAME)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        configured_logger = LoggerFactory.create(settings)

        self.assertEqual(configured_logger.level, logging.DEBUG)
        self.assertEqual(len(configured_logger.handlers), 2)

    def test_llm_factory_returns_openrouter_client(self) -> None:
        settings = build_settings()

        client = LLMClientFactory.create(settings)

        self.assertIsInstance(client, OpenRouterLLMClient)

    def test_openrouter_client_generate_methods_use_openai_client(self) -> None:
        settings = build_settings()
        client = OpenRouterLLMClient(settings)
        client._client = FakeOpenAIClient()
        messages = [{"role": "user", "content": "test"}]

        self.assertEqual(client.generate(messages), "hello")
        self.assertEqual(client.generate_json(messages), {"ok": True})
        self.assertEqual(list(client.stream(messages)), ["hel", "lo"])

    def test_llm_settings_defaults_match_documented_values(self) -> None:
        settings = build_settings()

        self.assertEqual(settings.llm_temperature, 0.0)
        self.assertIsNone(settings.llm_max_tokens)
        self.assertIsNone(settings.llm_top_p)
        self.assertEqual(settings.llm_max_attempts, 2)
        self.assertEqual(settings.llm_timeout_seconds, 60.0)
        self.assertEqual(settings.llm_transport_retries, 0)

    def test_llm_settings_reject_out_of_range_values(self) -> None:
        invalid_overrides = [
            {"LLM_TEMPERATURE": "3"},
            {"LLM_TOP_P": "1.5"},
            {"LLM_TOP_P": "0"},
            {"LLM_MAX_TOKENS": "0"},
            {"LLM_MAX_ATTEMPTS": "0"},
            {"LLM_TIMEOUT_SECONDS": "0"},
            {"LLM_TRANSPORT_RETRIES": "-1"},
            {"LLM_BASE_URL": "openrouter.ai/api/v1"},
        ]

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValidationError):
                    build_settings(**overrides)

    def test_llm_base_url_is_required(self) -> None:
        env = {"LLM_API_KEY": "test-key", "LLM_MODEL": "test-model"}

        with patch.dict(os.environ, env, clear=True), patch.object(
            Config,
            "_env_file",
            return_value=Path("__missing_test_env_file__.env"),
        ):
            with self.assertRaisesRegex(RuntimeError, "LLM_BASE_URL"):
                Config.load()

    def test_client_passes_connection_settings_to_openai_sdk(self) -> None:
        settings = build_settings(
            LLM_BASE_URL="https://example.test/api/v1/",
            LLM_TIMEOUT_SECONDS="12.5",
            LLM_TRANSPORT_RETRIES="3",
        )

        with patch("app.llm.openrouter.OpenAI") as openai_factory:
            OpenRouterLLMClient(settings)

        _, kwargs = openai_factory.call_args
        self.assertEqual(kwargs["api_key"], "test-key")
        # Хвостовой слэш срезается валидатором, чтобы адрес не склеивался с двойным слэшем.
        self.assertEqual(kwargs["base_url"], "https://example.test/api/v1")
        self.assertEqual(kwargs["timeout"], 12.5)
        self.assertEqual(kwargs["max_retries"], 3)

    def test_client_applies_generation_settings_to_every_request(self) -> None:
        settings = build_settings(
            LLM_TEMPERATURE="0.7",
            LLM_MAX_TOKENS="512",
            LLM_TOP_P="0.9",
        )
        client = OpenRouterLLMClient(settings)
        fake = FakeOpenAIClient()
        client._client = fake
        messages = [{"role": "user", "content": "test"}]

        client.generate(messages)
        list(client.stream(messages))

        self.assertEqual(len(fake.completions.calls), 2)
        for call in fake.completions.calls:
            self.assertEqual(call["model"], "test-model")
            self.assertEqual(call["temperature"], 0.7)
            self.assertEqual(call["max_tokens"], 512)
            self.assertEqual(call["top_p"], 0.9)

    def test_client_omits_unset_generation_parameters(self) -> None:
        client = OpenRouterLLMClient(build_settings())
        fake = FakeOpenAIClient()
        client._client = fake

        client.generate([{"role": "user", "content": "test"}])

        call = fake.completions.calls[0]
        self.assertEqual(call["temperature"], 0.0)
        self.assertNotIn("max_tokens", call)
        self.assertNotIn("top_p", call)

    def test_call_arguments_override_generation_settings(self) -> None:
        client = OpenRouterLLMClient(build_settings(LLM_TEMPERATURE="0.7"))
        fake = FakeOpenAIClient()
        client._client = fake

        client.generate([{"role": "user", "content": "test"}], temperature=0.1)

        self.assertEqual(fake.completions.calls[0]["temperature"], 0.1)

    def test_langfuse_without_keys_returns_noop_client(self) -> None:
        settings = build_settings()

        client = TracingClientFactory.create(settings)

        self.assertIsInstance(client, NoOpTracingClient)
        with client.create_trace("test") as trace:
            self.assertIsNone(trace)
        with client.create_span("test") as span:
            self.assertIsNone(span)
        client.flush()


if __name__ == "__main__":
    unittest.main()
