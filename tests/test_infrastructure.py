import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from app.config import Config, Settings
from app.llm import LLMClientFactory, OpenRouterLLMClient
from app.tracing.logger import LoggerFactory
from app.tracing.tracing import NoOpTracingClient, TracingClientFactory


class FakeCompletions:
    """Fake OpenAI completions API for LLM client smoke tests."""

    def create(self, **kwargs: Any) -> Any:
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
    """Fake OpenAI client exposing chat.completions.create."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


class TestInfrastructure(unittest.TestCase):
    """Infrastructure smoke tests for settings, logging, LLM, and tracing."""

    def test_settings_loads_from_environment(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "test-model",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Config.load()

        self.assertEqual(settings.openrouter_api_key, "test-key")
        self.assertEqual(settings.openrouter_model, "test-model")
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
        settings = Settings(
            OPENROUTER_API_KEY="test-key",
            OPENROUTER_MODEL="test-model",
            LOG_LEVEL="DEBUG",
        )
        logger = logging.getLogger(LoggerFactory.LOGGER_NAME)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        configured_logger = LoggerFactory.create(settings)

        self.assertEqual(configured_logger.level, logging.DEBUG)
        self.assertEqual(len(configured_logger.handlers), 2)

    def test_llm_factory_returns_openrouter_client(self) -> None:
        settings = Settings(
            OPENROUTER_API_KEY="test-key",
            OPENROUTER_MODEL="test-model",
        )

        client = LLMClientFactory.create(settings)

        self.assertIsInstance(client, OpenRouterLLMClient)

    def test_openrouter_client_generate_methods_use_openai_client(self) -> None:
        settings = Settings(
            OPENROUTER_API_KEY="test-key",
            OPENROUTER_MODEL="test-model",
        )
        client = OpenRouterLLMClient(settings)
        client._client = FakeOpenAIClient()
        messages = [{"role": "user", "content": "test"}]

        self.assertEqual(client.generate(messages), "hello")
        self.assertEqual(client.generate_json(messages), {"ok": True})
        self.assertEqual(list(client.stream(messages)), ["hel", "lo"])

    def test_langfuse_without_keys_returns_noop_client(self) -> None:
        settings = Settings(
            OPENROUTER_API_KEY="test-key",
            OPENROUTER_MODEL="test-model",
        )

        client = TracingClientFactory.create(settings)

        self.assertIsInstance(client, NoOpTracingClient)
        with client.create_trace("test") as trace:
            self.assertIsNone(trace)
        with client.create_span("test") as span:
            self.assertIsNone(span)
        client.flush()


if __name__ == "__main__":
    unittest.main()
