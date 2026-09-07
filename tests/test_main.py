"""Tests for CLI-facing error messages."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from app.llm.runner import (
    LLMRunnerProviderError,
    LLMRunnerStructuredOutputError,
    LLMRunnerTimeoutError,
)
from app.main import format_pipeline_error, run


def wrap_exception(cause: Exception) -> RuntimeError:
    try:
        raise RuntimeError("stage failed") from cause
    except RuntimeError as exc:
        return exc


class TestMainCliErrors(unittest.TestCase):
    def test_cli_mapping_for_402(self) -> None:
        exc = wrap_exception(
            LLMRunnerProviderError(
                "provider failed",
                status_code=402,
                retryable=False,
                attempts_executed=1,
            )
        )

        self.assertEqual(
            format_pipeline_error(exc),
            "Недостаточно доступа или кредитов у провайдера LLM.",
        )

    def test_cli_mapping_for_429(self) -> None:
        exc = wrap_exception(
            LLMRunnerProviderError(
                "provider failed",
                status_code=429,
                retryable=True,
                attempts_executed=2,
            )
        )

        self.assertEqual(
            format_pipeline_error(exc),
            "Провайдер LLM временно ограничил частоту запросов. Попробуйте позже.",
        )

    def test_cli_mapping_for_timeout(self) -> None:
        exc = wrap_exception(LLMRunnerTimeoutError("timed out"))

        self.assertEqual(
            format_pipeline_error(exc),
            "Провайдер LLM не ответил вовремя. Попробуйте позже.",
        )

    def test_cli_mapping_for_empty_model_response(self) -> None:
        exc = wrap_exception(
            LLMRunnerStructuredOutputError(
                "empty",
                error_kind="empty_content",
                attempts_executed=2,
            )
        )

        self.assertEqual(
            format_pipeline_error(exc),
            "Модель вернула пустой ответ. Попробуйте повторить запрос.",
        )

    def test_cli_mapping_for_malformed_json(self) -> None:
        exc = wrap_exception(
            LLMRunnerStructuredOutputError(
                "invalid json",
                error_kind="malformed_json",
                attempts_executed=2,
            )
        )

        self.assertEqual(
            format_pipeline_error(exc),
            "Модель вернула некорректный структурированный ответ. Попробуйте повторить запрос.",
        )

    def test_cli_mapping_for_probably_truncated_json(self) -> None:
        exc = wrap_exception(
            LLMRunnerStructuredOutputError(
                "truncated",
                error_kind="truncated_json",
                attempts_executed=2,
            )
        )

        self.assertEqual(
            format_pipeline_error(exc),
            "Ответ модели был обрезан и не может быть разобран. Попробуйте повторить запрос.",
        )

    def test_cli_mapping_for_length_finish(self) -> None:
        exc = wrap_exception(
            LLMRunnerStructuredOutputError(
                "length",
                error_kind="length_finish",
                attempts_executed=2,
            )
        )

        self.assertEqual(
            format_pipeline_error(exc),
            "Ответ модели был обрезан из-за лимита генерации. Попробуйте сократить бриф или увеличить лимит ответа.",
        )

    def test_normalize_only_success_path_is_unchanged(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = run(["--text", "Build a support bot", "--normalize-only"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["normalized_text"], "Build a support bot")


if __name__ == "__main__":
    unittest.main()
