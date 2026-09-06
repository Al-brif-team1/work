"""Tests for CLI-facing error messages."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from app.llm.runner import LLMRunnerProviderError, LLMRunnerTimeoutError
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
            "LLM provider access/credits error.",
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
            "LLM provider is temporarily rate-limited. Please try again later.",
        )

    def test_cli_mapping_for_timeout(self) -> None:
        exc = wrap_exception(LLMRunnerTimeoutError("timed out"))

        self.assertEqual(
            format_pipeline_error(exc),
            "LLM provider timed out. Please try again later.",
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
