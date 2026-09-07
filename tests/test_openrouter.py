"""Tests for OpenRouter provider adapter behavior."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.llm.client import LLMStructuredOutputError
from app.llm.openrouter import OpenRouterLLMClient


class TestOpenRouterLLMClient(unittest.TestCase):
    def test_retryable_status_codes_are_classified(self) -> None:
        for status_code in (408, 429, 500, 502, 503, 504):
            with self.subTest(status_code=status_code):
                self.assertTrue(
                    OpenRouterLLMClient._is_retryable_status(status_code)
                )

    def test_non_retryable_status_codes_are_classified(self) -> None:
        for status_code in (400, 401, 402, 403, 404, 422):
            with self.subTest(status_code=status_code):
                self.assertFalse(
                    OpenRouterLLMClient._is_retryable_status(status_code)
                )

    def test_empty_content_is_structured_output_error_with_metadata(self) -> None:
        client = object.__new__(OpenRouterLLMClient)
        client._create_completion = lambda messages, **kwargs: self._response(
            "",
            finish_reason="stop",
        )

        with self.assertRaises(LLMStructuredOutputError) as context:
            client.generate_json([{"role": "user", "content": "hello"}])

        self.assertEqual(context.exception.error_kind, "empty_content")
        self.assertEqual(context.exception.content_length, 0)
        self.assertEqual(context.exception.provider_metadata["finish_reason"], "stop")
        self.assertEqual(context.exception.provider_metadata["model"], "test-model")
        self.assertEqual(context.exception.provider_metadata["id"], "response-1")
        self.assertEqual(
            context.exception.provider_metadata["usage"]["completion_tokens"],
            1,
        )

    def test_malformed_json_is_structured_output_error(self) -> None:
        client = object.__new__(OpenRouterLLMClient)
        client._create_completion = lambda messages, **kwargs: self._response(
            "not json",
            finish_reason="stop",
        )

        with self.assertRaises(LLMStructuredOutputError) as context:
            client.generate_json([{"role": "user", "content": "hello"}])

        self.assertEqual(context.exception.error_kind, "malformed_json")
        self.assertEqual(context.exception.content_length, 8)

    def test_probably_truncated_json_is_structured_output_error(self) -> None:
        client = object.__new__(OpenRouterLLMClient)
        client._create_completion = lambda messages, **kwargs: self._response(
            '{"tasks": ["unfinished',
            finish_reason="stop",
        )

        with self.assertRaises(LLMStructuredOutputError) as context:
            client.generate_json([{"role": "user", "content": "hello"}])

        self.assertEqual(context.exception.error_kind, "truncated_json")

    def test_length_finish_is_confirmed_length_error(self) -> None:
        client = object.__new__(OpenRouterLLMClient)
        client._create_completion = lambda messages, **kwargs: self._response(
            '{"tasks": ["unfinished',
            finish_reason="length",
        )

        with self.assertRaises(LLMStructuredOutputError) as context:
            client.generate_json([{"role": "user", "content": "hello"}])

        self.assertEqual(context.exception.error_kind, "length_finish")
        self.assertEqual(
            context.exception.provider_metadata["finish_reason"],
            "length",
        )

    @staticmethod
    def _response(content: str, *, finish_reason: str) -> SimpleNamespace:
        return SimpleNamespace(
            id="response-1",
            model="test-model",
            usage=SimpleNamespace(
                model_dump=lambda mode="json": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                }
            ),
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    message=SimpleNamespace(content=content),
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
