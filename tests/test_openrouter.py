"""Tests for OpenRouter provider adapter behavior."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
