"""Tests for deterministic empty/nonsense brief rejection."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any
from unittest.mock import patch

from app.config import Settings
from app.input import BriefInputFactory
from app.llm.client import LLMClient
from app.pipeline import (
    EMPTY_OR_NONSENSE_BRIEF_REASON_CODE,
    EMPTY_OR_NONSENSE_BRIEF_RULE_KEY,
    BriefAnalysisPipeline,
    EmptyBriefRejectionStage,
    is_empty_or_obvious_nonsense,
)
from app.main import run
from app.schemas import AIContext, DecisionStatus


class FailingLLMClient(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: Any, **kwargs: Any) -> str:  # pragma: no cover
        self.calls += 1
        raise AssertionError("LLM must not be called")

    def generate_json(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise AssertionError("LLM must not be called")

    def stream(self, messages: Any, **kwargs: Any):  # pragma: no cover
        self.calls += 1
        raise AssertionError("LLM must not be called")


class RecordingStage:
    def __init__(self) -> None:
        self.calls = 0

    def run_context(self, context: AIContext) -> AIContext:
        self.calls += 1
        return context


class TestEmptyBriefRejection(unittest.TestCase):
    def test_empty_string_is_rejected(self) -> None:
        self._assert_rejected("")

    def test_whitespace_string_is_rejected(self) -> None:
        self._assert_rejected("     ")

    def test_single_letter_is_rejected(self) -> None:
        self._assert_rejected("а")

    def test_symbol_only_string_is_rejected(self) -> None:
        self._assert_rejected("!!!@@@###")

    def test_random_letter_token_is_not_deterministically_rejected(self) -> None:
        self.assertFalse(is_empty_or_obvious_nonsense("xqzjkkqwr"))

    def test_short_meaningful_briefs_are_not_nonsense(self) -> None:
        for text in ("Нужен сайт", "Сделать логотип", "Telegram-бот", "Нужна презентация"):
            with self.subTest(text=text):
                self.assertFalse(is_empty_or_obvious_nonsense(text))

    def test_empty_brief_short_circuits_downstream_stages(self) -> None:
        downstream = RecordingStage()
        pipeline = BriefAnalysisPipeline(
            stages=[EmptyBriefRejectionStage(), downstream],
        )

        context = pipeline.run_context(BriefInputFactory().from_text(""))

        self.assertEqual(downstream.calls, 0)
        self.assertIsNotNone(context.final_response_payload)

    def test_empty_brief_does_not_call_downstream_llm_stages(self) -> None:
        llm_client = FailingLLMClient()
        settings = Settings(
            LLM_API_KEY="test-key",
            LLM_MODEL="test-model",
            LLM_BASE_URL="https://example.invalid",
        )
        pipeline = BriefAnalysisPipeline.from_llm_client(
            llm_client,
            settings=settings,
        )

        context = pipeline.run_context(BriefInputFactory().from_text(""))

        self.assertEqual(llm_client.calls, 0)
        self.assertEqual(
            context.arbitration_result.final_status,
            DecisionStatus.reject,
        )
        self.assertEqual(context.extraction_result.technical_info.attempts, 0)
        self.assertEqual(context.assessment_result.technical_info.attempts, 0)
        self.assertIsNone(context.mvp_planning_result)

    def test_empty_brief_result_is_not_polluted_by_lower_priority_reasons(self) -> None:
        context = self._reject_context("")
        payload = context.final_response_payload

        self.assertEqual(
            context.arbitration_result.triggered_rules[0].rule_key,
            EMPTY_OR_NONSENSE_BRIEF_RULE_KEY,
        )
        self.assertEqual(
            context.arbitration_result.metadata["reason_code"],
            EMPTY_OR_NONSENSE_BRIEF_REASON_CODE,
        )
        self.assertEqual(payload["assessment"]["recommendation"], "reject")
        self.assertEqual(payload["extracted_fields"]["missing_information"], [])
        self.assertEqual(payload["clarifying_questions"], [])
        self.assertEqual(context.assessment_result.risks, [])
        self.assertEqual(payload["assessment"]["risks"], [])
        self.assertEqual(
            context.arbitration_result.triggered_rules[0].metadata["reason_code"],
            EMPTY_OR_NONSENSE_BRIEF_REASON_CODE,
        )
        lower_priority_codes = {
            "missing_information",
            "scope_too_large",
            "mentor_expertise_required",
            "restricted_topic",
            "out_of_scope_request",
        }
        self.assertTrue(
            lower_priority_codes.isdisjoint(
                set(context.arbitration_result.metadata.values())
            )
        )

    def test_cli_empty_text_returns_reject_business_result(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        settings = Settings(
            LLM_API_KEY="test-key",
            LLM_MODEL="test-model",
            LLM_BASE_URL="https://example.invalid",
        )
        llm_client = FailingLLMClient()

        with (
            patch("app.main.Config.load", return_value=settings),
            patch("app.main.LLMClientFactory.create", return_value=llm_client),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = run(["--text", ""])

        self.assertEqual(exit_code, 0)
        self.assertEqual(llm_client.calls, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["assessment"]["recommendation"], "reject")
        self.assertEqual(payload["assessment"]["risks"], [])
        self.assertIn("блокирующего", payload["assessment"]["reasons"][0])

    def _assert_rejected(self, text: str) -> None:
        context = self._reject_context(text)

        self.assertEqual(
            context.arbitration_result.final_status,
            DecisionStatus.reject,
        )
        self.assertEqual(
            context.arbitration_result.triggered_rules[0].rule_key,
            EMPTY_OR_NONSENSE_BRIEF_RULE_KEY,
        )

    @staticmethod
    def _reject_context(text: str) -> AIContext:
        pipeline = BriefAnalysisPipeline(stages=[EmptyBriefRejectionStage()])
        return pipeline.run_context(BriefInputFactory().from_text(text))


if __name__ == "__main__":
    unittest.main()
