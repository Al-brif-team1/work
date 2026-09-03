"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import unittest
from typing import Any

from app.config import CriteriaLoader
from app.input import BriefInputFactory
from app.llm.runner import LLMRunResult, LLMTokenUsage
from app.pipeline import CompletenessCheckStage, DeterministicArbiterStage, Extractor
from app.schemas import (
    AIContext,
    AssessmentRecommendation,
    AssessmentResult,
    AssessmentTechnicalInfo,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    ExtractedBrief,
    ExtractedFact,
    FactStatus,
    TrafficLightResult,
    TrafficLightStatus,
)
from app.tracing.tracing import NoOpTracingClient


class FakeLLMRunner:
    """Класс «FakeLLMRunner» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, payload: ExtractedBrief) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def run_json(self, **kwargs: Any) -> LLMRunResult[ExtractedBrief]:
        self.calls.append(kwargs)
        return LLMRunResult(
            payload=self.payload,
            raw_response={},
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

    def run(self, **kwargs: Any) -> LLMRunResult[ExtractedBrief]:
        self.calls.append(kwargs)
        return LLMRunResult(
            payload=self.payload,
            raw_response={},
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


def make_extracted_brief() -> ExtractedBrief:
    """Выполняет шаг «make extracted brief». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return ExtractedBrief(
        project_goal=ExtractedFact(
            status=FactStatus.explicit,
            value="Build a portal",
            evidence=["Build a portal"],
            confidence=0.9,
        ),
        tasks=[
            ExtractedFact(
                status=FactStatus.explicit,
                value="Build landing page",
                evidence=["Build landing page"],
                confidence=0.9,
            )
        ],
        project_type=ExtractedFact(
            status=FactStatus.explicit,
            value="development",
            evidence=["development"],
            confidence=0.9,
        ),
        project_direction=ExtractedFact(
            status=FactStatus.explicit,
            value="product development",
            evidence=["product development"],
            confidence=0.8,
        ),
        expected_result=ExtractedFact(
            status=FactStatus.explicit,
            value="Working portal",
            evidence=["Working portal"],
            confidence=0.9,
        ),
    )


def make_complete_result() -> CompletenessResult:
    """Выполняет шаг «make complete result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return CompletenessResult(
        is_complete=True,
        missing_information=[],
        present_information=[
            CompletenessItem(
                field_key="placeholder_field",
                field_path="project_goal",
                title="Project goal",
                status=CompletenessStatus.present,
                value="Build a portal",
                reason=None,
            )
        ],
        clarification_information=[],
        warnings=[],
    )


def make_assessment_result() -> AssessmentResult:
    """Выполняет шаг «make assessment result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return AssessmentResult(
        criterion_evaluations=[
            CriterionEvaluation(
                criterion="placeholder_criterion",
                criterion_title="Goal clarity",
                status=CriterionEvaluationStatus.met,
                evidence=["Build a portal"],
                confidence=0.9,
            )
        ],
        risks=[],
        evidence=[],
        has_risks=False,
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        confidence=0.9,
        traffic_light=TrafficLightResult(status=TrafficLightStatus.green),
        technical_info=AssessmentTechnicalInfo(
            attempts=1,
            prompt_name="assessment.md",
            criteria_count=1,
            risk_types_count=1,
        ),
    )


class TestPipelineArchitecture(unittest.TestCase):
    """Класс «TestPipelineArchitecture» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_llm_stage_can_use_injected_runner_without_client(self) -> None:
        runner = FakeLLMRunner(make_extracted_brief())
        extractor = Extractor(
            llm_runner=runner,
            tracing_client=NoOpTracingClient(),
        )
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a portal"))

        updated = extractor.run_context(context)

        self.assertEqual(updated.extracted_brief, runner.payload)
        self.assertEqual(runner.calls[0]["trace_name"], "extractor.brief")

    def test_deterministic_stages_accept_assessment_result_from_context(self) -> None:
        context = (
            AIContext.from_brief(BriefInputFactory().from_text("Build a portal"))
            .with_extracted_brief(make_extracted_brief())
            .with_completeness_result(make_complete_result())
            .with_assessment_result(make_assessment_result())
        )
        arbiter = DeterministicArbiterStage(criteria_config=CriteriaLoader.load())

        updated = arbiter.run_context(context)

        self.assertIsNotNone(updated.arbitration_result)
        self.assertEqual(updated.arbitration_result.final_status.value, "ACCEPT")

    def test_completeness_checker_supports_context_contract(self) -> None:
        context = AIContext.from_brief(
            BriefInputFactory().from_text("Build a portal")
        ).with_extracted_brief(make_extracted_brief())
        checker = CompletenessCheckStage(criteria_config=CriteriaLoader.load())

        updated = checker.run_context(context)

        self.assertIsNotNone(updated.completeness_result)
        self.assertTrue(updated.completeness_result.is_complete)


if __name__ == "__main__":
    unittest.main()
