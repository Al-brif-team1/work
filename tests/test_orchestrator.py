"""Tests for end-to-end pipeline orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import unittest

from app.pipeline import (
    AssessmentStage,
    BriefAnalysisPipeline,
    CompletenessCheckStage,
    DeterministicArbiterStage,
    Extractor,
    LLMSelfChecker,
    MVPPlannerStage,
    ResponseWriterStage,
    SelfChecker,
    TemplateQuestionGeneratorStage,
)
from app.schemas import (
    AIContext,
    ArbitrationResult,
    AssessmentRecommendation,
    AssessmentPayload,
    AssessmentResult,
    AssessmentTechnicalInfo,
    BriefAnalysisResult,
    CompletenessResult,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    DecisionStatus,
    ExtractedBrief,
    ExtractedFact,
    FactStatus,
)


class ExtractionStageStub:
    """Pipeline stage stub that adds extracted data."""

    def run_context(self, context: AIContext) -> AIContext:
        """Add a deterministic extracted brief."""
        return context.with_extracted_brief(
            ExtractedBrief(
                project_goal=ExtractedFact(
                    status=FactStatus.explicit,
                    value="Сделать сайт",
                ),
                tasks=[ExtractedFact(status=FactStatus.explicit, value="Главная")],
                project_type=ExtractedFact(
                    status=FactStatus.explicit,
                    value="education",
                ),
                project_direction=ExtractedFact(
                    status=FactStatus.explicit,
                    value="development",
                ),
                expected_result=ExtractedFact(
                    status=FactStatus.explicit,
                    value="Рабочий сайт",
                ),
            )
        )


class CompletenessStageStub:
    """Pipeline stage stub that adds completeness data."""

    def run_context(self, context: AIContext) -> AIContext:
        """Mark the brief as complete."""
        return context.with_completeness_result(
            CompletenessResult(
                is_complete=True,
                missing_information=[],
                present_information=[],
                clarification_information=[],
            )
        )


class AssessmentStageStub:
    """Pipeline stage stub that adds assessment data."""

    def run_context(self, context: AIContext) -> AIContext:
        """Add an accept-ready assessment."""
        return context.with_assessment_result(
            AssessmentResult(
                criterion_evaluations=[
                    CriterionEvaluation(
                        criterion="goal_clarity",
                        status=CriterionEvaluationStatus.met,
                        explanation="Цель понятна.",
                    )
                ],
                risks=[],
                evidence=[],
                has_risks=False,
                recommendation=AssessmentRecommendation.ready_for_arbitration,
                summary="Образовательный сайт.",
                confidence=0.9,
                technical_info=AssessmentTechnicalInfo(),
            )
        )


class ArbiterStageStub:
    """Pipeline stage stub that adds arbitration data."""

    def run_context(self, context: AIContext) -> AIContext:
        """Add an ACCEPT arbitration result."""
        return context.with_arbitration_result(
            ArbitrationResult(
                final_status=DecisionStatus.accept,
                reasons=["Проект реалистичен."],
                evidence=[],
                triggered_rules=[],
                confidence=0.95,
                metadata={},
            )
        )


class FakeProductionLLMClient:
    """Minimal fake for factory integration tests."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise AssertionError("Production pipeline stages should request structured JSON in this test.")

    def generate_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("Unexpected LLM call.")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterable[str]:
        raise AssertionError("Streaming is not used by the production pipeline factory.")


def _minimal_extraction_payload() -> dict[str, Any]:
    return {
        "project_goal": {
            "status": "explicit",
            "value": "Create a small support portal for customer requests.",
            "evidence": ["Need a support portal for requests."],
            "confidence": 0.95,
        },
        "tasks": [
            {
                "status": "explicit",
                "value": "Collect support requests through a web form.",
                "evidence": ["The portal should collect requests."],
                "confidence": 0.95,
            }
        ],
        "project_type": {
            "status": "explicit",
            "value": "web_app",
            "evidence": ["support portal"],
            "confidence": 0.9,
        },
        "project_direction": {
            "status": "explicit",
            "value": "development",
            "evidence": ["Create a small support portal."],
            "confidence": 0.9,
        },
        "expected_result": {
            "status": "explicit",
            "value": "A working MVP with a request form and confirmation screen.",
            "evidence": ["working MVP"],
            "confidence": 0.95,
        },
        "materials": [
            {
                "status": "explicit",
                "value": "Use the existing product copy.",
                "evidence": ["existing product copy"],
                "confidence": 0.8,
            }
        ],
        "technologies": [],
        "stack": [],
        "constraints": [],
        "deadlines": [],
        "existing_resources": [],
        "integrations": [],
        "other_facts": [],
    }


def _ready_assessment_payload() -> dict[str, Any]:
    return AssessmentPayload(
        criterion_evaluations=[
            CriterionEvaluation(
                criterion="goal_clarity",
                status=CriterionEvaluationStatus.met,
                explanation="The goal is explicit.",
            ),
            CriterionEvaluation(
                criterion="expected_result",
                status=CriterionEvaluationStatus.met,
                explanation="The expected MVP result is explicit.",
            ),
            CriterionEvaluation(
                criterion="scope_definition",
                status=CriterionEvaluationStatus.met,
                explanation="The initial scope is narrow enough.",
            ),
        ],
        risks=[],
        evidence=[
            {
                "source": "brief",
                "quote": "Need a small web support portal for customer requests.",
                "related_criteria": ["goal_clarity", "expected_result", "scope_definition"],
                "confidence": 0.9,
            }
        ],
        has_risks=False,
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        summary="The brief is ready for implementation planning.",
        confidence=0.9,
    ).model_dump(mode="json")


def _build_factory_pipeline(fake_client: FakeProductionLLMClient) -> BriefAnalysisPipeline:
    return BriefAnalysisPipeline.from_llm_client(
        fake_client,
        model_name="fake-model",
    )


class TestBriefAnalysisPipeline(unittest.TestCase):
    """Unit tests for the orchestrator contract."""

    def test_production_factory_uses_only_active_stages(self) -> None:
        pipeline = _build_factory_pipeline(FakeProductionLLMClient([]))

        expected_stage_types = (
            Extractor,
            CompletenessCheckStage,
            AssessmentStage,
            DeterministicArbiterStage,
            TemplateQuestionGeneratorStage,
            MVPPlannerStage,
            ResponseWriterStage,
        )
        # This is a white-box architecture contract: factory order is part of the migration safety net.
        self.assertEqual(tuple(type(stage) for stage in pipeline._stages), expected_stage_types)

        legacy_stage_types = (
            SelfChecker,
            LLMSelfChecker,
        )
        self.assertFalse(any(isinstance(stage, legacy_stage_types) for stage in pipeline._stages))

    def test_production_factory_runs_end_to_end_with_fake_llm(self) -> None:
        fake_client = FakeProductionLLMClient(
            [
                _minimal_extraction_payload(),
                _ready_assessment_payload(),
            ]
        )
        pipeline = _build_factory_pipeline(fake_client)

        result = pipeline.analyze_text(
            "Need a small web support portal for customer requests. "
            "It should include a request form and confirmation screen."
        )

        self.assertIsInstance(result, BriefAnalysisResult)
        self.assertEqual(result.assessment.recommendation, "accept")
        self.assertTrue(result.customer_response_draft.strip())
        self.assertEqual(result.clarifying_questions, [])
        self.assertEqual(result.mvp_suggestion, "")
        self.assertEqual(len(fake_client.calls), 2)

    def test_analyze_text_returns_public_result(self) -> None:
        pipeline = BriefAnalysisPipeline(
            stages=[
                ExtractionStageStub(),
                CompletenessStageStub(),
                AssessmentStageStub(),
                ArbiterStageStub(),
                ResponseWriterStage(),
            ]
        )

        result = pipeline.analyze_text("Нужно сделать сайт для образовательного проекта.")

        self.assertEqual(result.assessment.recommendation, "accept")
        self.assertEqual(result.extracted_fields.goal, "Сделать сайт")
        self.assertIn("студенческий проект", result.customer_response_draft)


if __name__ == "__main__":
    unittest.main()
