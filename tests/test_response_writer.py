"""Tests for deterministic customer response drafting."""

from __future__ import annotations

import unittest

from app.input import BriefInputFactory
from app.pipeline import BriefAnalysisResultError, ResponseWriterStage
from app.schemas import (
    AIContext,
    ArbitrationResult,
    AssessmentRecommendation,
    AssessmentResult,
    AssessmentTechnicalInfo,
    ClarificationQuestion,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    DecisionStatus,
    ExtractedBrief,
    ExtractedFact,
    FactStatus,
    MVPPlan,
    MVPPlanningResult,
    MVPPlanningTechnicalInfo,
    QuestionGenerationResult,
    QuestionGenerationTechnicalInfo,
    Risk,
    RiskSeverity,
)


def make_context(status: DecisionStatus = DecisionStatus.accept) -> AIContext:
    """Create a completed analysis context before response writing."""
    brief = BriefInputFactory().from_text(
        "Нужно сделать сайт для образовательного проекта."
    )
    extracted = ExtractedBrief(
        project_goal=ExtractedFact(
            status=FactStatus.explicit,
            value="Сделать сайт для образовательного проекта",
        ),
        tasks=[
            ExtractedFact(status=FactStatus.explicit, value="Создать главную страницу")
        ],
        project_type=ExtractedFact(
            status=FactStatus.explicit,
            value="education",
        ),
        project_direction=ExtractedFact(
            status=FactStatus.explicit,
            value="development",
        ),
        materials=[ExtractedFact(status=FactStatus.explicit, value="контент")],
        expected_result=ExtractedFact(status=FactStatus.explicit, value="сайт"),
    )
    completeness = CompletenessResult(
        is_complete=status is not DecisionStatus.clarify,
        missing_information=(
            [
                CompletenessItem(
                    field_key="materials",
                    field_path="materials",
                    title="Available materials",
                    status=CompletenessStatus.missing,
                )
            ]
            if status is DecisionStatus.clarify
            else []
        ),
        present_information=[],
        clarification_information=[],
    )
    assessment = AssessmentResult(
        criterion_evaluations=[
            CriterionEvaluation(
                criterion="goal_clarity",
                criterion_title="Goal clarity",
                status=CriterionEvaluationStatus.met,
                explanation="Цель описана.",
            )
        ],
        risks=(
            [
                Risk(
                    type="scope_too_large",
                    description="Слишком широкий объём.",
                    severity=RiskSeverity.high,
                )
            ]
            if status is DecisionStatus.simplify
            else []
        ),
        evidence=[],
        has_risks=status is DecisionStatus.simplify,
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        summary="Сайт для образовательного проекта.",
        confidence=0.8,
        technical_info=AssessmentTechnicalInfo(),
    )
    arbitration = ArbitrationResult(
        final_status=status,
        reasons=["Тестовое основание."],
        evidence=[],
        triggered_rules=[],
        confidence=0.9,
        metadata={},
    )
    questions = QuestionGenerationResult(
        questions=[
            ClarificationQuestion(
                question="Какие материалы уже есть?",
                related_field="materials",
                reason="Нужны материалы.",
            )
        ],
        technical_info=QuestionGenerationTechnicalInfo(
            llm_invoked=False,
            attempts=0,
            trace_enabled=False,
            trace_name="question_generator.template",
        ),
    )
    return (
        AIContext.from_brief(brief)
        .with_extracted_brief(extracted)
        .with_completeness_result(completeness)
        .with_assessment_result(assessment)
        .with_arbitration_result(arbitration)
        .with_clarification_result(questions)
    )


def make_empty_question_result() -> QuestionGenerationResult:
    """Create an empty clarification result for characterization tests."""
    return QuestionGenerationResult(
        questions=[],
        technical_info=QuestionGenerationTechnicalInfo(
            llm_invoked=False,
            attempts=0,
            trace_enabled=False,
            trace_name="question_generator.template",
        ),
    )


def make_mvp_planning_result() -> MVPPlanningResult:
    """Create a minimal MVP planning result for characterization tests."""
    return MVPPlanningResult(
        plan=MVPPlan(
            core_goal="Build a focused website MVP",
            keep=["Public landing page"],
            remove=["Advanced analytics"],
            simplify=["Use one template"],
            mvp_scope=["Landing page and contact form"],
            rationale=["Keep first release small"],
        ),
        technical_info=MVPPlanningTechnicalInfo(
            llm_invoked=True,
            attempts=1,
            prompt_name="mvp_planner.md",
            trace_enabled=False,
            trace_name="mvp_planner.brief",
        ),
    )


class TestResponseWriterStage(unittest.TestCase):
    """Unit tests for deterministic response writer."""

    def test_writes_response_and_public_payload(self) -> None:
        updated = ResponseWriterStage().run_context(make_context())

        self.assertIsNotNone(updated.final_response_text)
        self.assertIsNotNone(updated.final_response_payload)
        self.assertEqual(updated.final_response_payload["assessment"]["recommendation"], "accept")
        self.assertEqual(
            updated.final_response_payload["extracted_fields"]["direction"],
            "development",
        )

    def test_clarify_response_includes_questions(self) -> None:
        updated = ResponseWriterStage().run_context(
            make_context(DecisionStatus.clarify)
        )

        self.assertIn("Какие материалы уже есть?", updated.final_response_text)
        self.assertEqual(
            updated.final_response_payload["clarifying_questions"],
            ["Какие материалы уже есть?"],
        )

    def test_clarify_public_payload_requires_questions(self) -> None:
        context = make_context(DecisionStatus.clarify).with_clarification_result(
            make_empty_question_result()
        )

        with self.assertRaises(BriefAnalysisResultError):
            ResponseWriterStage().run_context(context)

    def test_simplify_public_payload_requires_mvp_plan(self) -> None:
        context = make_context(DecisionStatus.simplify)

        with self.assertRaises(BriefAnalysisResultError):
            ResponseWriterStage().run_context(context)

    def test_simplify_public_payload_includes_mvp_plan(self) -> None:
        context = make_context(DecisionStatus.simplify).with_mvp_planning_result(
            make_mvp_planning_result()
        )

        updated = ResponseWriterStage().run_context(context)

        self.assertEqual(
            updated.final_response_payload["assessment"]["recommendation"],
            "simplify",
        )
        self.assertIn(
            "Build a focused website MVP",
            updated.final_response_payload["mvp_suggestion"],
        )

    def test_non_simplify_public_payload_ignores_stale_mvp_plan(self) -> None:
        context = make_context(DecisionStatus.accept).with_mvp_planning_result(
            make_mvp_planning_result()
        )

        updated = ResponseWriterStage().run_context(context)

        self.assertEqual(
            updated.final_response_payload["assessment"]["recommendation"],
            "accept",
        )
        self.assertEqual(updated.final_response_payload["mvp_suggestion"], "")

    def test_reject_and_mentor_review_public_payload_do_not_require_mvp_plan(self) -> None:
        for status in (DecisionStatus.reject, DecisionStatus.mentor_review):
            with self.subTest(status=status):
                updated = ResponseWriterStage().run_context(make_context(status))

                self.assertEqual(
                    updated.final_response_payload["assessment"]["recommendation"],
                    status.value.lower(),
                )
                self.assertEqual(updated.final_response_payload["mvp_suggestion"], "")


if __name__ == "__main__":
    unittest.main()
