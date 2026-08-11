"""Tests for the MVP planner."""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from typing import Any

from app.input import BriefInputFactory
from app.pipeline import MVPPlannerError, MVPPlannerStage
from app.schemas import (
    AIContext,
    ArbitrationResult,
    AssessmentRecommendation,
    AssessmentResult,
    AssessmentTechnicalInfo,
    BriefInput,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    DecisionStatus,
    ExtractedBrief,
    ExtractedFact,
    FactStatus,
    MVPPlanningResult,
    Risk,
    RiskSeverity,
)


class RecordingTraceContext:
    """Simple trace/span context used by unit tests."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class RecordingTracingClient:
    """Tracing double that records trace and span updates."""

    def __init__(self) -> None:
        self.trace = RecordingTraceContext()
        self.span = RecordingTraceContext()
        self.trace_calls: list[dict[str, Any]] = []
        self.span_calls: list[dict[str, Any]] = []

    @contextmanager
    def create_trace(self, name: str, **kwargs: Any):
        self.trace_calls.append({"name": name, **kwargs})
        yield self.trace

    @contextmanager
    def create_span(self, name: str, **kwargs: Any):
        self.span_calls.append({"name": name, **kwargs})
        yield self.span

    def flush(self) -> None:
        return None


class FakeLLMClient:
    """Deterministic LLM client used for MVP planner unit tests."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate_json(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, **kwargs})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response

        return response

    def stream(self, messages: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


def make_brief_input() -> BriefInput:
    """Create a brief input for tests."""
    return BriefInputFactory().from_text(
        "Build a customer portal with authentication and reporting."
    )


def make_extracted_brief() -> ExtractedBrief:
    """Create a minimal extracted brief for tests."""
    return ExtractedBrief(
        project_goal=ExtractedFact(
            status=FactStatus.explicit,
            value="Build a customer portal",
            evidence=["Build a customer portal."],
            confidence=0.95,
            notes=None,
        ),
        tasks=[
            ExtractedFact(
                status=FactStatus.explicit,
                value="Implement authentication",
                evidence=["authentication"],
                confidence=0.9,
                notes=None,
            )
        ],
        project_type=ExtractedFact(
            status=FactStatus.explicit,
            value="web_app",
            evidence=["portal"],
            confidence=0.9,
            notes=None,
        ),
        project_direction=ExtractedFact(
            status=FactStatus.explicit,
            value="product development",
            evidence=["product development"],
            confidence=0.9,
            notes=None,
        ),
        technologies=[],
        stack=[],
        materials=[],
        expected_result=ExtractedFact(
            status=FactStatus.explicit,
            value="A working portal",
            evidence=["working portal"],
            confidence=0.9,
            notes=None,
        ),
        constraints=[],
        deadlines=[],
        existing_resources=[],
        integrations=[],
        other_facts=[],
    )


def make_assessment_result() -> AssessmentResult:
    """Create a unified assessment result with traceable planning inputs."""
    return AssessmentResult(
        criterion_evaluations=[
            CriterionEvaluation(
                criterion="scope_definition",
                criterion_title="Scope definition",
                status=CriterionEvaluationStatus.risk_detected,
                evidence=["Reporting and authentication in one first release."],
                explanation="The first release scope combines multiple major features.",
                confidence=0.82,
                notes=None,
            )
        ],
        risks=[
            Risk(
                type="scope_too_large",
                description="Initial scope includes reporting and authentication.",
                severity=RiskSeverity.high,
                evidence=["authentication and reporting"],
                confidence=0.84,
                notes=None,
            )
        ],
        evidence=[],
        has_risks=True,
        recommendation=AssessmentRecommendation.high_risk_review,
        summary="The brief needs simplification before implementation planning.",
        confidence=0.86,
        technical_info=AssessmentTechnicalInfo(
            attempts=1,
            prompt_name="assessment.md",
            trace_enabled=False,
            trace_name="assessment.brief",
            model_name="assessment-model",
            retriever_used=False,
            retrieved_context_count=0,
            criteria_count=1,
            risk_types_count=4,
            raw_response=None,
            recovered_errors=[],
            provider_metadata={"provider": "fake"},
        ),
    )


def make_empty_assessment_result() -> AssessmentResult:
    """Create empty assessment output for planner branch tests."""
    return AssessmentResult(
        criterion_evaluations=[],
        risks=[],
        evidence=[],
        has_risks=False,
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        summary=None,
        confidence=None,
        technical_info=AssessmentTechnicalInfo(
            attempts=1,
            prompt_name="assessment.md",
            trace_enabled=False,
            trace_name="assessment.brief",
            model_name=None,
            retriever_used=False,
            retrieved_context_count=0,
            criteria_count=0,
            risk_types_count=0,
            raw_response=None,
            recovered_errors=[],
            provider_metadata={},
        ),
    )


def make_arbitration_result(status: DecisionStatus) -> ArbitrationResult:
    """Create an arbitration result for tests."""
    return ArbitrationResult(
        final_status=status,
        reasons=["Synthetic test decision."],
        evidence=["Synthetic evidence."],
        triggered_rules=[],
        confidence=0.8,
        metadata={},
    )


def make_valid_payload() -> dict[str, Any]:
    """Return a valid MVP planning payload."""
    return {
        "core_goal": "Build a customer portal",
        "keep": [
            "Authentication for internal users",
            "Basic reporting dashboard",
        ],
        "remove": [
            "Advanced analytics module",
        ],
        "simplify": [
            "Reduce reporting to a single dashboard",
        ],
        "mvp_scope": [
            "Login/logout flow",
            "One dashboard with core metrics",
        ],
        "rationale": [
            "This keeps the original goal while removing high-complexity extras.",
        ],
    }


class TestMVPPlannerStage(unittest.TestCase):
    """Unit tests for the MVP planner."""

    def test_does_not_call_llm_when_status_is_not_simplify(self) -> None:
        llm_client = FakeLLMClient([])
        planner = MVPPlannerStage(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            model_name="test-model",
        )

        result = planner.plan_assessment(
            make_brief_input(),
            make_extracted_brief(),
            make_empty_assessment_result(),
            make_arbitration_result(DecisionStatus.accept),
        )

        self.assertIsNone(result.plan)
        self.assertFalse(result.technical_info.llm_invoked)
        self.assertEqual(result.technical_info.attempts, 0)
        self.assertEqual(len(llm_client.calls), 0)
        self.assertIsNotNone(result.technical_info.skipped_reason)
        self.assertIsInstance(planner, MVPPlannerStage)

    def test_generates_mvp_plan_for_simplify_status(self) -> None:
        llm_client = FakeLLMClient([make_valid_payload()])
        tracing_client = RecordingTracingClient()
        planner = MVPPlannerStage(
            llm_client=llm_client,
            tracing_client=tracing_client,
            model_name="test-model",
        )

        result = planner.plan_assessment(
            make_brief_input(),
            make_extracted_brief(),
            make_empty_assessment_result(),
            make_arbitration_result(DecisionStatus.simplify),
        )

        self.assertIsInstance(result, MVPPlanningResult)
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.core_goal, "Build a customer portal")
        self.assertEqual(len(result.plan.keep), 2)
        self.assertTrue(result.technical_info.llm_invoked)
        self.assertEqual(result.technical_info.attempts, 1)
        self.assertEqual(result.technical_info.model_name, "test-model")
        self.assertEqual(len(llm_client.calls), 1)
        self.assertEqual(tracing_client.trace_calls[0]["name"], "mvp_planner.brief")
        self.assertEqual(tracing_client.span_calls[0]["name"], "mvp_planner.llm")

    def test_retries_on_invalid_llm_payload(self) -> None:
        llm_client = FakeLLMClient(
            [
                {"keep": [], "remove": [], "simplify": [], "mvp_scope": [], "rationale": []},
                make_valid_payload(),
            ]
        )
        planner = MVPPlannerStage(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            model_name="test-model",
            max_retries=2,
        )

        result = planner.plan_assessment(
            make_brief_input(),
            make_extracted_brief(),
            make_empty_assessment_result(),
            make_arbitration_result(DecisionStatus.simplify),
        )

        self.assertEqual(result.technical_info.attempts, 2)
        self.assertEqual(len(result.technical_info.recovered_errors), 1)
        self.assertIsNotNone(result.plan)

    def test_run_context_uses_assessment_result_with_legacy_prompt_shape(self) -> None:
        llm_client = FakeLLMClient([make_valid_payload()])
        planner = MVPPlannerStage(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            model_name="test-model",
        )
        context = (
            AIContext.from_brief(make_brief_input())
            .with_extracted_brief(make_extracted_brief())
            .with_assessment_result(make_assessment_result())
            .with_arbitration_result(make_arbitration_result(DecisionStatus.simplify))
        )

        self.assertIsNotNone(context.assessment_result)
        self.assertFalse(hasattr(context, "risk_analysis_result"))
        self.assertFalse(hasattr(context, "evaluation_result"))

        updated = planner.run_context(context)

        self.assertEqual(len(llm_client.calls), 1)
        self.assertIsNotNone(updated.mvp_planning_result)
        self.assertIsNotNone(updated.mvp_planning_result.plan)

        messages = llm_client.calls[0]["messages"]
        user_message = messages[1]["content"]
        planning_json = user_message.split(
            "Create a minimal simplification plan from this structured context:",
            maxsplit=1,
        )[1].strip()
        planning_context = json.loads(planning_json)

        self.assertIn("risk_analysis_result", planning_context)
        self.assertIn("evaluation_result", planning_context)
        self.assertNotIn("assessment_result", planning_context)

        risk = planning_context["risk_analysis_result"]["risks"][0]
        self.assertEqual(risk["type"], "scope_too_large")
        self.assertEqual(
            risk["description"],
            "Initial scope includes reporting and authentication.",
        )
        self.assertEqual(risk["severity"], "high")
        self.assertEqual(risk["evidence"], ["authentication and reporting"])

        evaluation = planning_context["evaluation_result"][
            "criterion_evaluations"
        ][0]
        self.assertEqual(evaluation["criterion"], "scope_definition")
        self.assertEqual(evaluation["status"], "risk_detected")
        self.assertEqual(
            evaluation["evidence"],
            ["Reporting and authentication in one first release."],
        )
        self.assertEqual(
            evaluation["explanation"],
            "The first release scope combines multiple major features.",
        )

        expected_summary = (
            "The brief needs simplification before implementation planning."
        )
        self.assertEqual(
            planning_context["risk_analysis_result"]["summary"],
            expected_summary,
        )
        self.assertEqual(
            planning_context["evaluation_result"]["summary"],
            expected_summary,
        )

    def test_raises_when_all_retries_fail(self) -> None:
        llm_client = FakeLLMClient(
            [
                RuntimeError("temporary failure"),
                RuntimeError("temporary failure"),
            ]
        )
        planner = MVPPlannerStage(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            model_name="test-model",
            max_retries=2,
        )

        with self.assertRaises(MVPPlannerError):
            planner.plan_assessment(
                make_brief_input(),
                make_extracted_brief(),
                make_empty_assessment_result(),
                make_arbitration_result(DecisionStatus.simplify),
            )


if __name__ == "__main__":
    unittest.main()
