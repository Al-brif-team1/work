"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from typing import Any

from app.input import BriefInputFactory
from app.pipeline import DeterministicValidator, LLMSelfChecker, SelfChecker
from app.schemas import (
    AIContext,
    ArbitrationResult,
    AssessmentEvidence,
    AssessmentRecommendation,
    AssessmentResult,
    AssessmentTechnicalInfo,
    BriefInput,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    DecisionStatus,
    Document,
    DocumentMetadata,
    ExtractedBrief,
    ExtractedFact,
    FactStatus,
    Risk,
    RiskSeverity,
    SelfCheckContext,
    SelfCheckResult,
    SearchResult,
)


class RecordingTraceContext:
    """Класс «RecordingTraceContext» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class RecordingTracingClient:
    """Класс «RecordingTracingClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

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
    """Класс «FakeLLMClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

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
    """Выполняет шаг «make brief input». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return BriefInputFactory().from_text(
        "Build a customer portal for reporting and account access."
    )


def make_extracted_brief() -> ExtractedBrief:
    """Выполняет шаг «make extracted brief». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return ExtractedBrief(
        project_goal=ExtractedFact(
            status=FactStatus.explicit,
            value="Build a customer portal",
            evidence=["Build a customer portal."],
            confidence=0.95,
            notes=None,
        ),
        tasks=[],
        project_type=ExtractedFact(
            status=FactStatus.explicit,
            value="web_app",
            evidence=["customer portal"],
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
            value="Working portal",
            evidence=["Working portal"],
            confidence=0.9,
            notes=None,
        ),
        constraints=[],
        deadlines=[],
        existing_resources=[],
        integrations=[],
        other_facts=[],
    )


def make_completeness_result() -> CompletenessResult:
    """Выполняет шаг «make completeness result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    present_item = CompletenessItem(
        field_key="project_goal",
        field_path="project_goal",
        title="Project goal",
        status=CompletenessStatus.present,
        value="Build a customer portal",
        reason=None,
        notes=None,
    )
    return CompletenessResult(
        is_complete=True,
        missing_information=[],
        present_information=[present_item],
        clarification_information=[],
        warnings=[],
    )


def make_assessment_result() -> AssessmentResult:
    """Выполняет шаг «make assessment result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return AssessmentResult(
        criterion_evaluations=[
            CriterionEvaluation(
                criterion="goal_clarity",
                criterion_title="Goal clarity",
                status=CriterionEvaluationStatus.met,
                evidence=["Build a customer portal."],
                explanation="The goal is explicit and supported by the brief.",
                confidence=0.91,
                notes=None,
            )
        ],
        risks=[
            Risk(
                type="scope_too_large",
                description="Reporting and account access may exceed the first release.",
                severity=RiskSeverity.medium,
                evidence=["reporting and account access"],
                confidence=0.76,
                notes=None,
            )
        ],
        evidence=[],
        has_risks=True,
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        summary="The brief is clear but has moderate scope risk.",
        confidence=0.88,
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
    """Выполняет шаг «make empty assessment result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
    """Выполняет шаг «make arbitration result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return ArbitrationResult(
        final_status=status,
        reasons=["Synthetic decision."],
        evidence=["Synthetic evidence."],
        triggered_rules=[],
        confidence=0.9,
        metadata={},
    )


def make_context(
    status: DecisionStatus,
    response_text: str,
    response_payload: dict[str, Any] | None = None,
) -> SelfCheckContext:
    """Выполняет шаг «make context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return SelfCheckContext(
        response_text=response_text,
        response_payload=response_payload,
        brief_input=make_brief_input(),
        extracted_brief=make_extracted_brief(),
        completeness_result=make_completeness_result(),
        assessment_result=make_empty_assessment_result(),
        arbitration_result=make_arbitration_result(status),
        clarification_result=None,
        mvp_planning_result=None,
        retrieved_context=[],
    )


def make_llm_payload() -> dict[str, Any]:
    """Выполняет шаг «make llm payload». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return {
        "is_valid": True,
        "issues": [],
        "warnings": ["Semantic check passed."],
        "checked_fields": ["response_text", "response_payload"],
    }


class TestDeterministicValidator(unittest.TestCase):
    """Класс «TestDeterministicValidator» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_detects_status_mismatch(self) -> None:
        validator = DeterministicValidator()
        context = make_context(
            DecisionStatus.accept,
            "The project was accepted.",
            {"status": "REJECT", "facts": ["Build a customer portal"]},
        )

        result = validator.validate(context)

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("Response status does not match" in issue for issue in result.issues)
        )

    def test_requires_questions_for_clarify_status(self) -> None:
        validator = DeterministicValidator()
        context = make_context(
            DecisionStatus.clarify,
            "We need clarification.",
            {"status": "CLARIFY", "questions": []},
        )

        result = validator.validate(context)

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("clarification questions" in issue.lower() for issue in result.issues)
        )

    def test_allows_questions_for_accept_with_clarifications_status(self) -> None:
        validator = DeterministicValidator()
        context = make_context(
            DecisionStatus.accept_with_clarifications,
            "The project is accepted with clarifications.",
            {
                "status": "ACCEPT_WITH_CLARIFICATIONS",
                "questions": ["Which materials are available?"],
            },
        )

        result = validator.validate(context)

        self.assertTrue(result.is_valid)
        self.assertIn(
            "acceptance_with_clarifications_consistency",
            result.checked_fields,
        )

    def test_requires_mvp_plan_for_simplify_status(self) -> None:
        validator = DeterministicValidator()
        context = make_context(
            DecisionStatus.simplify,
            "The project should be simplified.",
            {"status": "SIMPLIFY"},
        )

        result = validator.validate(context)

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("mvp plan" in issue.lower() for issue in result.issues)
        )

    def test_detects_unsupported_fact_in_payload(self) -> None:
        validator = DeterministicValidator()
        context = make_context(
            DecisionStatus.accept,
            "The project is accepted.",
            {"status": "ACCEPT", "facts": ["Introduce GraphQL federation"]},
        )

        result = validator.validate(context)

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("unsupported fact" in issue.lower() for issue in result.issues)
        )

    def test_uses_retrieved_metadata_extra_without_crashing(self) -> None:
        validator = DeterministicValidator()
        context = make_context(
            DecisionStatus.accept,
            "The project is accepted.",
            {"status": "ACCEPT", "facts": ["enterprise reporting"]},
        ).model_copy(
            update={
                "retrieved_context": [
                    SearchResult(
                        document=Document(
                            id="doc-1",
                            text="Implementation note.",
                            metadata=DocumentMetadata(
                                source="kb.md",
                                owner="enterprise reporting",
                            ),
                        ),
                        score=0.9,
                        rank=1,
                    )
                ]
            }
        )

        result = validator.validate(context)

        self.assertTrue(result.is_valid)


class TestLLMSelfChecker(unittest.TestCase):
    """Класс «TestLLMSelfChecker» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_runs_llm_only_after_deterministic_validation_passes(self) -> None:
        llm_client = FakeLLMClient([make_llm_payload()])
        tracing_client = RecordingTracingClient()
        checker = LLMSelfChecker(
            llm_client=llm_client,
            tracing_client=tracing_client,
            model_name="test-model",
        )
        deterministic_result = SelfCheckResult(
            is_valid=True,
            issues=[],
            warnings=[],
            checked_fields=["arbitration_result.final_status"],
            technical_info=None,
        )

        result = checker.check(
            make_context(
                DecisionStatus.accept,
                "The project is accepted and remains consistent.",
                {"status": "ACCEPT", "facts": ["Build a customer portal"]},
            ),
            deterministic_result,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.warnings, ["Semantic check passed."])
        self.assertEqual(result.technical_info.llm_invoked, True)
        self.assertEqual(len(llm_client.calls), 1)
        self.assertEqual(tracing_client.trace_calls[0]["name"], "self_check.llm")
        self.assertEqual(
            tracing_client.span_calls[0]["name"],
            "self_check.llm_review",
        )

    def test_skips_llm_when_deterministic_result_has_issues(self) -> None:
        llm_client = FakeLLMClient([make_llm_payload()])
        checker = LLMSelfChecker(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            model_name="test-model",
        )
        deterministic_result = SelfCheckResult(
            is_valid=False,
            issues=["Status mismatch."],
            warnings=[],
            checked_fields=["arbitration_result.final_status"],
            technical_info=None,
        )

        result = checker.check(
            make_context(
                DecisionStatus.accept,
                "The project is accepted.",
                {"status": "REJECT"},
            ),
            deterministic_result,
        )

        self.assertEqual(result.issues, ["Status mismatch."])
        self.assertEqual(len(llm_client.calls), 0)


class TestSelfChecker(unittest.TestCase):
    """Класс «TestSelfChecker» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_orchestrator_skips_llm_on_deterministic_failure(self) -> None:
        llm_client = FakeLLMClient([make_llm_payload()])
        llm_checker = LLMSelfChecker(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            model_name="test-model",
        )
        checker = SelfChecker(llm_self_checker=llm_checker)

        result = checker.check(
            make_context(
                DecisionStatus.accept,
                "The project is accepted.",
                {"status": "REJECT"},
            )
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(len(llm_client.calls), 0)

    def test_run_context_uses_assessment_result_with_legacy_prompt_shape(self) -> None:
        llm_client = FakeLLMClient([make_llm_payload()])
        llm_checker = LLMSelfChecker(
            llm_client=llm_client,
            tracing_client=RecordingTracingClient(),
            model_name="test-model",
        )
        checker = SelfChecker(llm_self_checker=llm_checker)
        context = (
            AIContext.from_brief(make_brief_input())
            .with_extracted_brief(make_extracted_brief())
            .with_completeness_result(make_completeness_result())
            .with_assessment_result(make_assessment_result())
            .with_arbitration_result(make_arbitration_result(DecisionStatus.accept))
            .with_final_response(
                "The project is accepted and remains consistent.",
                {
                    "status": "ACCEPT",
                    "facts": ["Build a customer portal"],
                },
            )
        )

        self.assertIsNotNone(context.assessment_result)
        self.assertFalse(hasattr(context, "risk_analysis_result"))
        self.assertFalse(hasattr(context, "evaluation_result"))

        updated = checker.run_context(context)

        self.assertIsNotNone(updated.self_check_result)
        self.assertTrue(updated.self_check_result.is_valid)
        self.assertEqual(len(llm_client.calls), 1)

        messages = llm_client.calls[0]["messages"]
        prompt_context = json.loads(messages[1]["content"])

        self.assertIn("risk_analysis_result", prompt_context)
        self.assertIn("evaluation_result", prompt_context)
        self.assertNotIn("assessment_result", prompt_context)

        risk_analysis = prompt_context["risk_analysis_result"]
        self.assertTrue(risk_analysis["has_risks"])
        self.assertEqual(
            risk_analysis["summary"],
            "The brief is clear but has moderate scope risk.",
        )
        risk = risk_analysis["risks"][0]
        self.assertEqual(risk["type"], "scope_too_large")
        self.assertEqual(
            risk["description"],
            "Reporting and account access may exceed the first release.",
        )
        self.assertEqual(risk["severity"], "medium")
        self.assertEqual(risk["evidence"], ["reporting and account access"])
        self.assertEqual(
            risk_analysis["technical_info"]["prompt_name"],
            "assessment.md",
        )
        self.assertNotIn("provider_metadata", risk_analysis["technical_info"])

        evaluation_result = prompt_context["evaluation_result"]
        self.assertEqual(
            evaluation_result["summary"],
            "The brief is clear but has moderate scope risk.",
        )
        evaluation = evaluation_result["criterion_evaluations"][0]
        self.assertEqual(evaluation["criterion"], "goal_clarity")
        self.assertEqual(evaluation["status"], "met")
        self.assertEqual(evaluation["evidence"], ["Build a customer portal."])
        self.assertEqual(
            evaluation["explanation"],
            "The goal is explicit and supported by the brief.",
        )
        self.assertEqual(
            evaluation_result["technical_info"]["criteria_count"],
            1,
        )
        self.assertNotIn("provider_metadata", evaluation_result["technical_info"])

    def test_assessment_backed_context_preserves_legacy_sections_without_extra_fields(
        self,
    ) -> None:
        assessment = make_assessment_result()
        assessment = assessment.model_copy(
            update={
                "evidence": [
                    AssessmentEvidence(
                        source="assessment-extra-source",
                        quote="assessment-evidence-marker",
                    )
                ],
                "recommendation": AssessmentRecommendation.high_risk_review,
                "confidence": 0.42,
                "technical_info": assessment.technical_info.model_copy(
                    update={
                        "provider_metadata": {
                            "provider": "provider-metadata-marker",
                        },
                        "risk_types_count": 99,
                    }
                ),
            }
        )
        context = (
            AIContext.from_brief(make_brief_input())
            .with_extracted_brief(make_extracted_brief())
            .with_completeness_result(make_completeness_result())
            .with_assessment_result(assessment)
            .with_arbitration_result(make_arbitration_result(DecisionStatus.accept))
            .with_final_response(
                "The project is accepted and remains consistent.",
                {
                    "status": "ACCEPT",
                    "facts": ["Build a customer portal"],
                },
            )
        )
        self_check_context = SelfChecker._build_self_check_context(context)
        deterministic_result = SelfCheckResult(
            is_valid=True,
            issues=[],
            warnings=[],
            checked_fields=["arbitration_result.final_status"],
            technical_info=None,
        )
        checker = LLMSelfChecker(
            llm_client=FakeLLMClient([]),
            tracing_client=RecordingTracingClient(),
        )

        prompt_context = json.loads(
            checker._build_user_prompt(self_check_context, deterministic_result)
        )

        self.assertIn("risk_analysis_result", prompt_context)
        self.assertIn("evaluation_result", prompt_context)
        self.assertNotIn("assessment_result", prompt_context)

        risk_analysis = prompt_context["risk_analysis_result"]
        self.assertTrue(risk_analysis["has_risks"])
        self.assertEqual(risk_analysis["summary"], assessment.summary)
        self.assertEqual(risk_analysis["risks"][0]["type"], "scope_too_large")
        self.assertEqual(
            risk_analysis["technical_info"]["prompt_name"],
            "assessment.md",
        )
        self.assertNotIn("provider_metadata", risk_analysis["technical_info"])
        self.assertNotIn("risk_types_count", risk_analysis["technical_info"])
        self.assertNotIn("recommendation", risk_analysis)
        self.assertNotIn("confidence", risk_analysis)

        evaluation_result = prompt_context["evaluation_result"]
        self.assertEqual(evaluation_result["summary"], assessment.summary)
        self.assertEqual(
            evaluation_result["criterion_evaluations"][0]["criterion"],
            "goal_clarity",
        )
        self.assertEqual(
            evaluation_result["technical_info"]["criteria_count"],
            1,
        )
        self.assertNotIn("provider_metadata", evaluation_result["technical_info"])
        self.assertNotIn("risk_types_count", evaluation_result["technical_info"])
        self.assertNotIn("recommendation", evaluation_result)
        self.assertNotIn("confidence", evaluation_result)

        serialized_prompt = json.dumps(prompt_context)
        self.assertNotIn("assessment-evidence-marker", serialized_prompt)
        self.assertNotIn("provider-metadata-marker", serialized_prompt)
        self.assertNotIn("high_risk_review", serialized_prompt)

        corpus = DeterministicValidator._build_support_corpus(self_check_context)
        self.assertTrue(
            any("Reporting and account access may exceed" in item for item in corpus)
        )
        self.assertTrue(any("goal_clarity" in item for item in corpus))
        self.assertFalse(any("assessment-evidence-marker" in item for item in corpus))
        self.assertFalse(any("provider-metadata-marker" in item for item in corpus))
        self.assertFalse(any("high_risk_review" in item for item in corpus))


if __name__ == "__main__":
    unittest.main()
