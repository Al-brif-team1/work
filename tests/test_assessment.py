"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.config import CriteriaConfig, CriteriaLoader
from app.input import BriefInputFactory
from app.llm.runner import (
    LLMRunResult,
    LLMRunnerProviderError,
    LLMRunnerStructuredOutputError,
    LLMTokenUsage,
)
from app.pipeline import (
    AssessmentConfigError,
    AssessmentError,
    AssessmentPreparation,
    AssessmentStage,
)
from app.schemas import (
    AIContext,
    AssessmentEvidence,
    AssessmentPayload,
    AssessmentRecommendation,
    AssessmentResult,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    Document,
    DocumentMetadata,
    ExtractedBrief,
    ExtractedFact,
    FactStatus,
    Risk,
    RiskSeverity,
    SearchResult,
)
from app.tracing.tracing import NoOpTracingClient


class FakeRetriever:
    """Класс «FakeRetriever» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "metadata_filters": dict(metadata_filters or {}),
            }
        )
        return list(self.results)


class FakeLLMRunner:
    """Класс «FakeLLMRunner» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, payload: AssessmentPayload | Exception) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> LLMRunResult[AssessmentPayload]:
        self.calls.append(kwargs)
        if isinstance(self.payload, Exception):
            raise self.payload

        return LLMRunResult(
            payload=self.payload,
            raw_response={"ok": True},
            attempts=1,
            latency_seconds=0.01,
            token_usage=LLMTokenUsage(total_tokens_estimate=1),
            provider_metadata={"provider": "fake"},
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
            value="Build a support bot",
            evidence=["Build a support bot"],
            confidence=0.9,
        ),
        tasks=[
            ExtractedFact(
                status=FactStatus.explicit,
                value="Integrate support channels",
                evidence=["Integrate support channels"],
                confidence=0.8,
            )
        ],
        project_type=ExtractedFact(
            status=FactStatus.explicit,
            value="web_app",
            evidence=["web app"],
            confidence=0.9,
        ),
        project_direction=ExtractedFact(
            status=FactStatus.explicit,
            value="support automation",
            evidence=["support automation"],
            confidence=0.9,
        ),
        technologies=[
            ExtractedFact(
                status=FactStatus.explicit,
                value="Python",
                evidence=["Python"],
                confidence=0.9,
            )
        ],
        expected_result=ExtractedFact(
            status=FactStatus.explicit,
            value="Working bot",
            evidence=["Working bot"],
            confidence=0.9,
        ),
    )


def make_completeness_result() -> CompletenessResult:
    """Выполняет шаг «make completeness result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return CompletenessResult(
        is_complete=False,
        missing_information=[
            CompletenessItem(
                field_key="integrations",
                field_path="integrations",
                title="Integrations",
                status=CompletenessStatus.missing,
                value=None,
                reason="Integration details are unclear",
            )
        ],
        present_information=[],
        clarification_information=[],
        warnings=[],
    )


def make_context() -> AIContext:
    """Выполняет шаг «make context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    brief_input = BriefInputFactory().from_text(
        "Build a support bot with Python and external channels."
    )
    return (
        AIContext.from_brief(brief_input)
        .with_extracted_brief(make_extracted_brief())
        .with_completeness_result(make_completeness_result())
    )


def make_search_result() -> SearchResult:
    """Выполняет шаг «make search result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return SearchResult(
        document=Document(
            id="kb-1",
            text="Support channel integration guidance.",
            metadata=DocumentMetadata(source="kb.md", category="assessment"),
        ),
        score=0.92,
        rank=1,
    )


def load_test_criteria_config() -> CriteriaConfig:
    """Выполняет шаг «load test criteria config». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return CriteriaLoader.load()


def make_criteria_config_without_placeholder_risk() -> CriteriaConfig:
    """Выполняет шаг «make criteria config without placeholder risk». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    config = load_test_criteria_config()
    risk_analysis = config.evaluation.risk_analysis
    active_risk_types = [
        risk_type
        for risk_type in risk_analysis.risk_types
        if risk_type.key != "placeholder_risk"
    ]
    return config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={
                    "risk_analysis": risk_analysis.model_copy(
                        update={"risk_types": active_risk_types}
                    )
                }
            )
        }
    )


def make_assessment_payload() -> AssessmentPayload:
    """Выполняет шаг «make assessment payload». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return AssessmentPayload(
        criterion_evaluations=[
            CriterionEvaluation(
                criterion=" placeholder_criterion ",
                criterion_title=" Placeholder criterion ",
                status=CriterionEvaluationStatus.risk_detected,
                evidence=[" External channel requirements are vague ", "  "],
                explanation=" Needs clearer integrations ",
                confidence=0.81,
                notes=" Check integration scope ",
            )
        ],
        risks=[
            Risk(
                type=" placeholder_risk ",
                description=" External integrations are underspecified ",
                severity=RiskSeverity.medium,
                evidence=[" external channels ", ""],
                confidence=0.79,
                notes=" Clarify channels ",
            )
        ],
        evidence=[
            AssessmentEvidence(
                source=" brief ",
                quote=" external channels ",
                related_criteria=[" placeholder_criterion ", ""],
                related_risks=[" placeholder_risk ", ""],
                confidence=0.9,
            )
        ],
        has_risks=True,
        recommendation=AssessmentRecommendation.high_risk_review,
        summary=" Assessment identified one risk. ",
        confidence=0.82,
    )


class TestAssessmentModels(unittest.TestCase):
    """Класс «TestAssessmentModels» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_assessment_result_contains_criteria_risks_evidence_and_confidence(
        self,
    ) -> None:
        result = AssessmentResult(
            criterion_evaluations=[
                CriterionEvaluation(
                    criterion="placeholder_criterion",
                    criterion_title="Placeholder criterion",
                    status=CriterionEvaluationStatus.risk_detected,
                    evidence=["External channel requirements are vague"],
                    confidence=0.81,
                )
            ],
            risks=[
                Risk(
                    type="placeholder_risk",
                    description="External integrations are underspecified",
                    severity=RiskSeverity.medium,
                    evidence=["external channels"],
                    confidence=0.79,
                )
            ],
            evidence=[
                AssessmentEvidence(
                    source="brief",
                    quote="external channels",
                    related_criteria=["placeholder_criterion"],
                    related_risks=["placeholder_risk"],
                    confidence=0.9,
                )
            ],
            has_risks=True,
            recommendation=AssessmentRecommendation.high_risk_review,
            summary="Assessment identified one risk.",
            confidence=0.82,
        )

        self.assertEqual(result.criterion_evaluations[0].criterion, "placeholder_criterion")
        self.assertEqual(result.risks[0].type, "placeholder_risk")
        self.assertEqual(result.evidence[0].source, "brief")
        self.assertEqual(
            result.recommendation,
            AssessmentRecommendation.high_risk_review,
        )
        self.assertEqual(result.confidence, 0.82)

    def test_assessment_result_rejects_inconsistent_risk_flag(self) -> None:
        with self.assertRaises(ValidationError):
            AssessmentResult(
                criterion_evaluations=[],
                risks=[],
                evidence=[],
                has_risks=True,
                recommendation=AssessmentRecommendation.high_risk_review,
            )

    def test_assessment_payload_requires_non_decision_recommendation(self) -> None:
        payload = AssessmentPayload(
            criterion_evaluations=[],
            risks=[],
            evidence=[],
            has_risks=False,
            recommendation=AssessmentRecommendation.ready_for_arbitration,
        )

        self.assertEqual(
            payload.recommendation,
            AssessmentRecommendation.ready_for_arbitration,
        )

        with self.assertRaises(ValidationError):
            AssessmentPayload(
                criterion_evaluations=[],
                risks=[],
                evidence=[],
                has_risks=False,
                recommendation="accept",
            )

    def test_context_accepts_unified_assessment_result_by_copy(self) -> None:
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a bot"))
        result = AssessmentResult(
            criterion_evaluations=[],
            risks=[],
            evidence=[],
            has_risks=False,
            recommendation=AssessmentRecommendation.ready_for_arbitration,
        )

        updated = context.with_assessment_result(result)

        self.assertIsNone(context.assessment_result)
        self.assertIs(updated.assessment_result, result)


class TestAssessmentPreparation(unittest.TestCase):
    """Класс «TestAssessmentPreparation» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_prepare_uses_retriever_and_criteria_configuration(self) -> None:
        retriever = FakeRetriever([make_search_result()])
        preparation = AssessmentPreparation(
            retriever=retriever,
            criteria_config=load_test_criteria_config(),
        )

        prepared = preparation.prepare(
            make_context(),
            top_k=3,
            metadata_filters={"category": "assessment"},
        )

        self.assertEqual(prepared.criteria_count, 6)
        self.assertEqual(prepared.risk_types_count, 5)
        self.assertEqual(prepared.retrieved_context[0].document.id, "kb-1")
        self.assertEqual(prepared.context.retrieved_context[0].document.id, "kb-1")
        self.assertIn("Build a support bot", retriever.calls[0]["query"])
        self.assertIn("integrations", retriever.calls[0]["query"])
        self.assertEqual(retriever.calls[0]["top_k"], 3)
        self.assertEqual(
            retriever.calls[0]["metadata_filters"],
            {"category": "assessment"},
        )

    def test_prepare_reuses_existing_context_when_no_retriever_is_configured(
        self,
    ) -> None:
        existing_result = make_search_result()
        context = make_context().with_retrieved_context([existing_result])
        preparation = AssessmentPreparation(criteria_config=load_test_criteria_config())

        prepared = preparation.prepare(context)

        self.assertEqual(prepared.retrieved_context, [existing_result])
        self.assertEqual(prepared.context.retrieved_context, [existing_result])

    def test_prepare_requires_extraction_and_completeness_outputs(self) -> None:
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a bot"))
        preparation = AssessmentPreparation(criteria_config=load_test_criteria_config())

        with self.assertRaisesRegex(AssessmentError, "extracted_brief"):
            preparation.prepare(context)

    def test_config_requires_risk_analysis_section(self) -> None:
        config = load_test_criteria_config()
        invalid_config = config.model_copy(
            update={
                "evaluation": config.evaluation.model_copy(
                    update={"risk_analysis": None}
                )
            }
        )

        with self.assertRaisesRegex(AssessmentConfigError, "risk_analysis"):
            AssessmentPreparation(criteria_config=invalid_config)


class TestAssessmentStage(unittest.TestCase):
    """Класс «TestAssessmentStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_assessment_works_without_deprecated_placeholder_risk(self) -> None:
        config = make_criteria_config_without_placeholder_risk()
        payload = AssessmentPayload(
            criterion_evaluations=[],
            risks=[
                Risk(
                    type="scope_too_large",
                    description="The requested scope is broad.",
                    severity=RiskSeverity.high,
                    evidence=["external channels"],
                )
            ],
            evidence=[],
            has_risks=True,
            recommendation=AssessmentRecommendation.high_risk_review,
            summary="Scope needs review.",
        )
        stage = AssessmentStage(
            llm_runner=FakeLLMRunner(payload),
            tracing_client=NoOpTracingClient(),
            criteria_config=config,
        )
        prepared = stage._preparation.prepare(make_context())

        risk_type_keys = [risk_type.key for risk_type in prepared.risk_types]
        prompt = stage.build_prompt(prepared)
        result = stage.run(prepared)

        self.assertEqual(
            risk_type_keys,
            [
                "out_of_scope_request",
                "missing_materials",
                "scope_too_large",
                "mentor_expertise_required",
                "production_criticality",
            ],
        )
        self.assertEqual(prepared.risk_types_count, 5)
        for risk_type_key in risk_type_keys:
            self.assertIn(risk_type_key, prompt)
        self.assertNotIn("placeholder_risk", prompt)
        self.assertNotIn("Deprecated placeholder risk", prompt)
        self.assertEqual(result.risks[0].type, "scope_too_large")
        self.assertEqual(result.technical_info.risk_types_count, 5)

    def test_successful_assessment_uses_prompt_manager_and_llm_runner(self) -> None:
        runner = FakeLLMRunner(make_assessment_payload())
        retriever = FakeRetriever([make_search_result()])
        stage = AssessmentStage(
            llm_runner=runner,
            tracing_client=NoOpTracingClient(),
            retriever=retriever,
            criteria_config=load_test_criteria_config(),
        )

        result = stage.run(
            stage._preparation.prepare(
                make_context(),
                top_k=2,
                metadata_filters={"category": "assessment"},
            )
        )

        self.assertEqual(result.risks[0].type, "placeholder_risk")
        self.assertEqual(result.criterion_evaluations[0].criterion, "placeholder_criterion")
        self.assertEqual(result.evidence[0].quote, "external channels")
        self.assertEqual(result.summary, "Assessment identified one risk.")
        self.assertEqual(result.technical_info.criteria_count, 6)
        self.assertEqual(result.technical_info.risk_types_count, 5)
        self.assertEqual(result.technical_info.retrieved_context_count, 1)
        self.assertTrue(result.technical_info.retriever_used)
        self.assertEqual(runner.calls[0]["output_model"], AssessmentPayload)
        self.assertEqual(runner.calls[0]["trace_name"], "assessment.brief")
        self.assertEqual(runner.calls[0]["span_name"], "assessment.llm")
        self.assertIn("Build a support bot", runner.calls[0]["prompt"])
        self.assertIn("Evaluation criteria", runner.calls[0]["prompt"])
        self.assertIn("assessment analyst", runner.calls[0]["system_prompt"])
        self.assertNotIn("Build a support bot", runner.calls[0]["system_prompt"])

    def test_assess_updates_ai_context_by_copy(self) -> None:
        runner = FakeLLMRunner(make_assessment_payload())
        stage = AssessmentStage(
            llm_runner=runner,
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )
        context = make_context()

        updated = stage.assess(context)

        self.assertIsNone(context.assessment_result)
        self.assertIsNotNone(updated.assessment_result)
        self.assertEqual(
            updated.assessment_result.recommendation,
            AssessmentRecommendation.high_risk_review,
        )

    def test_run_context_uses_assessment_pipeline_contract(self) -> None:
        runner = FakeLLMRunner(make_assessment_payload())
        stage = AssessmentStage(
            llm_runner=runner,
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        updated = stage.run_context(make_context())

        self.assertIsNotNone(updated.assessment_result)

    def test_structured_output_error_uses_stage_error_contract(self) -> None:
        stage = AssessmentStage(
            llm_runner=FakeLLMRunner(
                LLMRunnerStructuredOutputError("invalid structured output")
            ),
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        with self.assertRaisesRegex(AssessmentError, "Unable to assess brief"):
            stage.assess(make_context())

    def test_provider_error_uses_stage_error_contract(self) -> None:
        stage = AssessmentStage(
            llm_runner=FakeLLMRunner(LLMRunnerProviderError("provider failed")),
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        with self.assertRaisesRegex(AssessmentError, "Unable to assess brief"):
            stage.assess(make_context())


if __name__ == "__main__":
    unittest.main()
