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
from app.pipeline.assessment import RestrictedTopicMatcher
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
    TrafficLightMatch,
    TrafficLightResult,
    TrafficLightStatus,
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


def make_context_without_tasks() -> AIContext:
    """Build context where explicit project work is present outside tasks."""
    brief_input = BriefInputFactory().from_text(
        "Нужно создать телеграм-бота на Python для записи клиентов и отправки уведомлений."
    )
    extracted = make_extracted_brief().model_copy(
        update={
            "project_goal": ExtractedFact(
                status=FactStatus.explicit,
                value="создать телеграм-бота на Python",
                evidence=["создать телеграм-бота на Python"],
                confidence=0.9,
            ),
            "tasks": [],
            "expected_result": ExtractedFact(
                status=FactStatus.explicit,
                value="телеграм-бот для записи клиентов и отправки уведомлений",
                evidence=[
                    "телеграм-бота на Python для записи клиентов и отправки уведомлений"
                ],
                confidence=0.9,
            ),
        }
    )
    return (
        AIContext.from_brief(brief_input)
        .with_extracted_brief(extracted)
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


def make_assessment_payload_with_traffic_light(
    *,
    status: TrafficLightStatus,
    matches: list[TrafficLightMatch],
) -> AssessmentPayload:
    """Build an otherwise ordinary assessment payload with traffic-light data."""
    return AssessmentPayload(
        criterion_evaluations=[],
        risks=[],
        evidence=[],
        has_risks=False,
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        traffic_light=TrafficLightResult(
            status=status,
            direction="Программирование",
            specialization="Питон/питон+",
            matches=matches,
            reason="LLM supplied summary",
        ),
    )


def make_traffic_light_match(
    status: TrafficLightStatus,
    task: str,
    matched_rule: str | None = None,
) -> TrafficLightMatch:
    """Build a single traffic-light match for assessment tests."""
    return TrafficLightMatch(
        task=task,
        matched_rule=matched_rule or f"rule for {task}",
        status=status,
        reason=f"{status.value} match",
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

        self.assertEqual(prepared.criteria_count, 7)
        self.assertEqual(prepared.risk_types_count, 6)
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
                "restricted_topic",
                "missing_materials",
                "scope_too_large",
                "mentor_expertise_required",
                "production_criticality",
            ],
        )
        self.assertEqual(prepared.risk_types_count, 6)
        for risk_type_key in risk_type_keys:
            self.assertIn(risk_type_key, prompt)
        self.assertNotIn("placeholder_risk", prompt)
        self.assertNotIn("Deprecated placeholder risk", prompt)
        self.assertEqual(result.risks[0].type, "scope_too_large")
        self.assertEqual(result.technical_info.risk_types_count, 6)

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
        self.assertEqual(result.technical_info.criteria_count, 7)
        self.assertEqual(result.technical_info.risk_types_count, 6)
        self.assertEqual(result.technical_info.retrieved_context_count, 1)
        self.assertTrue(result.technical_info.retriever_used)
        self.assertEqual(runner.calls[0]["output_model"], AssessmentPayload)
        self.assertEqual(runner.calls[0]["trace_name"], "assessment.brief")
        self.assertEqual(runner.calls[0]["span_name"], "assessment.llm")
        self.assertIn("Build a support bot", runner.calls[0]["prompt"])
        self.assertIn("Evaluation criteria", runner.calls[0]["prompt"])
        self.assertIn("Traffic-light config", runner.calls[0]["prompt"])
        self.assertIn("создание телеграм-бота", runner.calls[0]["prompt"])
        self.assertIn("assessment analyst", runner.calls[0]["system_prompt"])
        self.assertNotIn("Build a support bot", runner.calls[0]["system_prompt"])

    def test_traffic_light_prompt_uses_goal_and_expected_result_when_tasks_empty(
        self,
    ) -> None:
        stage = AssessmentStage(
            llm_runner=FakeLLMRunner(make_assessment_payload()),
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        prepared = stage._preparation.prepare(make_context_without_tasks())
        prompt = stage.build_prompt(prepared)
        system_prompt = stage.build_system_prompt(prepared)

        self.assertIn("создать телеграм-бота на Python", prompt)
        self.assertIn(
            "телеграм-бот для записи клиентов и отправки уведомлений",
            prompt,
        )
        self.assertIsNotNone(system_prompt)
        self.assertIn(
            "extracted_brief.tasks, extracted_brief.project_goal, and extracted_brief.expected_result",
            system_prompt,
        )
        self.assertIn("Do not decompose a goal into hidden subtasks.", system_prompt)

    def test_traffic_light_single_green_match_sets_overall_green(self) -> None:
        result = self._run_assessment_with_traffic_light(
            llm_status=TrafficLightStatus.green,
            match_statuses=[TrafficLightStatus.green],
        )

        self.assertEqual(result.traffic_light.status, TrafficLightStatus.green)

    def test_traffic_light_green_and_yellow_sets_overall_yellow(self) -> None:
        result = self._run_assessment_with_traffic_light(
            llm_status=TrafficLightStatus.green,
            match_statuses=[TrafficLightStatus.green, TrafficLightStatus.yellow],
        )

        self.assertEqual(result.traffic_light.status, TrafficLightStatus.yellow)

    def test_traffic_light_green_and_red_sets_overall_red(self) -> None:
        result = self._run_assessment_with_traffic_light(
            llm_status=TrafficLightStatus.green,
            match_statuses=[TrafficLightStatus.green, TrafficLightStatus.red],
        )

        self.assertEqual(result.traffic_light.status, TrafficLightStatus.red)

    def test_traffic_light_green_and_unknown_sets_overall_unknown(self) -> None:
        result = self._run_assessment_with_traffic_light(
            llm_status=TrafficLightStatus.green,
            match_statuses=[TrafficLightStatus.green, TrafficLightStatus.unknown],
        )

        self.assertEqual(result.traffic_light.status, TrafficLightStatus.unknown)

    def test_traffic_light_red_and_unknown_sets_overall_red(self) -> None:
        result = self._run_assessment_with_traffic_light(
            llm_status=TrafficLightStatus.green,
            match_statuses=[TrafficLightStatus.red, TrafficLightStatus.unknown],
        )

        self.assertEqual(result.traffic_light.status, TrafficLightStatus.red)

    def test_traffic_light_empty_matches_sets_overall_unknown(self) -> None:
        result = self._run_assessment_with_traffic_light(
            llm_status=TrafficLightStatus.green,
            match_statuses=[],
        )

        self.assertEqual(result.traffic_light.status, TrafficLightStatus.unknown)

    def test_traffic_light_llm_status_is_overridden_by_matches(self) -> None:
        result = self._run_assessment_with_traffic_light(
            llm_status=TrafficLightStatus.green,
            match_statuses=[TrafficLightStatus.red],
        )

        self.assertEqual(result.traffic_light.status, TrafficLightStatus.red)
        self.assertEqual(result.traffic_light.matches[0].status, TrafficLightStatus.red)

    def test_traffic_light_existing_rule_uses_config_color(self) -> None:
        result = self._run_assessment_with_traffic_light_matches(
            [
                make_traffic_light_match(
                    TrafficLightStatus.yellow,
                    "customer relationship management",
                    "CRM",
                )
            ]
        )

        self.assertEqual(result.traffic_light.matches[0].status, TrafficLightStatus.green)
        self.assertEqual(result.traffic_light.status, TrafficLightStatus.green)

    def test_traffic_light_existing_rule_sets_config_direction_and_specialization(
        self,
    ) -> None:
        result = self._run_assessment_with_traffic_light_matches(
            [
                make_traffic_light_match(
                    TrafficLightStatus.yellow,
                    "customer relationship management",
                    "CRM",
                )
            ]
        )

        self.assertEqual(result.traffic_light.direction, "programming")
        self.assertEqual(result.traffic_light.specialization, "python")

    def test_traffic_light_unknown_rule_becomes_unknown(self) -> None:
        result = self._run_assessment_with_traffic_light_matches(
            [
                make_traffic_light_match(
                    TrafficLightStatus.red,
                    "unmatched task",
                    "no such traffic-light rule",
                )
            ]
        )

        self.assertEqual(result.traffic_light.matches[0].status, TrafficLightStatus.unknown)
        self.assertEqual(result.traffic_light.status, TrafficLightStatus.unknown)
        self.assertIsNone(result.traffic_light.direction)
        self.assertIsNone(result.traffic_light.specialization)

    def test_traffic_light_duplicate_rule_text_becomes_unknown(self) -> None:
        result = self._run_assessment_with_traffic_light_matches(
            [
                make_traffic_light_match(
                    TrafficLightStatus.red,
                    "private network",
                    "VPN",
                )
            ]
        )

        self.assertEqual(result.traffic_light.matches[0].status, TrafficLightStatus.unknown)
        self.assertEqual(result.traffic_light.status, TrafficLightStatus.unknown)
        self.assertIsNone(result.traffic_light.direction)
        self.assertIsNone(result.traffic_light.specialization)

    def test_traffic_light_overall_status_uses_normalized_match_colors(self) -> None:
        result = self._run_assessment_with_traffic_light_matches(
            [
                make_traffic_light_match(
                    TrafficLightStatus.yellow,
                    "customer relationship management",
                    "CRM",
                ),
                make_traffic_light_match(
                    TrafficLightStatus.green,
                    "unity game",
                    "разработка игр на unity",
                ),
            ]
        )

        self.assertEqual(
            [match.status for match in result.traffic_light.matches],
            [TrafficLightStatus.green, TrafficLightStatus.red],
        )
        self.assertEqual(result.traffic_light.status, TrafficLightStatus.red)

    def _run_assessment_with_traffic_light(
        self,
        *,
        llm_status: TrafficLightStatus,
        match_statuses: list[TrafficLightStatus],
    ) -> AssessmentResult:
        rules_by_status = {
            TrafficLightStatus.green: "создание телеграм-бота",
            TrafficLightStatus.yellow: "простая текстовая игра (например, крестики-нолики, змейка или угадай слово)",
            TrafficLightStatus.red: "разработка игр на unity",
            TrafficLightStatus.unknown: "no matching traffic-light rule",
        }
        matches = [
            make_traffic_light_match(
                status,
                f"task {index}",
                rules_by_status[status],
            )
            for index, status in enumerate(match_statuses, start=1)
        ]
        stage = AssessmentStage(
            llm_runner=FakeLLMRunner(
                make_assessment_payload_with_traffic_light(
                    status=llm_status,
                    matches=matches,
                )
            ),
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        return stage.run(stage._preparation.prepare(make_context()))

    def _run_assessment_with_traffic_light_matches(
        self,
        matches: list[TrafficLightMatch],
    ) -> AssessmentResult:
        stage = AssessmentStage(
            llm_runner=FakeLLMRunner(
                make_assessment_payload_with_traffic_light(
                    status=TrafficLightStatus.green,
                    matches=matches,
                )
            ),
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        return stage.run(stage._preparation.prepare(make_context()))

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


def make_restricted_context(
    text: str,
    *,
    goal: str = "Build a support bot",
) -> AIContext:
    """Собирает контекст с заданным текстом брифа и целью проекта. Два источника разведены, чтобы проверить, что матчер смотрит и в текст, и в извлеченные факты."""
    brief = make_extracted_brief().model_copy(
        update={
            "project_goal": ExtractedFact(
                status=FactStatus.explicit,
                value=goal,
                evidence=[goal],
                confidence=0.9,
            )
        }
    )
    return (
        AIContext.from_brief(BriefInputFactory().from_text(text))
        .with_extracted_brief(brief)
        .with_completeness_result(make_completeness_result())
    )


def make_config_without_restricted_topics() -> CriteriaConfig:
    """Выполняет шаг «make config without restricted topics». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    config = load_test_criteria_config()
    evaluation = config.evaluation.model_copy(update={"restricted_topics": None})
    return config.model_copy(update={"evaluation": evaluation})


class TestRestrictedTopics(unittest.TestCase):
    """Класс «TestRestrictedTopics» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_matches_restricted_topic_in_brief_text(self) -> None:
        matcher = RestrictedTopicMatcher(load_test_criteria_config())

        hit = matcher.match(
            make_restricted_context("Нужна криптобиржа с обменом токенов.")
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit.topic.key, "crypto_assets")
        self.assertEqual(hit.keyword, "криптобирж")

    def test_matches_restricted_topic_in_extracted_fact_only(self) -> None:
        matcher = RestrictedTopicMatcher(load_test_criteria_config())

        hit = matcher.match(
            make_restricted_context(
                "Здравствуйте, хотим обсудить новый проект.",
                goal="Запустить онлайн-казино",
            )
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit.topic.key, "gambling")
        self.assertEqual(hit.source_title, "цель проекта")

    def test_ordinary_brief_does_not_match(self) -> None:
        matcher = RestrictedTopicMatcher(load_test_criteria_config())

        hit = matcher.match(
            make_restricted_context(
                "Нужен телеграм-бот для записи клиентов в барбершоп.",
                goal="Автоматизировать запись клиентов",
            )
        )

        self.assertIsNone(hit)

    def test_regulated_industry_is_not_a_restricted_topic(self) -> None:
        matcher = RestrictedTopicMatcher(load_test_criteria_config())

        hit = matcher.match(
            make_restricted_context(
                "Нужен дашборд посещаемости для медицинской клиники.",
                goal="Собрать дашборд посещаемости",
            )
        )

        self.assertIsNone(hit)

    def test_long_source_is_trimmed_to_a_window_around_the_match(self) -> None:
        matcher = RestrictedTopicMatcher(load_test_criteria_config())
        filler = "рассказываем о себе и о своей компании много слов подряд "

        hit = matcher.match(
            make_restricted_context(
                f"{filler * 4} нужна платформа для продажи NFT-коллекций. {filler * 4}",
                goal="Запустить площадку",
            )
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit.source_title, "текст брифа")
        self.assertIn("nft", hit.fragment)
        self.assertTrue(hit.fragment.startswith("…"))
        self.assertTrue(hit.fragment.endswith("…"))
        self.assertLess(len(hit.fragment), 200)

    def test_matcher_is_silent_when_config_has_no_restricted_topics(self) -> None:
        matcher = RestrictedTopicMatcher(make_config_without_restricted_topics())

        hit = matcher.match(
            make_restricted_context("Нужна криптобиржа с обменом токенов.")
        )

        self.assertIsNone(hit)

    def test_restricted_topic_short_circuits_llm_and_retriever(self) -> None:
        runner = FakeLLMRunner(AssertionError("LLM must not be called"))
        retriever = FakeRetriever([make_search_result()])
        stage = AssessmentStage(
            llm_runner=runner,
            retriever=retriever,
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        updated = stage.run_context(
            make_restricted_context("Нужна криптобиржа с обменом токенов.")
        )

        self.assertEqual(runner.calls, [])
        self.assertEqual(retriever.calls, [])
        self.assertIsNotNone(updated.assessment_result)

    def test_short_circuit_result_carries_the_ground_for_the_manager(self) -> None:
        stage = AssessmentStage(
            llm_runner=FakeLLMRunner(AssertionError("LLM must not be called")),
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        updated = stage.run_context(
            make_restricted_context("Нужна криптобиржа с обменом токенов.")
        )
        result = updated.assessment_result

        self.assertEqual(result.recommendation, AssessmentRecommendation.high_risk_review)
        self.assertTrue(result.has_risks)
        self.assertEqual(result.risks[0].type, "restricted_topic")
        self.assertEqual(result.risks[0].severity, RiskSeverity.critical)
        self.assertEqual(
            result.criterion_evaluations[0].criterion, "topic_eligibility"
        )
        self.assertEqual(
            result.criterion_evaluations[0].status,
            CriterionEvaluationStatus.not_met,
        )
        # Менеджер видит только описания рисков, поэтому цитата обязана быть в них,
        # а не только в отдельном поле evidence.
        self.assertIn("криптобирж", result.risks[0].description)
        self.assertEqual(result.technical_info.attempts, 0)

    def test_ordinary_brief_still_reaches_the_llm(self) -> None:
        runner = FakeLLMRunner(make_assessment_payload())
        stage = AssessmentStage(
            llm_runner=runner,
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        stage.run_context(
            make_restricted_context(
                "Нужен телеграм-бот для записи клиентов в барбершоп.",
                goal="Автоматизировать запись клиентов",
            )
        )

        self.assertEqual(len(runner.calls), 1)

    def test_restricted_topics_reach_the_assessment_prompt(self) -> None:
        stage = AssessmentStage(
            llm_runner=FakeLLMRunner(make_assessment_payload()),
            tracing_client=NoOpTracingClient(),
            criteria_config=load_test_criteria_config(),
        )

        prepared = stage._preparation.prepare(make_context())
        prompt = stage.build_prompt(prepared)

        self.assertEqual(len(prepared.restricted_topics), 3)
        for topic in prepared.restricted_topics:
            self.assertIn(topic.key, prompt)


if __name__ == "__main__":
    unittest.main()
