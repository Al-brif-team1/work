"""Tests for Demo UI DTO adapter."""

from __future__ import annotations

import json
import unittest

from demo_ui.dto import build_demo_response

from app.input import BriefInputFactory
from app.pipeline import BriefAnalysisPipeline, EmptyBriefRejectionStage
from app.schemas import (
    AIContext,
    ArbitrationResult,
    ArbitrationRuleHit,
    AssessmentEvidence,
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
    ExtractionResult,
    ExtractorTechnicalInfo,
    FactStatus,
    MVPPlan,
    MVPPlanningResult,
    MVPPlanningTechnicalInfo,
    QuestionGenerationResult,
    QuestionGenerationTechnicalInfo,
    Risk,
    RiskSeverity,
    TrafficLightMatch,
    TrafficLightResult,
    TrafficLightStatus,
)


def make_extracted_brief() -> ExtractedBrief:
    return ExtractedBrief(
        project_goal=ExtractedFact(
            status=FactStatus.explicit,
            value="Сделать портал поддержки",
            evidence=["Нужен портал поддержки"],
            confidence=0.9,
            notes="цель указана явно",
        ),
        tasks=[
            ExtractedFact(
                status=FactStatus.explicit,
                value="Форма обращения",
                evidence=["форма обращения"],
                confidence=0.9,
            )
        ],
        project_type=ExtractedFact(
            status=FactStatus.explicit,
            value="web_app",
            evidence=["портал"],
            confidence=0.8,
        ),
        project_direction=ExtractedFact(
            status=FactStatus.explicit,
            value="development",
            evidence=["портал поддержки"],
            confidence=0.8,
        ),
        technologies=[],
        stack=[],
        materials=[
            ExtractedFact(
                status=FactStatus.explicit,
                value="Тексты продукта",
                evidence=["есть тексты продукта"],
            )
        ],
        expected_result=ExtractedFact(
            status=FactStatus.explicit,
            value="Рабочая первая версия",
            evidence=["рабочая первая версия"],
            confidence=0.9,
        ),
        constraints=[],
        deadlines=[],
        existing_resources=[],
        integrations=[],
        other_facts=[],
    )


def make_base_context(
    *,
    status: DecisionStatus = DecisionStatus.accept,
    risks: list[Risk] | None = None,
    traffic_light: TrafficLightResult | None = None,
    matched_rule: str | None = "accept_ready",
    questions: QuestionGenerationResult | None = None,
    mvp_result: MVPPlanningResult | None = None,
) -> AIContext:
    risks = risks or []
    brief = BriefInputFactory().from_text(
        "Нужен портал поддержки с формой обращения и рабочей первой версией."
    )
    extraction = ExtractionResult(
        extracted_brief=make_extracted_brief(),
        technical_info=ExtractorTechnicalInfo(
            attempts=1,
            prompt_name="extractor.md",
            trace_enabled=False,
            trace_name="extractor.brief",
            model_name="test-model",
            raw_response={"secret": "not included"},
        ),
    )
    completeness = CompletenessResult(
        is_complete=True,
        missing_information=[],
        critical_missing_information=[],
        optional_missing_information=[],
        present_information=[
            CompletenessItem(
                field_key="project_goal",
                field_path="project_goal",
                title="Project goal",
                status=CompletenessStatus.present,
                value="Сделать портал поддержки",
            )
        ],
        clarification_information=[],
        warnings=[],
    )
    assessment = AssessmentResult(
        criterion_evaluations=[
            CriterionEvaluation(
                criterion="goal_clarity",
                criterion_title="Goal clarity",
                status=CriterionEvaluationStatus.met,
                evidence=["Нужен портал поддержки"],
                explanation="Цель описана явно.",
                confidence=0.9,
            )
        ],
        risks=risks,
        evidence=[
            AssessmentEvidence(
                source="brief",
                quote="Нужен портал поддержки",
                related_criteria=["goal_clarity"],
                confidence=0.9,
            )
        ],
        has_risks=bool(risks),
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        summary="Портал поддержки.",
        confidence=0.85,
        traffic_light=traffic_light or TrafficLightResult(status=TrafficLightStatus.green),
        technical_info=AssessmentTechnicalInfo(
            attempts=1,
            prompt_name="assessment.md",
            trace_enabled=False,
            trace_name="assessment.brief",
            model_name="test-model",
            criteria_count=7,
            risk_types_count=6,
            raw_response={"debug": "not included"},
            provider_metadata={"model": "test-model", "provider": "test-provider"},
        ),
    )
    metadata = {
        "configuration_version": "1",
        "signals": {
            "risk.total_count": len(risks),
            "traffic_light.status": assessment.traffic_light.status.value,
        },
    }
    if matched_rule is not None:
        metadata["matched_rule"] = matched_rule
    arbitration = ArbitrationResult(
        final_status=status,
        reasons=["Решение принято правилом."],
        evidence=["Project goal"],
        triggered_rules=[
            ArbitrationRuleHit(
                rule_key=matched_rule or "default_status",
                title="Test rule",
                status=status,
                conditions=["risk.total_count eq 0"],
                evidence=["Project goal"],
                explanation="Тестовое правило.",
                confidence=0.9,
                metadata={"signals": {"risk.total_count": len(risks)}},
            )
        ]
        if matched_rule is not None
        else [],
        confidence=0.9,
        metadata=metadata,
    )
    context = (
        AIContext.from_brief(brief)
        .with_extraction_result(extraction)
        .with_completeness_result(completeness)
        .with_assessment_result(assessment)
        .with_arbitration_result(arbitration)
    )
    if questions is not None:
        context = context.with_clarification_result(questions)
    if mvp_result is not None:
        context = context.with_mvp_planning_result(mvp_result)
    return context.with_final_response(
        "Здравствуйте! Проект можно брать.",
        {
            "summary": "Проект можно принять в работу. Цель: Сделать портал поддержки",
        },
    )


def make_questions() -> QuestionGenerationResult:
    return QuestionGenerationResult(
        questions=[
            ClarificationQuestion(
                question="Какие материалы уже готовы?",
                related_field="materials",
                reason="Нужно уточнить материалы.",
                priority=2,
            )
        ],
        summary="Generated 1 clarification questions from templates.",
        technical_info=QuestionGenerationTechnicalInfo(
            llm_invoked=False,
            attempts=0,
            prompt_name=None,
            trace_enabled=False,
            trace_name="question_generator.template",
            question_count=1,
        ),
    )


class TestDemoDTO(unittest.TestCase):
    def test_accept_context_serializes_core_sections(self) -> None:
        context = make_base_context(
            mvp_result=MVPPlanningResult(
                plan=None,
                technical_info=MVPPlanningTechnicalInfo(
                    llm_invoked=False,
                    attempts=0,
                    prompt_name="mvp_planner.md",
                    trace_enabled=False,
                    trace_name="mvp_planner.brief",
                    skipped_reason="MVP planner runs only when arbitration status is SIMPLIFY",
                ),
            )
        )

        dto = build_demo_response(context)

        self.assertTrue(dto["ok"])
        self.assertEqual(dto["decision"]["final_status"], "ACCEPT")
        self.assertEqual(dto["decision"]["matched_rule"], "accept_ready")
        self.assertTrue(dto["extraction"]["available"])
        self.assertFalse(dto["extraction"]["synthetic"])
        self.assertEqual(
            dto["extraction"]["extracted_brief"]["project_goal"]["evidence"],
            ["Нужен портал поддержки"],
        )
        self.assertTrue(dto["completeness"]["is_complete"])
        self.assertEqual(dto["traffic_light"]["status"], "green")
        self.assertEqual(dto["risks"]["count"], 0)
        self.assertIsNone(dto["risks"]["max_severity"])
        self.assertIsNone(dto["mvp"]["plan"])
        json.dumps(dto, ensure_ascii=False)

    def test_simplify_context_includes_risks_arbiter_and_mvp_plan(self) -> None:
        context = make_base_context(
            status=DecisionStatus.simplify,
            matched_rule="simplify_scope_too_large",
            risks=[
                Risk(
                    type="scope_too_large",
                    description="Объём первой версии слишком большой.",
                    severity=RiskSeverity.high,
                    evidence=["форма обращения, кабинет и аналитика"],
                    confidence=0.8,
                    notes="scope",
                ),
                Risk(
                    type="missing_materials",
                    description="Материалы не полностью готовы.",
                    severity=RiskSeverity.medium,
                ),
            ],
            mvp_result=MVPPlanningResult(
                plan=MVPPlan(
                    core_goal="Проверить сценарий обращения",
                    keep=["Форма обращения"],
                    simplify=["Кабинет сделать базовым"],
                    remove=["Расширенная аналитика"],
                    mvp_scope=["Форма", "Экран подтверждения"],
                    rationale=["Так уменьшается объём первой версии"],
                ),
                technical_info=MVPPlanningTechnicalInfo(
                    llm_invoked=True,
                    attempts=1,
                    prompt_name="mvp_planner.md",
                    trace_enabled=False,
                    trace_name="mvp_planner.brief",
                    model_name="test-model",
                ),
            ),
        )

        dto = build_demo_response(context)

        self.assertEqual(dto["decision"]["final_status"], "SIMPLIFY")
        self.assertEqual(dto["risks"]["count"], 2)
        self.assertEqual(dto["risks"]["max_severity"], "high")
        self.assertEqual(
            dto["arbiter"]["metadata"]["matched_rule"],
            "simplify_scope_too_large",
        )
        self.assertEqual(dto["mvp"]["plan"]["keep"], ["Форма обращения"])
        self.assertEqual(dto["mvp"]["plan"]["simplify"], ["Кабинет сделать базовым"])
        self.assertEqual(dto["mvp"]["plan"]["remove"], ["Расширенная аналитика"])
        self.assertEqual(dto["mvp"]["plan"]["mvp_scope"], ["Форма", "Экран подтверждения"])
        self.assertEqual(
            dto["mvp"]["plan"]["rationale"],
            ["Так уменьшается объём первой версии"],
        )

    def test_questions_are_serialized_without_generation(self) -> None:
        context = make_base_context(
            status=DecisionStatus.accept_with_clarifications,
            matched_rule="accept_with_missing_optional_information",
            questions=make_questions(),
        )

        dto = build_demo_response(context)

        self.assertTrue(dto["questions"]["available"])
        self.assertEqual(dto["questions"]["items"][0]["question"], "Какие материалы уже готовы?")
        self.assertEqual(dto["questions"]["items"][0]["related_field"], "materials")
        self.assertEqual(dto["questions"]["items"][0]["reason"], "Нужно уточнить материалы.")
        self.assertEqual(dto["questions"]["items"][0]["priority"], 2)
        self.assertFalse(dto["questions"]["technical_info"]["llm_invoked"])

    def test_early_reject_marks_synthetic_short_circuit_sections(self) -> None:
        pipeline = BriefAnalysisPipeline(stages=[EmptyBriefRejectionStage()])
        context = pipeline.run_context(BriefInputFactory().from_text(""))

        dto = build_demo_response(context)

        self.assertTrue(dto["decision"]["short_circuit"])
        self.assertEqual(dto["decision"]["short_circuit_stage"], "EmptyBriefRejectionStage")
        self.assertTrue(dto["extraction"]["available"])
        self.assertTrue(dto["extraction"]["synthetic"])
        self.assertTrue(dto["completeness"]["synthetic"])
        self.assertTrue(dto["assessment"]["synthetic"])
        self.assertTrue(dto["traffic_light"]["synthetic"])
        self.assertTrue(dto["risks"]["synthetic"])
        self.assertEqual(dto["decision"]["final_status"], "REJECT")

    def test_traffic_light_source_quote_is_preserved(self) -> None:
        source_quote = "форма обращения и рабочая первая версия"
        traffic_light = TrafficLightResult(
            status=TrafficLightStatus.yellow,
            direction="programming",
            specialization="web_frontend",
            matches=[
                TrafficLightMatch(
                    task="Форма обращения",
                    matched_rule="простые веб-формы с ограниченной логикой",
                    status=TrafficLightStatus.yellow,
                    source_quote=source_quote,
                    reason="Нужно уточнить объём логики.",
                )
            ],
        )
        context = make_base_context(
            status=DecisionStatus.accept_with_clarifications,
            matched_rule="accept_with_traffic_light_yellow",
            traffic_light=traffic_light,
        )

        dto = build_demo_response(context)

        self.assertEqual(dto["traffic_light"]["matches"][0]["source_quote"], source_quote)

    def test_absent_matched_rule_is_null(self) -> None:
        context = make_base_context(
            status=DecisionStatus.mentor_review,
            matched_rule=None,
        )

        dto = build_demo_response(context)

        self.assertIsNone(dto["decision"]["matched_rule"])
        self.assertNotIn("matched_rule", dto["arbiter"]["metadata"])

    def test_raw_response_and_restoration_map_are_not_exposed(self) -> None:
        context = make_base_context().with_stage_metadata(
            "SecurityGateStage",
            status="warning",
            restoration_map={"[EMAIL_1]": "person@example.com"},
        )

        dto = build_demo_response(context)

        self.assertNotIn("raw_response", dto["technical"]["llm"]["extractor"])
        self.assertNotIn("raw_response", dto["technical"]["llm"]["assessment"])
        self.assertNotIn(
            "restoration_map",
            dto["technical"]["stage_metadata"]["SecurityGateStage"],
        )
        self.assertEqual(
            dto["technical"]["llm"]["assessment"]["provider_metadata"],
            {"model": "test-model", "provider": "test-provider"},
        )


if __name__ == "__main__":
    unittest.main()
