"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from textwrap import dedent

from app.input import BriefInputFactory
from app.pipeline import BaseStage, DeterministicArbiterStage
from app.schemas import (
    AIContext,
    ArbitrationResult,
    AssessmentRecommendation,
    AssessmentResult,
    AssessmentTechnicalInfo,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    DecisionStatus,
    ExtractedFact,
    FactStatus,
    Risk,
    RiskSeverity,
    TrafficLightResult,
    TrafficLightStatus,
)


def write_criteria_yaml(path: Path) -> None:
    """Выполняет шаг «write criteria yaml». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    path.write_text(
        dedent(
            """
            evaluation:
              version: "1"
              description: Test arbitration configuration.
              project_types:
                - key: web_app
                  title: Web app
                  description: Web application.
                  task_types:
                    - implementation
                  aliases:
                    - website
              task_types:
                - key: implementation
                  title: Implementation
                  description: Implementation task type.
                  criteria:
                    - goal_clarity
              criteria:
                - key: goal_clarity
                  title: Goal clarity
                  description: Goal criterion.
                  complexity: medium
                  allowed_values:
                    - clear
                  decision_thresholds:
                    min_score: 0
                    max_score: 1
                    conditions:
                      - placeholder
                    description: Placeholder threshold.
                  status_signals:
                    - clear objective
              required_fields:
                - key: project_goal
                  field_path: project_goal
                  title: Project goal
                  description: Required goal.
                  required: true
                  customer_field_role: blocking
              decision_thresholds:
                - min_score: 0
                  max_score: 1
                  conditions:
                    - placeholder
                  description: Placeholder threshold.
              risk_analysis:
                version: "1"
                description: Test risk analysis configuration.
                risk_types:
                  - key: out_of_scope_request
                    title: Request outside the format
                    description: Request outside the student project format.
                    severity_hint: critical
                    signals:
                      - partnership_offer
                    evidence_hints:
                      - cooperation offered
                  - key: restricted_topic
                    title: Subject the Masterskaya does not take
                    description: Project on a restricted subject.
                    severity_hint: critical
                    signals:
                      - gambling
                    evidence_hints:
                      - casino or betting platform
                  - key: placeholder_risk
                    title: Placeholder risk
                    description: Placeholder risk definition.
                    severity_hint: medium
                    signals:
                      - placeholder
                    evidence_hints:
                      - placeholder
                  - key: scope_too_large
                    title: Scope too large
                    description: Project scope is too large for a student MVP.
                    severity_hint: high
                    signals:
                      - broad_scope
                    evidence_hints:
                      - full platform
                  - key: mentor_expertise_required
                    title: Mentor expertise required
                    description: Project needs mentor review.
                    severity_hint: medium
                    signals:
                      - specialized_expertise
                    evidence_hints:
                      - expert review needed
                  - key: production_criticality
                    title: Production criticality
                    description: Project has production-critical responsibility.
                    severity_hint: critical
                    signals:
                      - production_critical
                    evidence_hints:
                      - production responsibility
                decision_thresholds:
                  - min_score: 0
                    max_score: 1
                    conditions:
                      - placeholder
                    description: Placeholder threshold.
              arbitration:
                version: "1"
                description: Test deterministic arbitration rules.
                default_status: MENTOR_REVIEW
                rules:
                  - key: reject_by_business_risk
                    title: Reject by business risk
                    description: Reject when the brief has a concrete risk type that the Masterskaya does not accept.
                    status: REJECT
                    confidence: 1.0
                    conditions:
                      - field: risk.types
                        operator: any_in
                        value:
                          - restricted_topic
                          - out_of_scope_request
                          - production_criticality
                  - key: reject_by_traffic_light_red
                    title: Reject by traffic light red
                    description: Reject when the Traffic Light marks the requested work as red.
                    status: REJECT
                    confidence: 1.0
                    conditions:
                      - field: traffic_light.status
                        operator: eq
                        value: red
                  - key: clarify_blocking_missing_information
                    title: Clarify blocking missing information
                    description: Ask for clarification when blocking customer-facing information is missing.
                    status: CLARIFY
                    confidence: 0.85
                    conditions:
                      - field: completeness.blocking_missing_count
                        operator: gt
                        value: 0
                  - key: clarify_blocking_uncertainty
                    title: Clarify blocking uncertainty
                    description: Ask for clarification when blocking customer-facing information is uncertain.
                    status: CLARIFY
                    confidence: 0.85
                    conditions:
                      - field: completeness.blocking_clarification_count
                        operator: gt
                        value: 0
                  - key: simplify_scope_too_large
                    title: Simplify scope too large
                    description: Simplify when the project scope is too large for a student MVP.
                    status: SIMPLIFY
                    confidence: 0.9
                    conditions:
                      - field: risk.types
                        operator: any_in
                        value:
                          - scope_too_large
                  - key: mentor_review_expertise_required
                    title: Mentor review expertise required
                    description: Send to mentor review when the project requires specialist expertise before launch.
                    status: MENTOR_REVIEW
                    confidence: 0.8
                    conditions:
                      - field: risk.types
                        operator: any_in
                        value:
                          - mentor_expertise_required
                  - key: mentor_review_unknown_risk_type
                    title: Mentor review unknown risk type
                    description: Send to mentor review when assessment returns a risk type absent from configured risk taxonomy.
                    status: MENTOR_REVIEW
                    confidence: 0.75
                    conditions:
                      - field: risk.unknown_type_count
                        operator: gt
                        value: 0
                  - key: mentor_review_traffic_light_unknown
                    title: Mentor review traffic light unknown
                    description: Send to mentor review when the Traffic Light cannot classify the requested work.
                    status: MENTOR_REVIEW
                    confidence: 0.75
                    conditions:
                      - field: traffic_light.status
                        operator: eq
                        value: unknown
                  - key: accept_with_traffic_light_yellow
                    title: Accept with traffic light yellow
                    description: Accept with clarifications when the Traffic Light marks the requested work as yellow.
                    status: ACCEPT_WITH_CLARIFICATIONS
                    confidence: 0.9
                    conditions:
                      - field: traffic_light.status
                        operator: eq
                        value: yellow
                  - key: accept_with_missing_optional_information
                    title: Accept with missing optional information
                    description: Accept the brief while asking for optional customer-facing information that is missing.
                    status: ACCEPT_WITH_CLARIFICATIONS
                    confidence: 0.9
                    conditions:
                      - field: completeness.optional_missing_count
                        operator: gt
                        value: 0
                  - key: accept_with_optional_uncertainty
                    title: Accept with optional uncertainty
                    description: Accept the brief while asking for optional customer-facing information that is uncertain.
                    status: ACCEPT_WITH_CLARIFICATIONS
                    confidence: 0.9
                    conditions:
                      - field: completeness.optional_clarification_count
                        operator: gt
                        value: 0
                  - key: accept_ready
                    title: Accept ready brief
                    description: Accept when no higher-priority rule matched and no customer-facing clarification remains.
                    status: ACCEPT
                    confidence: 0.95
                    conditions:
                      - field: completeness.blocking_missing_count
                        operator: eq
                        value: 0
                      - field: completeness.blocking_clarification_count
                        operator: eq
                        value: 0
                      - field: completeness.optional_missing_count
                        operator: eq
                        value: 0
                      - field: completeness.optional_clarification_count
                        operator: eq
                        value: 0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def make_fact(
    value: str,
    *,
    status: FactStatus = FactStatus.explicit,
) -> ExtractedFact:
    """Выполняет шаг «make fact». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return ExtractedFact(
        status=status,
        value=value,
        evidence=[value],
        confidence=0.9,
        notes=None,
    )


def make_completeness_result(
    *,
    complete: bool,
    missing: int = 0,
    clarification: int = 0,
    optional_missing: int = 0,
    optional_clarification: int = 0,
) -> CompletenessResult:
    """Выполняет шаг «make completeness result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    missing_information = [
        CompletenessItem(
            field_key=f"missing_{index}",
            field_path="project_goal",
            title=f"Missing {index}",
            status=CompletenessStatus.missing,
            value=None,
            reason="Missing required information",
            notes=None,
        )
        for index in range(missing)
    ]
    clarification_information = [
        CompletenessItem(
            field_key=f"clarify_{index}",
            field_path="project_goal",
            title=f"Clarify {index}",
            status=CompletenessStatus.clarification,
            value=None,
            reason="Needs clarification",
            notes=None,
        )
        for index in range(clarification)
    ]
    optional_missing_information = [
        CompletenessItem(
            field_key=f"optional_missing_{index}",
            field_path="materials",
            title=f"Optional missing {index}",
            status=CompletenessStatus.missing,
            value=None,
            reason="Optional information is missing",
            notes=None,
        )
        for index in range(optional_missing)
    ] + [
        CompletenessItem(
            field_key=f"optional_clarify_{index}",
            field_path="materials",
            title=f"Optional clarify {index}",
            status=CompletenessStatus.clarification,
            value=None,
            reason="Optional information needs clarification",
            notes=None,
        )
        for index in range(optional_clarification)
    ]
    present_information = [
        CompletenessItem(
            field_key="project_goal",
            field_path="project_goal",
            title="Project goal",
            status=CompletenessStatus.present,
            value="Build a web app",
            reason=None,
            notes=None,
        )
    ]
    return CompletenessResult(
        is_complete=complete,
        missing_information=missing_information,
        critical_missing_information=missing_information,
        optional_missing_information=optional_missing_information,
        present_information=present_information if complete else present_information,
        clarification_information=clarification_information,
        warnings=[],
    )


def make_assessment_result(
    *,
    severities: list[RiskSeverity],
    statuses: list[CriterionEvaluationStatus],
    risk_types: list[str] | None = None,
    traffic_light_status: TrafficLightStatus = TrafficLightStatus.green,
) -> AssessmentResult:
    """Выполняет шаг «make assessment result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    risks = [
        Risk(
            type=(
                risk_types[index - 1]
                if risk_types is not None
                else f"risk_{index}"
            ),
            description=f"{severity.value} risk",
            severity=severity,
            evidence=[f"{severity.value} evidence {index}"],
            confidence=0.8,
            notes=None,
        )
        for index, severity in enumerate(severities, start=1)
    ]
    criterion_evaluations = [
        CriterionEvaluation(
            criterion=f"criterion_{index}",
            criterion_title=f"Criterion {index}",
            status=status,
            evidence=[f"criterion evidence {index}"],
            explanation=f"Criterion {index} explanation",
            confidence=0.8,
            notes=None,
        )
        for index, status in enumerate(statuses, start=1)
    ]
    return AssessmentResult(
        criterion_evaluations=criterion_evaluations,
        risks=risks,
        evidence=[],
        has_risks=bool(risks),
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        summary=None,
        traffic_light=TrafficLightResult(status=traffic_light_status),
        technical_info=AssessmentTechnicalInfo(
            attempts=1,
            prompt_name="assessment.md",
            trace_enabled=False,
            trace_name="assessment.brief",
            model_name=None,
            retriever_used=False,
            retrieved_context_count=0,
            criteria_count=len(criterion_evaluations),
            risk_types_count=0,
            raw_response=None,
            recovered_errors=[],
        ),
    )


class TestDeterministicArbiterStage(unittest.TestCase):
    """Класс «TestDeterministicArbiterStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.criteria_path = Path(self.tmpdir.name) / "criteria.yaml"
        write_criteria_yaml(self.criteria_path)
        self.arbiter = DeterministicArbiterStage(criteria_path=self.criteria_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_accept_status(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.accept)
        self.assertEqual(result.triggered_rules[0].rule_key, "accept_ready")

    def test_arbitration_diagnostics_are_written_to_stderr(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = self.arbiter.arbitrate_assessment(
                make_completeness_result(complete=True),
                make_assessment_result(
                    severities=[],
                    statuses=[CriterionEvaluationStatus.met],
                ),
            )

        self.assertEqual(result.final_status, DecisionStatus.accept)
        self.assertNotIn("[ARBITRATION DIAGNOSTICS]", stdout.getvalue())
        self.assertIn("[ARBITRATION DIAGNOSTICS]", stderr.getvalue())

    def test_rule_order_matches_decision_priority(self) -> None:
        self.assertEqual(
            [rule.key for rule in self.arbiter._arbitration.rules],
            [
                "reject_by_business_risk",
                "reject_by_traffic_light_red",
                "clarify_blocking_missing_information",
                "clarify_blocking_uncertainty",
                "simplify_scope_too_large",
                "mentor_review_expertise_required",
                "mentor_review_unknown_risk_type",
                "mentor_review_traffic_light_unknown",
                "accept_with_traffic_light_yellow",
                "accept_with_missing_optional_information",
                "accept_with_optional_uncertainty",
                "accept_ready",
            ],
        )

    def test_rules_do_not_use_old_coarse_business_signals(self) -> None:
        rule_fields = [
            condition.field
            for rule in self.arbiter._arbitration.rules
            for condition in rule.conditions
        ]

        self.assertNotIn("completeness.missing_count", rule_fields)
        self.assertNotIn("risk.max_severity", rule_fields)
        self.assertNotIn("evaluation.insufficient_information_count", rule_fields)

    def test_accept_status_returns_accept_ready(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.accept)
        self.assertEqual(result.triggered_rules[0].rule_key, "accept_ready")

    def test_reject_status_for_production_criticality(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["production_criticality"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.reject)
        self.assertEqual(result.triggered_rules[0].rule_key, "reject_by_business_risk")

    def test_reject_status_for_out_of_scope_request(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, missing=2),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.not_met],
                risk_types=["out_of_scope_request"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.reject)
        self.assertEqual(result.triggered_rules[0].rule_key, "reject_by_business_risk")
        self.assertEqual(
            result.evidence,
            ["out_of_scope_request: critical risk"],
        )

    def test_out_of_scope_request_outranks_simplify(self) -> None:
        # Правило стоит первым, поэтому тип риска решает даже тогда,
        # когда severity сама по себе увела бы бриф в SIMPLIFY.
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.not_met],
                risk_types=["out_of_scope_request"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.reject)
        self.assertEqual(result.triggered_rules[0].rule_key, "reject_by_business_risk")

    def test_reject_status_for_restricted_topic(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.not_met],
                risk_types=["restricted_topic"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.reject)
        self.assertEqual(
            result.triggered_rules[0].rule_key, "reject_by_business_risk"
        )

    def test_restricted_topic_outranks_generic_critical_risk(self) -> None:
        # Оба правила дают REJECT, но именованное стоит выше: иначе основание
        # отказа в трейсе выглядело бы как обычный критический риск.
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, missing=2),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.not_met],
                risk_types=["restricted_topic"],
            ),
        )

        self.assertEqual(
            result.triggered_rules[0].rule_key, "reject_by_business_risk"
        )

    def test_scope_too_large_is_simplify(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["scope_too_large"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.simplify)
        self.assertEqual(result.triggered_rules[0].rule_key, "simplify_scope_too_large")

    def test_mentor_expertise_required_is_mentor_review(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.medium],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["mentor_expertise_required"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_expertise_required",
        )

    def test_unknown_high_risk_type_is_mentor_review_fallback(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["unknown_delivery_risk"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_unknown_risk_type",
        )
        self.assertEqual(result.metadata["signals"]["risk.unknown_type_count"], 1)

    def test_unknown_critical_risk_type_is_mentor_review_fallback(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["unknown_critical_risk"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_unknown_risk_type",
        )
        self.assertEqual(result.metadata["signals"]["risk.unknown_type_count"], 1)

    def test_unknown_type_signals_include_only_types_absent_from_taxonomy(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[
                    RiskSeverity.high,
                    RiskSeverity.medium,
                    RiskSeverity.critical,
                ],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=[
                    "unknown_delivery_risk",
                    "mentor_expertise_required",
                    "production_criticality",
                ],
            ),
        )

        self.assertEqual(
            result.metadata["signals"]["risk.unknown_types"],
            ["unknown_delivery_risk"],
        )
        self.assertEqual(result.metadata["signals"]["risk.unknown_type_count"], 1)

    def test_unknown_risk_type_does_not_outrank_blocking_gap(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, missing=1),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["unknown_delivery_risk"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.clarify)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "clarify_blocking_missing_information",
        )
        self.assertEqual(result.metadata["signals"]["risk.unknown_type_count"], 1)

    def test_unknown_risk_type_does_not_outrank_scope_too_large(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.high, RiskSeverity.medium],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["unknown_delivery_risk", "scope_too_large"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.simplify)
        self.assertEqual(result.triggered_rules[0].rule_key, "simplify_scope_too_large")
        self.assertEqual(result.metadata["signals"]["risk.unknown_type_count"], 1)

    def test_unknown_risk_type_outranks_optional_gap(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True, optional_missing=1),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["unknown_delivery_risk"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_unknown_risk_type",
        )
        self.assertEqual(result.metadata["signals"]["risk.unknown_type_count"], 1)

    def test_unknown_risk_type_outranks_optional_uncertainty(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True, optional_clarification=1),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["unknown_delivery_risk"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_unknown_risk_type",
        )
        self.assertEqual(result.metadata["signals"]["risk.unknown_type_count"], 1)

    def test_unknown_low_risk_type_is_mentor_review(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.low],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["unknown_low_risk"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_unknown_risk_type",
        )

    def test_unknown_medium_risk_type_is_mentor_review(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.medium],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["unknown_medium_risk"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_unknown_risk_type",
        )

    def test_optional_missing_information_is_accept_with_clarifications(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True, optional_missing=1),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(
            result.final_status,
            DecisionStatus.accept_with_clarifications,
        )
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "accept_with_missing_optional_information",
        )

    def test_optional_uncertainty_is_accept_with_clarifications(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True, optional_clarification=1),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(
            result.final_status,
            DecisionStatus.accept_with_clarifications,
        )
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "accept_with_optional_uncertainty",
        )

    def test_reasons_hold_only_rule_description(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["production_criticality"],
            ),
        )

        self.assertEqual(
            result.reasons,
            [
                "Reject when the brief has a concrete risk type that "
                "the Masterskaya does not accept."
            ],
        )

    def test_rule_hit_keeps_conditions_and_signals(self) -> None:
        # Отладочные строки убраны из reasons, но сами данные должны остаться:
        # без них нельзя разобрать, почему арбитр выбрал именно это правило.
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["production_criticality"],
            ),
        )

        hit = result.triggered_rules[0]
        self.assertEqual(
            hit.conditions,
            [
                "risk.types any_in "
                "['restricted_topic', 'out_of_scope_request', "
                "'production_criticality']"
            ],
        )
        self.assertEqual(
            hit.metadata["signals"],
            {"risk.types": ["production_criticality"]},
        )
        self.assertEqual(result.metadata["signals"]["risk.max_severity"], "critical")

    def test_completeness_role_signals_are_available(self) -> None:
        completeness_result = make_completeness_result(
            complete=False,
            missing=2,
            clarification=1,
            optional_missing=3,
            optional_clarification=4,
        )

        signals = self.arbiter._build_signals(
            completeness_result=completeness_result,
            risks=[],
            criterion_evaluations=[],
        )

        self.assertEqual(signals["completeness.blocking_missing_count"], 2)
        self.assertEqual(signals["completeness.blocking_clarification_count"], 1)
        self.assertEqual(signals["completeness.optional_missing_count"], 3)
        self.assertEqual(signals["completeness.optional_clarification_count"], 4)
        self.assertIn(
            "completeness.blocking_missing_count",
            self.arbiter._supported_signals,
        )
        self.assertIn(
            "completeness.blocking_clarification_count",
            self.arbiter._supported_signals,
        )
        self.assertIn(
            "completeness.optional_missing_count",
            self.arbiter._supported_signals,
        )
        self.assertIn(
            "completeness.optional_clarification_count",
            self.arbiter._supported_signals,
        )

    def test_traffic_light_red_status_is_available_as_signal(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
                traffic_light_status=TrafficLightStatus.red,
            ),
        )

        self.assertEqual(result.metadata["signals"]["traffic_light.status"], "red")

    def test_traffic_light_green_status_is_available_as_signal(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
                traffic_light_status=TrafficLightStatus.green,
            ),
        )

        self.assertEqual(result.metadata["signals"]["traffic_light.status"], "green")

    def test_traffic_light_yellow_status_is_available_as_signal(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
                traffic_light_status=TrafficLightStatus.yellow,
            ),
        )

        self.assertEqual(result.metadata["signals"]["traffic_light.status"], "yellow")

    def test_traffic_light_unknown_status_is_available_as_signal(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
                traffic_light_status=TrafficLightStatus.unknown,
            ),
        )

        self.assertEqual(result.metadata["signals"]["traffic_light.status"], "unknown")

    def test_traffic_light_status_is_supported_in_config_conditions(self) -> None:
        self.criteria_path.write_text(
            self.criteria_path.read_text(encoding="utf-8").replace(
                "field: risk.unknown_type_count",
                "field: traffic_light.status",
                1,
            ),
            encoding="utf-8",
        )

        stage = DeterministicArbiterStage(criteria_path=self.criteria_path)

        self.assertIn("traffic_light.status", stage._supported_signals)

    def test_traffic_light_rules_follow_configured_priority(self) -> None:
        scenarios = [
            {
                "name": "red_good_brief",
                "traffic_light_status": TrafficLightStatus.red,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": None,
                "severities": [],
                "expected_status": DecisionStatus.reject,
                "expected_rule": "reject_by_traffic_light_red",
            },
            {
                "name": "red_blocking_missing",
                "traffic_light_status": TrafficLightStatus.red,
                "complete": False,
                "missing": 1,
                "optional_missing": 0,
                "risk_types": None,
                "severities": [],
                "expected_status": DecisionStatus.reject,
                "expected_rule": "reject_by_traffic_light_red",
            },
            {
                "name": "red_scope_too_large",
                "traffic_light_status": TrafficLightStatus.red,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": ["scope_too_large"],
                "severities": [RiskSeverity.high],
                "expected_status": DecisionStatus.reject,
                "expected_rule": "reject_by_traffic_light_red",
            },
            {
                "name": "red_production_criticality",
                "traffic_light_status": TrafficLightStatus.red,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": ["production_criticality"],
                "severities": [RiskSeverity.critical],
                "expected_status": DecisionStatus.reject,
                "expected_rule": "reject_by_business_risk",
            },
            {
                "name": "yellow_good_brief",
                "traffic_light_status": TrafficLightStatus.yellow,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": None,
                "severities": [],
                "expected_status": DecisionStatus.accept_with_clarifications,
                "expected_rule": "accept_with_traffic_light_yellow",
            },
            {
                "name": "yellow_blocking_missing",
                "traffic_light_status": TrafficLightStatus.yellow,
                "complete": False,
                "missing": 1,
                "optional_missing": 0,
                "risk_types": None,
                "severities": [],
                "expected_status": DecisionStatus.clarify,
                "expected_rule": "clarify_blocking_missing_information",
            },
            {
                "name": "yellow_scope_too_large",
                "traffic_light_status": TrafficLightStatus.yellow,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": ["scope_too_large"],
                "severities": [RiskSeverity.high],
                "expected_status": DecisionStatus.simplify,
                "expected_rule": "simplify_scope_too_large",
            },
            {
                "name": "unknown_good_brief",
                "traffic_light_status": TrafficLightStatus.unknown,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": None,
                "severities": [],
                "expected_status": DecisionStatus.mentor_review,
                "expected_rule": "mentor_review_traffic_light_unknown",
            },
            {
                "name": "unknown_blocking_missing",
                "traffic_light_status": TrafficLightStatus.unknown,
                "complete": False,
                "missing": 1,
                "optional_missing": 0,
                "risk_types": None,
                "severities": [],
                "expected_status": DecisionStatus.clarify,
                "expected_rule": "clarify_blocking_missing_information",
            },
            {
                "name": "unknown_scope_too_large",
                "traffic_light_status": TrafficLightStatus.unknown,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": ["scope_too_large"],
                "severities": [RiskSeverity.high],
                "expected_status": DecisionStatus.simplify,
                "expected_rule": "simplify_scope_too_large",
            },
            {
                "name": "green_good_brief",
                "traffic_light_status": TrafficLightStatus.green,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": None,
                "severities": [],
                "expected_status": DecisionStatus.accept,
                "expected_rule": "accept_ready",
            },
            {
                "name": "green_scope_too_large",
                "traffic_light_status": TrafficLightStatus.green,
                "complete": True,
                "missing": 0,
                "optional_missing": 0,
                "risk_types": ["scope_too_large"],
                "severities": [RiskSeverity.high],
                "expected_status": DecisionStatus.simplify,
                "expected_rule": "simplify_scope_too_large",
            },
            {
                "name": "green_optional_missing",
                "traffic_light_status": TrafficLightStatus.green,
                "complete": True,
                "missing": 0,
                "optional_missing": 1,
                "risk_types": None,
                "severities": [],
                "expected_status": DecisionStatus.accept_with_clarifications,
                "expected_rule": "accept_with_missing_optional_information",
            },
        ]

        for scenario in scenarios:
            with self.subTest(scenario=scenario["name"]):
                result = self.arbiter.arbitrate_assessment(
                    make_completeness_result(
                        complete=scenario["complete"],
                        missing=scenario["missing"],
                        optional_missing=scenario["optional_missing"],
                    ),
                    make_assessment_result(
                        severities=scenario["severities"],
                        statuses=[CriterionEvaluationStatus.met],
                        risk_types=scenario["risk_types"],
                        traffic_light_status=scenario["traffic_light_status"],
                    ),
                )

                self.assertEqual(result.final_status, scenario["expected_status"])
                self.assertEqual(
                    result.triggered_rules[0].rule_key,
                    scenario["expected_rule"],
                )

    def test_clarify_status_for_missing_information(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, missing=1),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.clarify)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "clarify_blocking_missing_information",
        )

    def test_clarify_status_for_blocking_clarification(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, clarification=1),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.clarify)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "clarify_blocking_uncertainty",
        )

    def test_high_severity_without_scope_type_does_not_simplify(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["placeholder_risk"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.accept)
        self.assertEqual(result.triggered_rules[0].rule_key, "accept_ready")

    def test_insufficient_information_evaluation_does_not_force_mentor_review(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[],
                statuses=[CriterionEvaluationStatus.insufficient_information],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.accept)
        self.assertEqual(result.triggered_rules[0].rule_key, "accept_ready")

    def test_stage_decision_branches_from_assessment_result_context(self) -> None:
        scenarios = [
            {
                "name": "accept",
                "complete": True,
                "missing": 0,
                "clarification": 0,
                "optional_missing": 0,
                "optional_clarification": 0,
                "severities": [],
                "risk_types": None,
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.accept,
                "expected_rule": "accept_ready",
            },
            {
                "name": "clarify",
                "complete": False,
                "missing": 1,
                "clarification": 0,
                "optional_missing": 0,
                "optional_clarification": 0,
                "severities": [],
                "risk_types": None,
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.clarify,
                "expected_rule": "clarify_blocking_missing_information",
            },
            {
                "name": "simplify",
                "complete": True,
                "missing": 0,
                "clarification": 0,
                "optional_missing": 0,
                "optional_clarification": 0,
                "severities": [RiskSeverity.high],
                "risk_types": ["scope_too_large"],
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.simplify,
                "expected_rule": "simplify_scope_too_large",
            },
            {
                "name": "mentor_review",
                "complete": True,
                "missing": 0,
                "clarification": 0,
                "optional_missing": 0,
                "optional_clarification": 0,
                "severities": [RiskSeverity.medium],
                "risk_types": ["mentor_expertise_required"],
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.mentor_review,
                "expected_rule": "mentor_review_expertise_required",
            },
            {
                "name": "accept_with_clarifications",
                "complete": True,
                "missing": 0,
                "clarification": 0,
                "optional_missing": 1,
                "optional_clarification": 0,
                "severities": [],
                "risk_types": None,
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.accept_with_clarifications,
                "expected_rule": "accept_with_missing_optional_information",
            },
            {
                "name": "reject",
                "complete": True,
                "missing": 0,
                "clarification": 0,
                "optional_missing": 0,
                "optional_clarification": 0,
                "severities": [RiskSeverity.critical],
                "risk_types": ["production_criticality"],
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.reject,
                "expected_rule": "reject_by_business_risk",
            },
        ]

        for scenario in scenarios:
            with self.subTest(scenario=scenario["name"]):
                context = (
                    AIContext.from_brief(
                        BriefInputFactory().from_text("Build a web app")
                    )
                    .with_completeness_result(
                        make_completeness_result(
                            complete=scenario["complete"],
                            missing=scenario["missing"],
                            clarification=scenario["clarification"],
                            optional_missing=scenario["optional_missing"],
                            optional_clarification=scenario[
                                "optional_clarification"
                            ],
                        )
                    )
                    .with_assessment_result(
                        make_assessment_result(
                            severities=scenario["severities"],
                            statuses=scenario["statuses"],
                            risk_types=scenario["risk_types"],
                        )
                    )
                )

                updated = self.arbiter.run_context(context)

                self.assertFalse(hasattr(context, "risk_analysis_result"))
                self.assertFalse(hasattr(context, "evaluation_result"))
                self.assertEqual(
                    updated.arbitration_result.final_status,
                    scenario["expected_status"],
                )
                self.assertEqual(
                    updated.arbitration_result.triggered_rules[0].rule_key,
                    scenario["expected_rule"],
                )

    def test_reject_outranks_clarify_and_simplify(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, missing=1),
            make_assessment_result(
                severities=[RiskSeverity.critical, RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["production_criticality", "scope_too_large"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.reject)
        self.assertEqual(result.triggered_rules[0].rule_key, "reject_by_business_risk")

    def test_reject_outranks_mentor_review(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.critical, RiskSeverity.medium],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["production_criticality", "mentor_expertise_required"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.reject)
        self.assertEqual(result.triggered_rules[0].rule_key, "reject_by_business_risk")

    def test_incomplete_but_acceptable_brief_is_clarify(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, missing=1),
            make_assessment_result(
                severities=[RiskSeverity.low],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.clarify)

    def test_blocking_gap_outranks_mentor_review(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, missing=1),
            make_assessment_result(
                severities=[RiskSeverity.medium],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["mentor_expertise_required"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.clarify)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "clarify_blocking_missing_information",
        )

    def test_clarify_outranks_simplify(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, clarification=1),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["scope_too_large"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.clarify)
        self.assertEqual(result.triggered_rules[0].rule_key, "clarify_blocking_uncertainty")

    def test_simplify_outranks_mentor_review_and_optional_clarifications(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True, optional_missing=1),
            make_assessment_result(
                severities=[RiskSeverity.high, RiskSeverity.medium],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["scope_too_large", "mentor_expertise_required"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.simplify)
        self.assertEqual(result.triggered_rules[0].rule_key, "simplify_scope_too_large")

    def test_mentor_review_outranks_accept_with_clarifications(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True, optional_clarification=1),
            make_assessment_result(
                severities=[RiskSeverity.medium],
                statuses=[CriterionEvaluationStatus.met],
                risk_types=["mentor_expertise_required"],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_expertise_required",
        )

    def test_reproducible_results(self) -> None:
        completeness_result = make_completeness_result(complete=True)
        assessment_result = make_assessment_result(
            severities=[RiskSeverity.low],
            statuses=[CriterionEvaluationStatus.met],
        )

        first = self.arbiter.arbitrate_assessment(
            completeness_result,
            assessment_result,
        )
        second = self.arbiter.arbitrate_assessment(
            completeness_result,
            assessment_result,
        )

        self.assertEqual(first, second)

    def test_parses_accept_with_clarifications_status_from_configuration(self) -> None:
        rule = self.arbiter._arbitration.rules[-1].model_copy(
            update={"status": "ACCEPT_WITH_CLARIFICATIONS"}
        )

        result = self.arbiter._build_result(
            rule=rule,
            signals={"completeness.is_complete": True},
            evidence_map={},
        )

        self.assertEqual(
            result.final_status,
            DecisionStatus.accept_with_clarifications,
        )

    def test_arbiter_stage_updates_context_via_base_stage(self) -> None:
        stage = DeterministicArbiterStage(criteria_path=self.criteria_path)
        context = (
            AIContext.from_brief(BriefInputFactory().from_text("Build a web app"))
            .with_completeness_result(make_completeness_result(complete=True))
            .with_assessment_result(
                make_assessment_result(
                    severities=[],
                    statuses=[CriterionEvaluationStatus.met],
                )
            )
        )

        updated = stage.run(context)

        self.assertIsInstance(stage, BaseStage)
        self.assertIsNone(context.arbitration_result)
        self.assertEqual(updated.arbitration_result.final_status, DecisionStatus.accept)


if __name__ == "__main__":
    unittest.main()
