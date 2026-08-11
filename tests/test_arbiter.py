"""Tests for the deterministic arbiter."""

from __future__ import annotations

import tempfile
import unittest
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
)


def write_criteria_yaml(path: Path) -> None:
    """Write a temporary criteria config for arbiter tests."""
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
                  - key: placeholder_risk
                    title: Placeholder risk
                    description: Placeholder risk definition.
                    severity_hint: medium
                    signals:
                      - placeholder
                    evidence_hints:
                      - placeholder
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
                  - key: reject_critical_risk
                    title: Reject critical risk
                    description: Reject when a critical risk is present.
                    status: REJECT
                    confidence: 1.0
                    conditions:
                      - field: risk.max_severity
                        operator: in
                        value:
                          - critical
                  - key: simplify_high_risk
                    title: Simplify high risk
                    description: Simplify when high risk is present.
                    status: SIMPLIFY
                    confidence: 0.9
                    conditions:
                      - field: risk.max_severity
                        operator: in
                        value:
                          - high
                  - key: clarify_missing_information
                    title: Clarify missing information
                    description: Clarify when required information is missing.
                    status: CLARIFY
                    confidence: 0.85
                    conditions:
                      - field: completeness.missing_count
                        operator: gt
                        value: 0
                  - key: mentor_review_insufficient_information
                    title: Mentor review for uncertainty
                    description: Escalate when evaluation remains uncertain.
                    status: MENTOR_REVIEW
                    confidence: 0.8
                    conditions:
                      - field: evaluation.insufficient_information_count
                        operator: gt
                        value: 0
                      - field: completeness.missing_count
                        operator: eq
                        value: 0
                  - key: accept_ready
                    title: Accept ready brief
                    description: Accept when the brief is complete and clean.
                    status: ACCEPT
                    confidence: 0.95
                    conditions:
                      - field: completeness.is_complete
                        operator: eq
                        value: true
                      - field: risk.max_severity
                        operator: in
                        value:
                          - none
                          - low
                      - field: evaluation.not_met_count
                        operator: eq
                        value: 0
                      - field: evaluation.insufficient_information_count
                        operator: eq
                        value: 0
                      - field: evaluation.risk_detected_count
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
    """Create a compact extracted fact."""
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
) -> CompletenessResult:
    """Create a completeness result with configurable counts."""
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
        present_information=present_information if complete else present_information,
        clarification_information=clarification_information,
        warnings=[],
    )


def make_assessment_result(
    *,
    severities: list[RiskSeverity],
    statuses: list[CriterionEvaluationStatus],
) -> AssessmentResult:
    """Create unified assessment output with configurable decision signals."""
    risks = [
        Risk(
            type=f"risk_{index}",
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
    """Unit tests for the deterministic arbiter."""

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
        self.assertTrue(result.evidence)

    def test_reject_status_for_critical_risk(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.reject)
        self.assertEqual(result.triggered_rules[0].rule_key, "reject_critical_risk")

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
            "clarify_missing_information",
        )

    def test_simplify_status_for_high_risk(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.high],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.simplify)
        self.assertEqual(result.triggered_rules[0].rule_key, "simplify_high_risk")

    def test_mentor_review_status_for_insufficient_information(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.low],
                statuses=[CriterionEvaluationStatus.insufficient_information],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.mentor_review)
        self.assertEqual(
            result.triggered_rules[0].rule_key,
            "mentor_review_insufficient_information",
        )

    def test_stage_decision_branches_from_assessment_result_context(self) -> None:
        scenarios = [
            {
                "name": "accept",
                "complete": True,
                "missing": 0,
                "severities": [],
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.accept,
                "expected_rule": "accept_ready",
            },
            {
                "name": "clarify",
                "complete": False,
                "missing": 1,
                "severities": [],
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.clarify,
                "expected_rule": "clarify_missing_information",
            },
            {
                "name": "simplify",
                "complete": True,
                "missing": 0,
                "severities": [RiskSeverity.high],
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.simplify,
                "expected_rule": "simplify_high_risk",
            },
            {
                "name": "mentor_review",
                "complete": True,
                "missing": 0,
                "severities": [RiskSeverity.low],
                "statuses": [CriterionEvaluationStatus.insufficient_information],
                "expected_status": DecisionStatus.mentor_review,
                "expected_rule": "mentor_review_insufficient_information",
            },
            {
                "name": "reject",
                "complete": True,
                "missing": 0,
                "severities": [RiskSeverity.critical],
                "statuses": [CriterionEvaluationStatus.met],
                "expected_status": DecisionStatus.reject,
                "expected_rule": "reject_critical_risk",
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
                        )
                    )
                    .with_assessment_result(
                        make_assessment_result(
                            severities=scenario["severities"],
                            statuses=scenario["statuses"],
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

    def test_critical_risk_wins_over_complete_brief(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=True),
            make_assessment_result(
                severities=[RiskSeverity.critical],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.reject)

    def test_incomplete_but_acceptable_brief_is_clarify(self) -> None:
        result = self.arbiter.arbitrate_assessment(
            make_completeness_result(complete=False, missing=1),
            make_assessment_result(
                severities=[RiskSeverity.low],
                statuses=[CriterionEvaluationStatus.met],
            ),
        )

        self.assertEqual(result.final_status, DecisionStatus.clarify)

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
