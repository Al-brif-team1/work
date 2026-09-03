"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import unittest

from app.config import CriteriaConfig
from app.input import BriefInputFactory
from app.pipeline import BaseStage, QuestionGenerationError, TemplateQuestionGeneratorStage
from app.schemas import (
    AIContext,
    ArbitrationResult,
    AssessmentResult,
    AssessmentRecommendation,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    TrafficLightMatch,
    TrafficLightResult,
    TrafficLightStatus,
    DecisionStatus,
)


def make_missing_item(field_key: str, title: str) -> CompletenessItem:
    """Выполняет шаг «make missing item». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return CompletenessItem(
        field_key=field_key,
        field_path=field_key,
        title=title,
        status=CompletenessStatus.missing,
        value=None,
        reason="Missing required information",
        notes=None,
    )


def make_clarification_item(field_key: str, title: str) -> CompletenessItem:
    return CompletenessItem(
        field_key=field_key,
        field_path=field_key,
        title=title,
        status=CompletenessStatus.clarification,
        value="uncertain value",
        reason=f"{title} requires clarification",
        notes=None,
    )


def make_arbitration_result(status: DecisionStatus) -> ArbitrationResult:
    return ArbitrationResult(final_status=status)


def make_assessment_with_traffic_light(
    status: TrafficLightStatus,
    *,
    matches: list[TrafficLightMatch] | None = None,
) -> AssessmentResult:
    return AssessmentResult(
        criterion_evaluations=[],
        risks=[],
        has_risks=False,
        recommendation=AssessmentRecommendation.ready_for_arbitration,
        traffic_light=TrafficLightResult(
            status=status,
            direction="analytics",
            specialization="dashboards",
            matches=matches or [],
        ),
    )


def make_yellow_match(
    *,
    task: str = "Build dashboard with advanced filters",
    matched_rule: str = "Advanced dashboard for analytics students",
    reason: str = "Students can do it if the first version has a bounded scope",
) -> TrafficLightMatch:
    return TrafficLightMatch(
        task=task,
        matched_rule=matched_rule,
        status=TrafficLightStatus.yellow,
        reason=reason,
    )


def make_criteria_config(
    customer_field_roles: dict[str, str],
) -> CriteriaConfig:
    required_fields = [
        {
            "key": field_key,
            "field_path": field_key,
            "title": field_key.replace("_", " ").title(),
            "description": f"{field_key} field.",
            "required": role == "blocking",
            "customer_field_role": role,
        }
        for field_key, role in customer_field_roles.items()
    ]
    return CriteriaConfig.model_validate(
        {
            "evaluation": {
                "version": "1",
                "description": "Question generator test criteria.",
                "project_types": [
                    {
                        "key": "web_app",
                        "title": "Web app",
                        "description": "Web app.",
                        "task_types": ["implementation"],
                        "aliases": [],
                    }
                ],
                "task_types": [
                    {
                        "key": "implementation",
                        "title": "Implementation",
                        "description": "Implementation.",
                        "criteria": ["goal_clarity"],
                    }
                ],
                "criteria": [
                    {
                        "key": "goal_clarity",
                        "title": "Goal clarity",
                        "description": "Goal clarity.",
                    }
                ],
                "required_fields": required_fields,
            }
        }
    )


def make_completeness_result(
    missing_information: list[CompletenessItem],
    *,
    clarification_information: list[CompletenessItem] | None = None,
    optional_missing_information: list[CompletenessItem] | None = None,
) -> CompletenessResult:
    """Выполняет шаг «make completeness result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return CompletenessResult(
        is_complete=not missing_information,
        missing_information=missing_information,
        present_information=[] if missing_information else [
            CompletenessItem(
                field_key="project_goal",
                field_path="project_goal",
                title="Project goal",
                status=CompletenessStatus.present,
                value="Build a web app",
                reason=None,
                notes=None,
            )
        ],
        clarification_information=clarification_information or [],
        optional_missing_information=optional_missing_information or [],
        warnings=[],
    )


class TestTemplateQuestionGeneratorStage(unittest.TestCase):
    """Класс «TestTemplateQuestionGeneratorStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_returns_empty_questions_for_complete_brief(self) -> None:
        stage = TemplateQuestionGeneratorStage()

        result = stage.generate(make_completeness_result([]))

        self.assertEqual(result.questions, [])
        self.assertFalse(result.technical_info.llm_invoked)
        self.assertEqual(result.technical_info.attempts, 0)

    def test_generates_questions_for_missing_information(self) -> None:
        stage = TemplateQuestionGeneratorStage()

        result = stage.generate(
            make_completeness_result(
                [
                    make_missing_item("project_goal", "Project goal"),
                    make_missing_item("tasks", "Tasks"),
                ]
            )
        )

        self.assertEqual(len(result.questions), 2)
        self.assertEqual(result.questions[0].related_field, "project_goal")
        self.assertFalse(result.technical_info.llm_invoked)
        self.assertEqual(result.technical_info.question_count, 2)
        self.assertEqual(result.technical_info.missing_template_fields, [])

    def test_clarify_generates_questions_for_blocking_missing_and_clarification(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage(
            templates={
                "project_goal": "What is the main goal?",
                "expected_result": "What should be delivered?",
                "materials": "What materials are available?",
            }
        )

        result = stage.generate(
            make_completeness_result(
                [make_missing_item("project_goal", "Project goal")],
                clarification_information=[
                    make_clarification_item("expected_result", "Expected result")
                ],
                optional_missing_information=[
                    make_missing_item("materials", "Available materials")
                ],
            ),
            arbitration_result=make_arbitration_result(DecisionStatus.clarify),
        )

        self.assertEqual(
            [question.related_field for question in result.questions],
            ["project_goal", "expected_result"],
        )

    def test_accept_with_clarifications_generates_questions_for_optional_items(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage(
            templates={
                "project_goal": "What is the main goal?",
                "materials": "What materials are available?",
                "deadlines": "What deadlines matter?",
            }
        )

        result = stage.generate(
            make_completeness_result(
                [make_missing_item("project_goal", "Project goal")],
                optional_missing_information=[
                    make_missing_item("materials", "Available materials"),
                    make_clarification_item("deadlines", "Deadlines"),
                ],
            ),
            arbitration_result=make_arbitration_result(
                DecisionStatus.accept_with_clarifications
            ),
        )

        self.assertEqual(
            [question.related_field for question in result.questions],
            ["materials", "deadlines"],
        )

    def test_accept_with_clarifications_generates_traffic_light_yellow_question_for_complete_brief(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage()
        match = make_yellow_match()

        result = stage.generate(
            make_completeness_result([]),
            assessment_result=make_assessment_with_traffic_light(
                TrafficLightStatus.yellow,
                matches=[match],
            ),
            arbitration_result=make_arbitration_result(
                DecisionStatus.accept_with_clarifications
            ),
        )

        self.assertGreaterEqual(len(result.questions), 1)
        self.assertEqual(result.questions[0].related_field, "traffic_light")
        self.assertIn(match.task, result.questions[0].question)
        self.assertIn(match.reason, result.questions[0].question)
        self.assertFalse(result.technical_info.llm_invoked)

    def test_accept_with_clarifications_keeps_optional_question_with_traffic_light_yellow(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage(
            templates={"materials": "What materials are available?"}
        )

        result = stage.generate(
            make_completeness_result(
                [],
                optional_missing_information=[
                    make_missing_item("materials", "Available materials")
                ],
            ),
            assessment_result=make_assessment_with_traffic_light(
                TrafficLightStatus.yellow,
                matches=[make_yellow_match()],
            ),
            arbitration_result=make_arbitration_result(
                DecisionStatus.accept_with_clarifications
            ),
        )

        self.assertEqual(result.questions[0].question, "What materials are available?")
        self.assertIn("traffic_light", [question.related_field for question in result.questions])

    def test_duplicate_traffic_light_yellow_matches_do_not_create_duplicate_questions(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage()
        match = make_yellow_match()

        result = stage.generate(
            make_completeness_result([]),
            assessment_result=make_assessment_with_traffic_light(
                TrafficLightStatus.yellow,
                matches=[match, match.model_copy()],
            ),
            arbitration_result=make_arbitration_result(
                DecisionStatus.accept_with_clarifications
            ),
        )

        self.assertEqual(
            [question.related_field for question in result.questions],
            ["traffic_light"],
        )

    def test_traffic_light_fallback_questions_are_not_generated_for_non_yellow_statuses(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage()

        for status in (
            TrafficLightStatus.green,
            TrafficLightStatus.unknown,
            TrafficLightStatus.red,
        ):
            with self.subTest(status=status):
                result = stage.generate(
                    make_completeness_result([]),
                    assessment_result=make_assessment_with_traffic_light(
                        status,
                        matches=[
                            TrafficLightMatch(
                                task="Build dashboard",
                                matched_rule="Dashboard rule",
                                status=status,
                                reason="Traffic Light reason",
                            )
                        ]
                        if status is not TrafficLightStatus.unknown
                        else [],
                    ),
                    arbitration_result=make_arbitration_result(
                        DecisionStatus.accept_with_clarifications
                    ),
                )

                self.assertEqual(result.questions, [])

    def test_integrations_optional_missing_and_uncertain_generate_questions(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage(
            templates={
                "integrations": "Are integrations needed?",
            }
        )

        for item in [
            make_missing_item("integrations", "Integrations"),
            make_clarification_item("integrations", "Integrations"),
        ]:
            with self.subTest(status=item.status):
                result = stage.generate(
                    make_completeness_result(
                        [],
                        optional_missing_information=[item],
                    ),
                    arbitration_result=make_arbitration_result(
                        DecisionStatus.accept_with_clarifications
                    ),
                )

                self.assertEqual(
                    [question.related_field for question in result.questions],
                    ["integrations"],
                )

    def test_internal_customer_field_role_does_not_generate_question(self) -> None:
        stage = TemplateQuestionGeneratorStage(
            criteria_config=make_criteria_config(
                {
                    "project_goal": "blocking",
                    "internal_reference": "internal",
                }
            ),
            templates={
                "project_goal": "What is the main goal?",
                "internal_reference": "What is the internal reference?",
            }
        )

        result = stage.generate(
            make_completeness_result(
                [
                    make_missing_item("project_goal", "Project goal"),
                    make_missing_item("internal_reference", "Internal reference"),
                ]
            ),
            arbitration_result=make_arbitration_result(DecisionStatus.clarify),
        )

        self.assertEqual(
            [question.related_field for question in result.questions],
            ["project_goal"],
        )

    def test_question_generator_uses_customer_field_role_source_of_truth(self) -> None:
        templates = {"review_note": "What review note is needed?"}
        completeness_result = make_completeness_result(
            [make_missing_item("review_note", "Review note")]
        )

        internal_stage = TemplateQuestionGeneratorStage(
            criteria_config=make_criteria_config({"review_note": "internal"}),
            templates=templates,
        )
        blocking_stage = TemplateQuestionGeneratorStage(
            criteria_config=make_criteria_config({"review_note": "blocking"}),
            templates=templates,
        )

        internal_result = internal_stage.generate(
            completeness_result,
            arbitration_result=make_arbitration_result(DecisionStatus.clarify),
        )
        blocking_result = blocking_stage.generate(
            completeness_result,
            arbitration_result=make_arbitration_result(DecisionStatus.clarify),
        )

        self.assertEqual(internal_result.questions, [])
        self.assertEqual(
            [question.related_field for question in blocking_result.questions],
            ["review_note"],
        )

    def test_clarify_skips_optional_field_misplaced_in_blocking_list(self) -> None:
        stage = TemplateQuestionGeneratorStage(
            criteria_config=make_criteria_config({"materials": "optional"}),
            templates={"materials": "What materials are available?"},
        )

        result = stage.generate(
            make_completeness_result(
                [make_missing_item("materials", "Available materials")]
            ),
            arbitration_result=make_arbitration_result(DecisionStatus.clarify),
        )

        self.assertEqual(result.questions, [])

    def test_accept_with_clarifications_skips_blocking_field_misplaced_in_optional_list(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage(
            criteria_config=make_criteria_config({"project_goal": "blocking"}),
            templates={"project_goal": "What is the main goal?"},
        )

        result = stage.generate(
            make_completeness_result(
                [],
                optional_missing_information=[
                    make_missing_item("project_goal", "Project goal")
                ],
            ),
            arbitration_result=make_arbitration_result(
                DecisionStatus.accept_with_clarifications
            ),
        )

        self.assertEqual(result.questions, [])

    def test_clarify_generates_only_blocking_role_question(self) -> None:
        stage = TemplateQuestionGeneratorStage(
            criteria_config=make_criteria_config({"project_goal": "blocking"}),
            templates={"project_goal": "What is the main goal?"},
        )

        result = stage.generate(
            make_completeness_result(
                [make_missing_item("project_goal", "Project goal")]
            ),
            arbitration_result=make_arbitration_result(DecisionStatus.clarify),
        )

        self.assertEqual(
            [question.related_field for question in result.questions],
            ["project_goal"],
        )

    def test_accept_with_clarifications_generates_only_optional_role_question(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage(
            criteria_config=make_criteria_config({"integrations": "optional"}),
            templates={"integrations": "Are integrations needed?"},
        )

        result = stage.generate(
            make_completeness_result(
                [],
                optional_missing_information=[
                    make_missing_item("integrations", "Integrations")
                ],
            ),
            arbitration_result=make_arbitration_result(
                DecisionStatus.accept_with_clarifications
            ),
        )

        self.assertEqual(
            [question.related_field for question in result.questions],
            ["integrations"],
        )

    def test_internal_field_generates_no_question_for_clarify_or_accept_with_clarifications(
        self,
    ) -> None:
        stage = TemplateQuestionGeneratorStage(
            criteria_config=make_criteria_config({"internal_reference": "internal"}),
            templates={"internal_reference": "What is the internal reference?"},
        )

        clarify_result = stage.generate(
            make_completeness_result(
                [make_missing_item("internal_reference", "Internal reference")]
            ),
            arbitration_result=make_arbitration_result(DecisionStatus.clarify),
        )
        accept_with_clarifications_result = stage.generate(
            make_completeness_result(
                [],
                optional_missing_information=[
                    make_missing_item("internal_reference", "Internal reference")
                ],
            ),
            arbitration_result=make_arbitration_result(
                DecisionStatus.accept_with_clarifications
            ),
        )

        self.assertEqual(clarify_result.questions, [])
        self.assertEqual(accept_with_clarifications_result.questions, [])

    def test_missing_template_fields_are_reported_without_llm_fallback(self) -> None:
        stage = TemplateQuestionGeneratorStage(
            templates={"project_goal": "What is the main goal?"},
        )

        result = stage.generate(
            make_completeness_result(
                [
                    make_missing_item("project_goal", "Project goal"),
                    make_missing_item("tasks", "Tasks"),
                ]
            )
        )

        self.assertEqual(result.technical_info.attempts, 0)
        self.assertFalse(result.technical_info.llm_invoked)
        self.assertEqual(len(result.questions), 1)
        self.assertEqual(result.technical_info.missing_template_fields, ["tasks"])

    def test_includes_assessment_status_in_summary(self) -> None:
        stage = TemplateQuestionGeneratorStage(
            templates={"project_goal": "What is the project goal?"}
        )

        result = stage.generate(
            make_completeness_result([make_missing_item("project_goal", "Project goal")]),
            assessment_recommendation=AssessmentRecommendation.needs_clarification,
        )

        self.assertIn("needs_clarification", result.summary)

    def test_template_stage_updates_context_via_base_stage(self) -> None:
        stage = TemplateQuestionGeneratorStage(
            templates={"project_goal": "Опишите основную цель проекта."}
        )
        context = AIContext.from_brief(
            BriefInputFactory().from_text("Need a product")
        ).with_completeness_result(
            make_completeness_result(
                [make_missing_item("project_goal", "Project goal")]
            )
        )

        updated = stage.run(context)

        self.assertIsInstance(stage, BaseStage)
        self.assertIsNone(context.clarification_result)
        self.assertEqual(len(updated.clarification_result.questions), 1)

    def test_run_context_requires_completeness_result(self) -> None:
        stage = TemplateQuestionGeneratorStage()
        context = AIContext.from_brief(BriefInputFactory().from_text("Need a product"))

        with self.assertRaises(QuestionGenerationError):
            stage.run_context(context)


if __name__ == "__main__":
    unittest.main()
