"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import unittest

from app.input import BriefInputFactory
from app.pipeline import BaseStage, QuestionGenerationError, TemplateQuestionGeneratorStage
from app.schemas import (
    AIContext,
    AssessmentRecommendation,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
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


def make_completeness_result(
    missing_information: list[CompletenessItem],
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
        clarification_information=[],
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
