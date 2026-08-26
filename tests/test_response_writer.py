"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import re
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
    """Выполняет шаг «make context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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


ALLOWED_LATIN_WORDS = ("MVP",)


def find_latin_words(text: str) -> list[str]:
    """Ищет в тексте латинские слова. Нужна аббревиатура MVP, все остальное в письме заказчику - признак утечки внутренних строк."""
    cleaned = text
    for word in ALLOWED_LATIN_WORDS:
        cleaned = cleaned.replace(word, " ")
    return re.findall(r"[A-Za-z]{2,}", cleaned)


def make_reject_context(
    risk_severity: RiskSeverity = RiskSeverity.critical,
) -> AIContext:
    """Собирает контекст отказа: один выполненный критерий, один проваленный и риск. Так видно, что в письмо попадают только основания самого отказа."""
    context = make_context(DecisionStatus.reject)
    assessment = context.assessment_result
    assert assessment is not None

    return context.with_assessment_result(
        assessment.model_copy(
            update={
                "criterion_evaluations": [
                    CriterionEvaluation(
                        criterion="goal_clarity",
                        criterion_title="Goal clarity",
                        status=CriterionEvaluationStatus.met,
                        explanation="Цель описана чётко.",
                    ),
                    CriterionEvaluation(
                        criterion="student_fit",
                        criterion_title="Student project fit",
                        status=CriterionEvaluationStatus.not_met,
                        explanation=(
                            "Промышленная эксплуатация несовместима с учебным форматом."
                        ),
                    ),
                ],
                "risks": [
                    Risk(
                        type="production_criticality",
                        description="Заказчик ожидает промышленной надёжности.",
                        severity=risk_severity,
                    )
                ],
                "has_risks": True,
            }
        )
    )


def make_empty_question_result() -> QuestionGenerationResult:
    """Выполняет шаг «make empty question result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
    """Выполняет шаг «make mvp planning result». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return MVPPlanningResult(
        plan=MVPPlan(
            core_goal="Сайт образовательного проекта с описанием курсов",
            keep=["Главная страница с описанием проекта"],
            remove=["Личный кабинет с аналитикой посещений"],
            simplify=["Взять готовый шаблон вместо индивидуального дизайна"],
            mvp_scope=["Главная страница", "Форма обратной связи"],
            rationale=["Первая версия должна умещаться в один учебный семестр"],
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
    """Класс «TestResponseWriterStage» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_writes_response_and_public_payload(self) -> None:
        updated = ResponseWriterStage().run_context(make_context())

        self.assertIsNotNone(updated.final_response_text)
        self.assertIsNotNone(updated.final_response_payload)
        self.assertEqual(updated.final_response_payload["assessment"]["recommendation"], "accept")
        self.assertEqual(
            updated.final_response_payload["extracted_fields"]["direction"],
            "development",
        )

    def test_missing_public_string_fields_are_empty_strings(self) -> None:
        context = make_context()
        assert context.extracted_brief is not None
        extracted = context.extracted_brief.model_copy(
            update={
                "project_goal": ExtractedFact(status=FactStatus.missing, value=None),
                "expected_result": ExtractedFact(status=FactStatus.missing, value=None),
                "project_type": ExtractedFact(status=FactStatus.missing, value=None),
                "project_direction": ExtractedFact(status=FactStatus.missing, value=None),
            }
        )

        updated = ResponseWriterStage().run_context(context.with_extracted_brief(extracted))

        self.assertEqual(updated.final_response_payload["extracted_fields"]["goal"], "")
        self.assertEqual(
            updated.final_response_payload["extracted_fields"]["expected_result"],
            "",
        )
        self.assertEqual(updated.final_response_payload["extracted_fields"]["domain"], "")
        self.assertEqual(
            updated.final_response_payload["extracted_fields"]["direction"],
            "",
        )

    def test_public_reasons_exclude_arbitration_diagnostics(self) -> None:
        context = make_context().with_arbitration_result(
            ArbitrationResult(
                final_status=DecisionStatus.accept,
                reasons=[
                    "Matched conditions: risk.max_severity in ['high']",
                    "Signals: risk.max_severity='high'",
                    "No arbitration rule matched; default status selected from configuration.",
                ],
                evidence=[],
                triggered_rules=[],
                confidence=0.9,
                metadata={},
            )
        )

        updated = ResponseWriterStage().run_context(context)

        self.assertEqual(
            updated.final_response_payload["assessment"]["reasons"],
            ["Цель описана."],
        )
        serialized_reasons = "\n".join(
            updated.final_response_payload["assessment"]["reasons"]
        )
        self.assertNotIn("Matched conditions:", serialized_reasons)
        self.assertNotIn("Signals:", serialized_reasons)
        self.assertNotIn("No arbitration rule matched", serialized_reasons)

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
            "Сайт образовательного проекта с описанием курсов",
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


class TestResponseWriterReasons(unittest.TestCase):
    """Проверяет блок «Основания оценки»: он должен объяснять вердикт словами оценки брифа, а не техническими строками арбитра."""

    def test_accept_reasons_use_criterion_explanations(self) -> None:
        updated = ResponseWriterStage().run_context(make_context())

        self.assertIn("Основания оценки:", updated.final_response_text)
        self.assertIn("Цель описана.", updated.final_response_text)

    def test_reasons_ignore_arbitration_reasons(self) -> None:
        # В фикстуре арбитр отдает англоязычное описание правила из criteria.yaml.
        context = make_context().with_arbitration_result(
            ArbitrationResult(
                final_status=DecisionStatus.accept,
                reasons=[
                    "Accept when the brief is complete, criteria are met "
                    "and no relevant risks remain.",
                    "Matched conditions: completeness.is_complete eq True",
                    "Signals: completeness.is_complete=True",
                ],
                confidence=0.9,
            )
        )

        updated = ResponseWriterStage().run_context(context)

        self.assertNotIn("Accept when the brief", updated.final_response_text)
        self.assertNotIn("Matched conditions", updated.final_response_text)
        self.assertNotIn("Signals", updated.final_response_text)

    def test_reject_reasons_show_problems_instead_of_met_criteria(self) -> None:
        context = make_reject_context()

        updated = ResponseWriterStage().run_context(context)

        self.assertNotIn("Цель описана чётко.", updated.final_response_text)
        self.assertIn(
            "Промышленная эксплуатация несовместима с учебным форматом.",
            updated.final_response_text,
        )
        self.assertIn(
            "Заказчик ожидает промышленной надёжности.",
            updated.final_response_text,
        )

    def test_reasons_block_is_omitted_when_nothing_explains_verdict(self) -> None:
        # Единственный критерий выполнен, рисков нет - обосновывать отказ нечем.
        updated = ResponseWriterStage().run_context(
            make_context(DecisionStatus.reject)
        )

        self.assertNotIn("Основания оценки:", updated.final_response_text)

    def test_low_severity_risks_are_not_reported_to_customer(self) -> None:
        context = make_reject_context(risk_severity=RiskSeverity.low)

        updated = ResponseWriterStage().run_context(context)

        self.assertNotIn(
            "Заказчик ожидает промышленной надёжности.",
            updated.final_response_text,
        )

    def test_missing_information_uses_russian_field_titles(self) -> None:
        updated = ResponseWriterStage().run_context(
            make_context(DecisionStatus.clarify)
        )

        self.assertEqual(
            updated.final_response_payload["extracted_fields"]["missing_information"],
            ["Доступные материалы"],
        )

    def test_customer_response_draft_has_no_latin_words(self) -> None:
        cases = {
            DecisionStatus.accept: make_context(),
            DecisionStatus.clarify: make_context(DecisionStatus.clarify),
            DecisionStatus.mentor_review: make_context(DecisionStatus.mentor_review),
            DecisionStatus.reject: make_reject_context(),
            DecisionStatus.simplify: make_context(
                DecisionStatus.simplify
            ).with_mvp_planning_result(make_mvp_planning_result()),
        }

        for status, context in cases.items():
            with self.subTest(status=status):
                updated = ResponseWriterStage().run_context(context)

                self.assertEqual(
                    find_latin_words(updated.final_response_text),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
