"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

import re
import unittest

from pydantic import ValidationError

from app.config import CriteriaLoader
from app.input import BriefInputFactory
from app.pipeline import (
    BriefAnalysisResultBuilder,
    BriefAnalysisResultError,
    ResponseWriterStage,
)
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
    BriefAnalysisResult,
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


TOP_LEVEL_KEYS = {
    "summary",
    "extracted_fields",
    "assessment",
    "clarifying_questions",
    "mvp_suggestion",
    "customer_response_draft",
}
EXTRACTED_FIELD_KEYS = {
    "goal",
    "expected_result",
    "tasks",
    "domain",
    "direction",
    "available_materials",
    "missing_information",
    "complexity_factors",
}
ASSESSMENT_KEYS = {"recommendation", "confidence", "reasons", "risks"}


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


def context_with_direction_inputs(
    *,
    project_direction: str,
    project_type: str = "education",
    project_goal: str = "Сделать сайт для образовательного проекта",
    tasks: list[str] | None = None,
    expected_result: str = "сайт",
    with_config: bool = False,
) -> AIContext:
    """Собирает контекст с разными входами для публичной классификации direction."""
    context = make_context()
    assert context.extracted_brief is not None
    extracted = context.extracted_brief.model_copy(
        update={
            "project_goal": ExtractedFact(
                status=FactStatus.explicit,
                value=project_goal,
            ),
            "tasks": [
                ExtractedFact(status=FactStatus.explicit, value=task)
                for task in (tasks or [])
            ],
            "project_type": ExtractedFact(
                status=FactStatus.explicit,
                value=project_type,
            ),
            "project_direction": ExtractedFact(
                status=FactStatus.explicit,
                value=project_direction,
            ),
            "expected_result": ExtractedFact(
                status=FactStatus.explicit,
                value=expected_result,
            ),
        }
    )
    updated = context.with_extracted_brief(extracted)
    if not with_config:
        return updated

    return updated.model_copy(
        update={
            "technical": updated.technical.model_copy(
                update={"configuration": CriteriaLoader.load()}
            )
        }
    )


def public_direction(context: AIContext) -> str:
    """Возвращает direction из public JSON payload."""
    updated = ResponseWriterStage().run_context(context)
    return updated.final_response_payload["extracted_fields"]["direction"]


def make_public_payload() -> dict:
    """Возвращает минимальный валидный public JSON payload для contract-тестов."""
    return {
        "summary": "Project summary",
        "extracted_fields": {
            "goal": "Build a project",
            "expected_result": "Working result",
            "tasks": ["Task"],
            "domain": "education",
            "direction": "development",
            "available_materials": ["Content"],
            "missing_information": [],
            "complexity_factors": [],
        },
        "assessment": {
            "recommendation": "accept",
            "confidence": "high",
            "reasons": ["Criterion explanation"],
            "risks": [],
        },
        "clarifying_questions": [],
        "mvp_suggestion": "",
        "customer_response_draft": "Customer response",
    }


def assert_no_none(value) -> None:
    """Проверяет public JSON рекурсивно: None/null в нём быть не должно."""
    if isinstance(value, dict):
        for nested in value.values():
            assert_no_none(nested)
        return
    if isinstance(value, list):
        for nested in value:
            assert_no_none(nested)
        return
    if value is None:
        raise AssertionError("public JSON must not contain None")


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


class TestPublicJsonContract(unittest.TestCase):
    """Проверяет неизменный публичный JSON-контракт финального результата."""

    def test_public_payload_has_exact_key_sets(self) -> None:
        payload = BriefAnalysisResult.model_validate(
            make_public_payload()
        ).model_dump(mode="json")

        self.assertEqual(set(payload.keys()), TOP_LEVEL_KEYS)
        self.assertEqual(set(payload["extracted_fields"].keys()), EXTRACTED_FIELD_KEYS)
        self.assertEqual(set(payload["assessment"].keys()), ASSESSMENT_KEYS)

    def test_serialized_public_payload_contains_no_none_values(self) -> None:
        payload = BriefAnalysisResult.model_validate(
            make_public_payload()
        ).model_dump(mode="json")

        assert_no_none(payload)

    def test_public_schema_rejects_none_values(self) -> None:
        cases = []
        payload = make_public_payload()
        payload["summary"] = None
        cases.append(payload)

        payload = make_public_payload()
        payload["customer_response_draft"] = None
        cases.append(payload)

        payload = make_public_payload()
        payload["extracted_fields"]["goal"] = None
        cases.append(payload)

        payload = make_public_payload()
        payload["extracted_fields"]["tasks"] = None
        cases.append(payload)

        payload = make_public_payload()
        payload["assessment"]["reasons"] = None
        cases.append(payload)

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    BriefAnalysisResult.model_validate(payload)

    def test_empty_public_strings_and_lists_are_preserved(self) -> None:
        payload_data = make_public_payload()
        payload_data["summary"] = ""
        payload_data["customer_response_draft"] = ""
        payload_data["extracted_fields"].update(
            {
                "goal": "",
                "expected_result": "",
                "tasks": [],
                "domain": "",
                "available_materials": [],
                "missing_information": [],
                "complexity_factors": [],
            }
        )
        payload_data["assessment"].update(
            {
                "reasons": [],
                "risks": [],
            }
        )
        payload_data["clarifying_questions"] = []
        payload_data["mvp_suggestion"] = ""

        payload = BriefAnalysisResult.model_validate(payload_data).model_dump(
            mode="json"
        )

        self.assertEqual(payload["summary"], "")
        self.assertEqual(payload["customer_response_draft"], "")
        self.assertEqual(payload["extracted_fields"]["goal"], "")
        self.assertEqual(payload["extracted_fields"]["expected_result"], "")
        self.assertEqual(payload["extracted_fields"]["domain"], "")
        self.assertEqual(payload["extracted_fields"]["tasks"], [])
        self.assertEqual(payload["extracted_fields"]["available_materials"], [])
        self.assertEqual(payload["extracted_fields"]["missing_information"], [])
        self.assertEqual(payload["extracted_fields"]["complexity_factors"], [])
        self.assertEqual(payload["assessment"]["reasons"], [])
        self.assertEqual(payload["assessment"]["risks"], [])
        self.assertEqual(payload["clarifying_questions"], [])
        self.assertEqual(payload["mvp_suggestion"], "")

    def test_builder_uses_empty_customer_response_when_text_is_absent(self) -> None:
        payload = BriefAnalysisResultBuilder().build(make_context()).model_dump(
            mode="json"
        )

        self.assertEqual(payload["customer_response_draft"], "")

    def test_public_schemas_reject_extra_fields(self) -> None:
        cases = []
        payload = make_public_payload()
        payload["extra"] = "unexpected"
        cases.append(payload)

        payload = make_public_payload()
        payload["extracted_fields"]["extra"] = "unexpected"
        cases.append(payload)

        payload = make_public_payload()
        payload["assessment"]["extra"] = "unexpected"
        cases.append(payload)

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    BriefAnalysisResult.model_validate(payload)

    def test_public_schema_rejects_invalid_direction(self) -> None:
        payload = make_public_payload()
        payload["extracted_fields"]["direction"] = "unknown"

        with self.assertRaises(ValidationError):
            BriefAnalysisResult.model_validate(payload)

    def test_public_schema_rejects_invalid_recommendation(self) -> None:
        payload = make_public_payload()
        payload["assessment"]["recommendation"] = "unknown"

        with self.assertRaises(ValidationError):
            BriefAnalysisResult.model_validate(payload)

    def test_public_schema_accepts_accept_with_clarifications_recommendation(self) -> None:
        payload = make_public_payload()
        payload["assessment"]["recommendation"] = "accept_with_clarifications"
        payload["clarifying_questions"] = ["Какие материалы уже есть?"]

        result = BriefAnalysisResult.model_validate(payload)

        self.assertEqual(
            result.assessment.recommendation,
            "accept_with_clarifications",
        )

    def test_public_schema_rejects_invalid_confidence(self) -> None:
        payload = make_public_payload()
        payload["assessment"]["confidence"] = "certain"

        with self.assertRaises(ValidationError):
            BriefAnalysisResult.model_validate(payload)


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

    def test_public_direction_accepts_canonical_development(self) -> None:
        self.assertEqual(public_direction(make_context()), "development")

    def test_public_direction_uses_config_aliases(self) -> None:
        context = context_with_direction_inputs(
            project_direction="разработка",
            with_config=True,
        )

        self.assertEqual(public_direction(context), "development")

    def test_public_direction_uses_project_direction_keywords_for_development(self) -> None:
        context = context_with_direction_inputs(
            project_direction="разработка веб-сервиса",
            project_type="operations",
            project_goal="Clarify stakeholder expectations",
            tasks=["Collect context"],
            expected_result="Shared understanding",
        )

        self.assertEqual(public_direction(context), "development")

    def test_public_direction_uses_project_direction_keywords_for_design(self) -> None:
        context = context_with_direction_inputs(
            project_direction="UX редизайн интерфейса",
            project_type="operations",
            project_goal="Clarify stakeholder expectations",
            tasks=["Collect context"],
            expected_result="Shared understanding",
        )

        self.assertEqual(public_direction(context), "design")

    def test_public_direction_uses_project_direction_keywords_for_ai(self) -> None:
        context = context_with_direction_inputs(
            project_direction="LLM ассистент",
            project_type="operations",
            project_goal="Clarify stakeholder expectations",
            tasks=["Collect context"],
            expected_result="Shared understanding",
        )

        self.assertEqual(public_direction(context), "ai")

    def test_public_direction_classifies_web_service_as_development(self) -> None:
        context = context_with_direction_inputs(
            project_direction="support automation",
            project_type="software",
            project_goal="Build a web service for support automation",
            tasks=["Create backend API"],
            expected_result="Working service",
        )

        self.assertEqual(public_direction(context), "development")

    def test_public_direction_classifies_mobile_app_as_development(self) -> None:
        context = context_with_direction_inputs(
            project_direction="customer product",
            project_type="software",
            project_goal="Build a mobile application for clients",
            tasks=["Create mobile app screens"],
            expected_result="Working app",
        )

        self.assertEqual(public_direction(context), "development")

    def test_public_direction_classifies_ux_ui_redesign_as_design(self) -> None:
        context = context_with_direction_inputs(
            project_direction="customer experience",
            project_type="product",
            project_goal="UX UI redesign of checkout interface",
            tasks=["Prepare mockup"],
            expected_result="Updated interface",
        )

        self.assertEqual(public_direction(context), "design")

    def test_public_direction_classifies_data_research_as_analytics(self) -> None:
        context = context_with_direction_inputs(
            project_direction="business study",
            project_type="business",
            project_goal="Run data research and analytics",
            tasks=["Build dashboard with metrics"],
            expected_result="Research conclusions",
        )

        self.assertEqual(public_direction(context), "analytics")

    def test_public_direction_classifies_smm_promotion_as_marketing(self) -> None:
        context = context_with_direction_inputs(
            project_direction="communications",
            project_type="brand",
            project_goal="Prepare SMM promotion campaign",
            tasks=["Plan advertising"],
            expected_result="Marketing plan",
        )

        self.assertEqual(public_direction(context), "marketing")

    def test_public_direction_classifies_llm_ml_project_as_ai(self) -> None:
        context = context_with_direction_inputs(
            project_direction="assistant",
            project_type="software",
            project_goal="Build an LLM bot with ML classification",
            tasks=["Create API integration"],
            expected_result="AI assistant prototype",
        )

        self.assertEqual(public_direction(context), "ai")

    def test_public_direction_classifies_course_methodology_as_education(self) -> None:
        context = context_with_direction_inputs(
            project_direction="learning product",
            project_type="education",
            project_goal="Create course methodology",
            tasks=["Prepare learning materials"],
            expected_result="Course program",
        )

        self.assertEqual(public_direction(context), "education")

    def test_public_direction_accepts_explicit_mixed(self) -> None:
        context = context_with_direction_inputs(project_direction="mixed")

        self.assertEqual(public_direction(context), "mixed")

    def test_public_direction_prefers_development_for_education_website(self) -> None:
        context = context_with_direction_inputs(
            project_direction="learning portal",
            project_type="education",
            project_goal="Сделать сайт для образовательного проекта",
            tasks=["Разработать портал"],
            expected_result="Рабочий сайт",
        )

        self.assertEqual(public_direction(context), "development")

    def test_public_direction_unknown_value_raises_clear_error(self) -> None:
        context = context_with_direction_inputs(
            project_direction="internal alignment",
            project_type="operations",
            project_goal="Clarify stakeholder expectations",
            tasks=["Collect context"],
            expected_result="Shared understanding",
        )

        with self.assertRaisesRegex(
            BriefAnalysisResultError,
            "Unable to classify public project direction",
        ):
            public_direction(context)

    def test_missing_public_string_fields_are_empty_strings(self) -> None:
        context = make_context()
        assert context.extracted_brief is not None
        extracted = context.extracted_brief.model_copy(
            update={
                "project_goal": ExtractedFact(status=FactStatus.missing, value=None),
                "expected_result": ExtractedFact(status=FactStatus.missing, value=None),
                "project_type": ExtractedFact(status=FactStatus.missing, value=None),
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
            "development",
        )

    def test_missing_public_direction_without_signals_raises_clear_error(self) -> None:
        context = make_context()
        assert context.extracted_brief is not None
        extracted = context.extracted_brief.model_copy(
            update={
                "project_goal": ExtractedFact(status=FactStatus.missing, value=None),
                "tasks": [],
                "project_type": ExtractedFact(status=FactStatus.missing, value=None),
                "project_direction": ExtractedFact(status=FactStatus.missing, value=None),
                "expected_result": ExtractedFact(status=FactStatus.missing, value=None),
            }
        )

        with self.assertRaisesRegex(
            BriefAnalysisResultError,
            "Unable to classify public project direction",
        ):
            ResponseWriterStage().run_context(context.with_extracted_brief(extracted))

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

    def test_accept_with_clarifications_response_includes_questions(self) -> None:
        updated = ResponseWriterStage().run_context(
            make_context(DecisionStatus.accept_with_clarifications)
        )

        self.assertIn("Какие материалы уже есть?", updated.final_response_text)
        self.assertEqual(
            updated.final_response_payload["assessment"]["recommendation"],
            "accept_with_clarifications",
        )
        self.assertEqual(
            updated.final_response_payload["clarifying_questions"],
            ["Какие материалы уже есть?"],
        )

    def test_accept_with_clarifications_public_payload_requires_questions(self) -> None:
        context = make_context(
            DecisionStatus.accept_with_clarifications
        ).with_clarification_result(make_empty_question_result())

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

    def test_out_of_scope_reject_letter_explains_the_ground(self) -> None:
        # Заявка не про формат Мастерской: заказчик должен увидеть причину
        # отказа, а не уточняющие вопросы.
        base = make_reject_context()
        assessment = base.assessment_result
        assert assessment is not None
        context = base.with_assessment_result(
            assessment.model_copy(
                update={
                    "criterion_evaluations": [
                        CriterionEvaluation(
                            criterion="request_eligibility",
                            criterion_title="Request eligibility",
                            status=CriterionEvaluationStatus.not_met,
                            explanation=(
                                "Заявка предлагает сотрудничество, а не задачу "
                                "с цифровым результатом для заказчика."
                            ),
                        ),
                    ],
                    "risks": [
                        Risk(
                            type="out_of_scope_request",
                            description=(
                                "Бриф не содержит проектной задачи для команды "
                                "выпускников."
                            ),
                            severity=RiskSeverity.critical,
                        )
                    ],
                }
            )
        )

        updated = ResponseWriterStage().run_context(context)

        self.assertIn("Основания оценки:", updated.final_response_text)
        self.assertIn(
            "Заявка предлагает сотрудничество",
            updated.final_response_text,
        )
        self.assertIn(
            "Бриф не содержит проектной задачи",
            updated.final_response_text,
        )
        self.assertNotIn("Вопросы", updated.final_response_text)

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
            DecisionStatus.accept_with_clarifications: make_context(
                DecisionStatus.accept_with_clarifications
            ),
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
