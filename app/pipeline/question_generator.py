"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import (
    CriteriaConfig,
    CriteriaConfigError,
    CriteriaLoader,
    get_criteria_config,
)
from app.pipeline.contracts import BaseStage
from app.schemas import (
    AIContext,
    AssessmentRecommendation,
    AssessmentResult,
    ArbitrationResult,
    ClarificationQuestion,
    CompletenessResult,
    CompletenessItem,
    DecisionStatus,
    QuestionGenerationResult,
    QuestionGenerationTechnicalInfo,
    TrafficLightStatus,
)
from app.tracing.tracing import NoOpTracingClient, TracingClient


class QuestionGenerationError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class QuestionGeneratorConfigError(QuestionGenerationError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class TemplateQuestionGeneratorStage(BaseStage[AIContext, AIContext]):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Робот этапа. Он выполняет участок конвейера «template question generator stage». Этот этап работает как детерминированный робот: обычный код, без творческих догадок ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseStage, поэтому он умеет работать в нашем конвейере."""

    def __init__(
        self,
        *,
        criteria_config: CriteriaConfig | None = None,
        criteria_path: str | Path | None = None,
        templates: dict[str, str] | None = None,
        templates_path: str | Path | None = None,
        tracing_client: TracingClient | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or NoOpTracingClient(),
        )
        if criteria_config is not None and criteria_path is not None:
            raise ValueError("Pass either criteria_config or criteria_path, not both")
        if templates is not None and templates_path is not None:
            raise ValueError("Pass either templates or templates_path, not both")

        if criteria_config is not None:
            config = criteria_config
        elif criteria_path is not None:
            try:
                config = CriteriaLoader.load(Path(criteria_path))
            except CriteriaConfigError as exc:
                raise QuestionGeneratorConfigError(str(exc)) from exc
        else:
            try:
                config = get_criteria_config()
            except CriteriaConfigError as exc:
                raise QuestionGeneratorConfigError(str(exc)) from exc

        self._customer_field_roles = {
            field_def.key: field_def.customer_field_role
            for field_def in config.evaluation.required_fields
        }
        self._templates = (
            dict(templates)
            if templates is not None
            else self._load_templates(templates_path or self._default_templates_path())
        )

    def generate(
        self,
        completeness_result: CompletenessResult,
        *,
        assessment_recommendation: AssessmentRecommendation | None = None,
        assessment_result: AssessmentResult | None = None,
        arbitration_result: ArbitrationResult | None = None,
    ) -> QuestionGenerationResult:
        """Выполняет шаг «generate». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        questions: list[ClarificationQuestion] = []
        missing_template_fields: list[str] = []

        question_items = self._select_question_items(
            completeness_result=completeness_result,
            arbitration_result=arbitration_result,
        )
        for index, item in enumerate(question_items, start=1):
            template = self._templates.get(item.field_key)
            if template is None:
                missing_template_fields.append(item.field_key)
                continue

            questions.append(
                ClarificationQuestion(
                    question=template,
                    related_field=item.field_key,
                    reason=item.reason or f"Missing required field: {item.title}",
                    priority=index,
                )
            )

        questions.extend(
            self._traffic_light_yellow_questions(
                assessment_result=assessment_result,
                arbitration_result=arbitration_result,
                existing_questions=questions,
            )
        )

        summary = self._build_summary(
            questions_count=len(questions),
            missing_template_fields=missing_template_fields,
            assessment_recommendation=assessment_recommendation,
        )
        return QuestionGenerationResult(
            questions=questions,
            summary=summary,
            technical_info=QuestionGenerationTechnicalInfo(
                llm_invoked=False,
                attempts=0,
                prompt_name=None,
                trace_enabled=not isinstance(self._tracing_client, NoOpTracingClient),
                trace_name="question_generator.template",
                model_name=None,
                question_count=len(questions),
                missing_template_fields=missing_template_fields,
                raw_response=None,
                recovered_errors=[],
            ),
        )

    def generate_context(self, context: AIContext) -> AIContext:
        """Выполняет шаг «generate context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if context.completeness_result is None:
            raise QuestionGenerationError(
                "Question generation requires completeness_result in AIContext"
            )

        result = self.generate(
            context.completeness_result,
            assessment_recommendation=(
                context.assessment_result.recommendation
                if context.assessment_result is not None
                else None
            ),
            assessment_result=context.assessment_result,
            arbitration_result=context.arbitration_result,
        )
        return context.with_clarification_result(result)

    def run_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
        return self.run(context)

    def _run(self, stage_input: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""
        return self.generate_context(stage_input)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return exc

    def _select_question_items(
        self,
        *,
        completeness_result: CompletenessResult,
        arbitration_result: ArbitrationResult | None,
    ) -> list[CompletenessItem]:
        """Select customer-facing completeness items that match the final decision."""
        expected_role: str | None = None
        if arbitration_result is None:
            expected_role = "blocking"
            items = [
                *completeness_result.missing_information,
                *completeness_result.clarification_information,
            ]
        elif arbitration_result.final_status is DecisionStatus.clarify:
            expected_role = "blocking"
            items = [
                *completeness_result.missing_information,
                *completeness_result.clarification_information,
            ]
        elif (
            arbitration_result.final_status
            is DecisionStatus.accept_with_clarifications
        ):
            expected_role = "optional"
            items = list(completeness_result.optional_missing_information)
        else:
            items = []

        return [
            item
            for item in items
            if self._customer_field_roles.get(item.field_key) == expected_role
        ]

    def _traffic_light_yellow_questions(
        self,
        *,
        assessment_result: AssessmentResult | None,
        arbitration_result: ArbitrationResult | None,
        existing_questions: list[ClarificationQuestion],
    ) -> list[ClarificationQuestion]:
        if (
            arbitration_result is None
            or arbitration_result.final_status
            is not DecisionStatus.accept_with_clarifications
            or assessment_result is None
            or assessment_result.traffic_light.status is not TrafficLightStatus.yellow
        ):
            return []

        result: list[ClarificationQuestion] = []
        seen_matches: set[tuple[str, str, str]] = set()
        seen_questions = {
            self._normalize_question_text(item.question)
            for item in existing_questions
        }
        for match in assessment_result.traffic_light.matches:
            if match.status is not TrafficLightStatus.yellow:
                continue

            key = (
                self._normalize_question_text(match.task),
                self._normalize_question_text(match.matched_rule),
                self._normalize_question_text(match.reason),
            )
            if key in seen_matches:
                continue
            seen_matches.add(key)

            question = self._build_traffic_light_question(match.task, match.reason)
            normalized_question = self._normalize_question_text(question)
            if normalized_question in seen_questions:
                continue
            seen_questions.add(normalized_question)

            result.append(
                ClarificationQuestion(
                    question=question,
                    related_field="traffic_light",
                    reason=(
                        f"Traffic Light yellow match: {match.matched_rule}. "
                        f"{match.reason}"
                    ),
                    priority=len(existing_questions) + len(result) + 1,
                )
            )
        return result

    @staticmethod
    def _build_traffic_light_question(task: str, reason: str) -> str:
        return (
            f'По задаче "{task}" нужно уточнить: {reason}. '
            "Как это должно быть учтено в первой версии проекта?"
        )

    @staticmethod
    def _normalize_question_text(value: str) -> str:
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _build_summary(
        *,
        questions_count: int,
        missing_template_fields: list[str],
        assessment_recommendation: AssessmentRecommendation | None,
    ) -> str:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        parts = [f"Generated {questions_count} clarification questions from templates."]
        if missing_template_fields:
            parts.append(
                "Missing templates for fields: "
                + ", ".join(sorted(missing_template_fields))
                + "."
            )
        if assessment_recommendation is not None:
            parts.append(f"Assessment recommendation: {assessment_recommendation.value}.")
        return " ".join(parts)

    @staticmethod
    def _load_templates(path: str | Path) -> dict[str, str]:
        """Выполняет шаг «load templates». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        template_path = Path(path)
        try:
            raw = json.loads(template_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QuestionGeneratorConfigError(
                f"Unable to load question templates: {template_path}"
            ) from exc

        if not isinstance(raw, dict):
            raise QuestionGeneratorConfigError("question templates must be a mapping")

        templates: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise QuestionGeneratorConfigError("question template keys must be strings")
            if not isinstance(value, str) or not value.strip():
                raise QuestionGeneratorConfigError(
                    f"question template for {key!r} must be a non-empty string"
                )
            templates[key.strip()] = value.strip()
        return templates

    @staticmethod
    def _default_templates_path() -> Path:
        """Возвращает значение по умолчанию, чтобы этап мог работать без ручной настройки."""
        return Path(__file__).resolve().parents[2] / "config" / "question_templates.json"
