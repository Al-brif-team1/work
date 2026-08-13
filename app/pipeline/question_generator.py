"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

import json
from pathlib import Path

from app.pipeline.contracts import BaseStage
from app.schemas import (
    AIContext,
    AssessmentRecommendation,
    ClarificationQuestion,
    CompletenessResult,
    QuestionGenerationResult,
    QuestionGenerationTechnicalInfo,
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
        templates: dict[str, str] | None = None,
        templates_path: str | Path | None = None,
        tracing_client: TracingClient | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or NoOpTracingClient(),
        )
        if templates is not None and templates_path is not None:
            raise ValueError("Pass either templates or templates_path, not both")
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
    ) -> QuestionGenerationResult:
        """Выполняет шаг «generate». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        questions: list[ClarificationQuestion] = []
        missing_template_fields: list[str] = []

        for index, item in enumerate(completeness_result.missing_information, start=1):
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
