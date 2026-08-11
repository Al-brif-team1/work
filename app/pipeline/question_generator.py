"""Deterministic clarification question generation from templates."""

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
    """Raised when clarification question generation fails."""


class QuestionGeneratorConfigError(QuestionGenerationError):
    """Raised when question template configuration is missing or invalid."""


class TemplateQuestionGeneratorStage(BaseStage[AIContext, AIContext]):
    """Generate clarification questions from deterministic field templates."""

    def __init__(
        self,
        *,
        templates: dict[str, str] | None = None,
        templates_path: str | Path | None = None,
        tracing_client: TracingClient | None = None,
    ) -> None:
        """Initialize the template-based generator."""
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
        """Generate questions for missing fields without invoking an LLM."""
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
        """Return context enriched with template-generated questions."""
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
        """Run this stage using the common AIContext pipeline contract."""
        return self.run(context)

    def _run(self, stage_input: AIContext) -> AIContext:
        """Run deterministic question generation."""
        return self.generate_context(stage_input)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Preserve question-generation-specific errors."""
        return exc

    @staticmethod
    def _build_summary(
        *,
        questions_count: int,
        missing_template_fields: list[str],
        assessment_recommendation: AssessmentRecommendation | None,
    ) -> str:
        """Build a compact deterministic generation summary."""
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
        """Load question templates from a JSON mapping."""
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
        """Return the default question-template configuration path."""
        return Path(__file__).resolve().parents[2] / "config" / "question_templates.json"
