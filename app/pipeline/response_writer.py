"""Deterministic customer response drafting."""

from __future__ import annotations

from typing import Any

from app.pipeline.contracts import BaseStage
from app.pipeline.result_builder import BriefAnalysisResultBuilder
from app.schemas import AIContext, DecisionStatus
from app.tracing.tracing import NoOpTracingClient, TracingClient


class ResponseWriterError(RuntimeError):
    """Raised when a customer response draft cannot be generated."""


class ResponseWriterStage(BaseStage[AIContext, AIContext]):
    """Generate a concise customer response draft from completed analysis."""

    def __init__(
        self,
        *,
        result_builder: BriefAnalysisResultBuilder | None = None,
        tracing_client: TracingClient | None = None,
    ) -> None:
        """Initialize the deterministic response writer."""
        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or NoOpTracingClient(),
        )
        self._result_builder = result_builder or BriefAnalysisResultBuilder()

    def write_context(self, context: AIContext) -> AIContext:
        """Return context enriched with final response text and public payload."""
        response_text = self._build_response_text(context)
        context_with_text = context.with_final_response(response_text)
        payload = self._result_builder.build(context_with_text)
        return context_with_text.with_final_response(
            response_text,
            response_payload=payload.model_dump(mode="json"),
        )

    def run_context(self, context: AIContext) -> AIContext:
        """Run this stage using the common AIContext pipeline contract."""
        return self.run(context)

    def _run(self, stage_input: AIContext) -> AIContext:
        """Run deterministic response writing."""
        return self.write_context(stage_input)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Preserve response-writer-specific errors."""
        return exc

    def _build_trace_input(self, stage_input: AIContext) -> dict[str, Any]:
        """Build safe trace metadata for response drafting."""
        return {
            "has_arbitration_result": stage_input.arbitration_result is not None,
            "has_questions": bool(
                stage_input.clarification_result
                and stage_input.clarification_result.questions
            ),
            "has_mvp_plan": bool(
                stage_input.mvp_planning_result
                and stage_input.mvp_planning_result.plan
            ),
        }

    def _build_trace_output(self, stage_output: AIContext) -> dict[str, Any]:
        """Build safe trace output for response drafting."""
        return {
            "status": "success",
            "response_length": len(stage_output.final_response_text or ""),
        }

    def _build_response_text(self, context: AIContext) -> str:
        """Build a customer-facing draft without changing business decisions."""
        if context.arbitration_result is None:
            raise ResponseWriterError("Response writer requires arbitration_result")

        status = context.arbitration_result.final_status
        if status is DecisionStatus.accept:
            return self._accept_response(context)
        if status is DecisionStatus.clarify:
            return self._clarify_response(context)
        if status is DecisionStatus.simplify:
            return self._simplify_response(context)
        if status is DecisionStatus.mentor_review:
            return self._mentor_review_response(context)
        if status is DecisionStatus.reject:
            return self._reject_response(context)

        raise ResponseWriterError(f"Unsupported arbitration status: {status}")

    def _accept_response(self, context: AIContext) -> str:
        """Build a draft for an acceptable project."""
        summary = self._summary(context)
        reasons = self._format_reasons(context)
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. По предварительной оценке проект можно брать "
            "в работу как студенческий проект.\n\n"
            f"Кратко о проекте: {summary}\n\n"
            f"{reasons}"
            "Следующий шаг: согласовать состав работ, сроки и формат материалов "
            "для старта команды."
        )

    def _clarify_response(self, context: AIContext) -> str:
        """Build a draft for a project that needs clarification."""
        questions = self._format_questions(context)
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. Сейчас по заявке не хватает данных для корректной "
            "оценки студенческого проекта.\n\n"
            f"{questions}\n\n"
            "После уточнений мы сможем повторно оценить реалистичность проекта, "
            "риски и формат участия студентов."
        )

    def _simplify_response(self, context: AIContext) -> str:
        """Build a draft for a project that needs an MVP scope."""
        mvp = self._format_mvp(context)
        questions = self._format_questions(context, optional=True)
        question_block = f"\n\nДополнительно нужно уточнить:\n{questions}" if questions else ""
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. Идея потенциально подходит для студенческого "
            "проекта, но текущий объём выглядит слишком широким для первой "
            f"версии.\n\n{mvp}{question_block}\n\n"
            "Предлагаем обсудить такой сокращённый формат и отдельно вынести "
            "промышленные или сильно рискованные задачи за рамки MVP."
        )

    def _mentor_review_response(self, context: AIContext) -> str:
        """Build a draft for a project that requires mentor expertise."""
        reasons = self._format_reasons(context)
        questions = self._format_questions(context, optional=True)
        question_block = f"\n\nВопросы для уточнения:\n{questions}" if questions else ""
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. По предварительной оценке заявка требует "
            "дополнительной экспертизы наставника перед финальным решением.\n\n"
            f"{reasons}{question_block}\n\n"
            "Мы вернёмся с рекомендацией после экспертной проверки состава работ "
            "и рисков."
        )

    def _reject_response(self, context: AIContext) -> str:
        """Build a draft for a project that does not fit the program."""
        reasons = self._format_reasons(context)
        return (
            "Здравствуйте!\n\n"
            "Спасибо за бриф. По предварительной оценке проект в текущем виде "
            "не подходит для формата студенческой работы.\n\n"
            f"{reasons}"
            "Если вы готовы существенно изменить постановку задачи, можно "
            "подготовить новый бриф с более ограниченным и учебно реализуемым "
            "объёмом."
        )

    def _summary(self, context: AIContext) -> str:
        """Return the best available project summary."""
        if context.assessment_result and context.assessment_result.summary:
            return context.assessment_result.summary
        if context.extracted_brief and context.extracted_brief.project_goal.value:
            return context.extracted_brief.project_goal.value
        return "описание требует дополнительного уточнения"

    def _format_reasons(self, context: AIContext) -> str:
        """Format arbitration reasons into a readable block."""
        reasons = (
            context.arbitration_result.reasons
            if context.arbitration_result is not None
            else []
        )
        if not reasons:
            return ""
        return "Основания оценки:\n" + "\n".join(f"- {item}" for item in reasons) + "\n\n"

    def _format_questions(self, context: AIContext, *, optional: bool = False) -> str:
        """Format clarification questions if they were generated."""
        questions = (
            context.clarification_result.questions
            if context.clarification_result is not None
            else []
        )
        if not questions:
            return "" if optional else "Уточняющие вопросы пока не сформированы."
        return "\n".join(f"- {item.question}" for item in questions)

    def _format_mvp(self, context: AIContext) -> str:
        """Format MVP plan if available."""
        planning = context.mvp_planning_result
        if planning is None or planning.plan is None:
            return (
                "MVP-предложение пока не сформировано: требуется дополнительная "
                "проработка объёма."
            )

        plan = planning.plan
        parts = [f"Предлагаемый MVP: {plan.core_goal}"]
        if plan.keep:
            parts.append("Оставить в первой версии: " + "; ".join(plan.keep))
        if plan.simplify:
            parts.append("Упростить: " + "; ".join(plan.simplify))
        if plan.remove:
            parts.append("Исключить из первой версии: " + "; ".join(plan.remove))
        return "\n".join(parts)
