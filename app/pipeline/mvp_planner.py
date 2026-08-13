"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.llm.runner import LLMRunner
from app.pipeline.base import BaseLLMStage
from app.prompts import PromptManager
from app.schemas import (
    AIContext,
    ArbitrationResult,
    AssessmentResult,
    BriefInput,
    DecisionStatus,
    ExtractedBrief,
    MVPPlan,
    MVPPlanningResult,
    MVPPlanningTechnicalInfo,
)
from app.tracing.tracing import TracingClient
from app.tracing.tracing import NoOpTracingClient

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class MVPPlannerError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class MVPPlannerStage(BaseLLMStage):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Архитектор-техлид. Он через ИИ упрощает проект до MVP только тогда, когда это действительно нужно. Этот этап обращается к LLM, поэтому внутри работает ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseLLMStage, поэтому он умеет работать в нашем конвейере."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tracing_client: TracingClient | None = None,
        prompt_path: str | Path | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        prompt_manager: PromptManager | None = None,
        max_retries: int = 2,
        model_name: str | None = None,
        llm_runner: LLMRunner | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        super().__init__(
            llm_client=llm_client,
            tracing_client=tracing_client,
            prompt_path=prompt_path,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_manager=prompt_manager,
            max_retries=max_retries,
            model_name=model_name,
            llm_runner=llm_runner,
        )

    def plan_assessment(
        self,
        brief_input: BriefInput,
        extracted_brief: ExtractedBrief,
        assessment_result: AssessmentResult,
        arbitration_result: ArbitrationResult,
    ) -> MVPPlanningResult:
        """Выполняет шаг «plan assessment». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if arbitration_result.final_status != DecisionStatus.simplify:
            return MVPPlanningResult(
                plan=None,
                technical_info=MVPPlanningTechnicalInfo(
                    llm_invoked=False,
                    attempts=0,
                    prompt_name=self.prompt_name,
                    trace_enabled=not isinstance(self._tracing_client, NoOpTracingClient),
                    trace_name="mvp_planner.brief",
                    model_name=self.model_name,
                    skipped_reason=(
                        "MVP planner runs only when arbitration status is SIMPLIFY"
                    ),
                    raw_response=None,
                    recovered_errors=[],
                ),
            )

        return self._run_plan(
            brief_input=brief_input,
            extracted_brief=extracted_brief,
            risk_analysis_result=self._risk_analysis_prompt_section(
                assessment_result
            ),
            evaluation_result=self._evaluation_prompt_section(assessment_result),
            arbitration_result=arbitration_result,
            risk_count=len(assessment_result.risks),
            criterion_count=len(assessment_result.criterion_evaluations),
        )

    def plan_context(self, context: AIContext) -> AIContext:
        """Выполняет шаг «plan context». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if context.extracted_brief is None:
            raise MVPPlannerError("MVP planning requires extracted_brief in AIContext")
        if context.arbitration_result is None:
            raise MVPPlannerError("MVP planning requires arbitration_result in AIContext")

        if context.assessment_result is None:
            raise MVPPlannerError(
                "MVP planning requires assessment_result in AIContext"
            )

        result = self.plan_assessment(
            brief_input=context.brief_input,
            extracted_brief=context.extracted_brief,
            assessment_result=context.assessment_result,
            arbitration_result=context.arbitration_result,
        )
        return context.with_mvp_planning_result(result)

    def run_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
        return self.plan_context(context)

    def _validate_plan(self, plan: MVPPlan) -> None:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if not plan.keep:
            raise ValueError("keep must not be empty")
        if not plan.simplify:
            raise ValueError("simplify must not be empty")
        if not plan.mvp_scope:
            raise ValueError("mvp_scope must not be empty")
        if not plan.rationale:
            raise ValueError("rationale must not be empty")

    def _build_user_prompt(
        self,
        brief_input: BriefInput,
        extracted_brief: ExtractedBrief,
        risk_analysis_result: dict,
        evaluation_result: dict,
        arbitration_result: ArbitrationResult,
    ) -> str:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return json.dumps(
            {
                "brief_input": brief_input.model_dump(mode="json"),
                "extracted_brief": extracted_brief.model_dump(mode="json"),
                "risk_analysis_result": risk_analysis_result,
                "evaluation_result": evaluation_result,
                "arbitration_result": arbitration_result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )

    def _build_messages(
        self,
        brief_input: BriefInput,
        extracted_brief: ExtractedBrief,
        risk_analysis_result: dict,
        evaluation_result: dict,
        arbitration_result: ArbitrationResult,
    ):
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        rendered = self._render_prompt(
            {
                "planning_context": self._build_user_prompt(
                    brief_input=brief_input,
                    extracted_brief=extracted_brief,
                    risk_analysis_result=risk_analysis_result,
                    evaluation_result=evaluation_result,
                    arbitration_result=arbitration_result,
                )
            }
        )
        return (
            {"role": "system", "content": rendered.system},
            {"role": "user", "content": rendered.user or ""},
        )

    def _run_plan(
        self,
        *,
        brief_input: BriefInput,
        extracted_brief: ExtractedBrief,
        risk_analysis_result: dict,
        evaluation_result: dict,
        arbitration_result: ArbitrationResult,
        risk_count: int,
        criterion_count: int,
    ) -> MVPPlanningResult:
        """Выполняет шаг «run plan». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        run_result = self._execute_structured_stage(
            trace_name="mvp_planner.brief",
            span_name="mvp_planner.llm",
            trace_input={
                "arbitration_status": arbitration_result.final_status.value,
                "risk_count": risk_count,
                "criterion_count": criterion_count,
            },
            messages=self._build_messages(
                brief_input=brief_input,
                extracted_brief=extracted_brief,
                risk_analysis_result=risk_analysis_result,
                evaluation_result=evaluation_result,
                arbitration_result=arbitration_result,
            ),
            response_model=MVPPlan,
            payload_validator=self._validate_plan,
        )
        return MVPPlanningResult(
            plan=run_result.payload,
            technical_info=MVPPlanningTechnicalInfo(
                llm_invoked=True,
                attempts=run_result.attempts,
                prompt_name=run_result.prompt_name,
                trace_enabled=run_result.trace_enabled,
                trace_name=run_result.trace_name,
                model_name=run_result.model_name,
                skipped_reason=None,
                raw_response=run_result.raw_response,
                recovered_errors=run_result.recovered_errors,
            ),
        )

    @staticmethod
    def _risk_analysis_prompt_section(assessment_result: AssessmentResult) -> dict:
        """Выполняет шаг «risk analysis prompt section». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        technical_info = assessment_result.technical_info
        return {
            "risks": [
                risk.model_dump(mode="json")
                for risk in assessment_result.risks
            ],
            "has_risks": assessment_result.has_risks,
            "summary": assessment_result.summary,
            "technical_info": {
                "attempts": technical_info.attempts,
                "prompt_name": technical_info.prompt_name or "assessment",
                "trace_enabled": technical_info.trace_enabled,
                "trace_name": technical_info.trace_name,
                "model_name": technical_info.model_name,
                "retriever_used": technical_info.retriever_used,
                "retrieved_context_count": technical_info.retrieved_context_count,
                "raw_response": technical_info.raw_response,
                "recovered_errors": technical_info.recovered_errors,
            },
        }

    @staticmethod
    def _evaluation_prompt_section(assessment_result: AssessmentResult) -> dict:
        """Выполняет шаг «evaluation prompt section». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        technical_info = assessment_result.technical_info
        return {
            "criterion_evaluations": [
                item.model_dump(mode="json")
                for item in assessment_result.criterion_evaluations
            ],
            "summary": assessment_result.summary,
            "technical_info": {
                "attempts": technical_info.attempts,
                "prompt_name": technical_info.prompt_name or "assessment",
                "trace_enabled": technical_info.trace_enabled,
                "trace_name": technical_info.trace_name,
                "model_name": technical_info.model_name,
                "retriever_used": technical_info.retriever_used,
                "retrieved_context_count": technical_info.retrieved_context_count,
                "criteria_count": technical_info.criteria_count,
                "raw_response": technical_info.raw_response,
                "recovered_errors": technical_info.recovered_errors,
            },
        }

    def _render_system_prompt(self) -> str:
        """Готовит человекочитаемый текст из внутренних данных. Это нужно для промптов, объяснений или финального ответа."""
        return self._render_prompt({"planning_context": ""}).system

    @staticmethod
    def _default_prompt_path() -> Path:
        """Возвращает значение по умолчанию, чтобы этап мог работать без ручной настройки."""
        return Path(__file__).resolve().parents[2] / "prompts" / "mvp_planner.md"

    def _build_failure_exception(
        self,
        attempts: int,
        last_error: Exception | None,
    ) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return MVPPlannerError(f"Unable to generate MVP plan after {attempts} attempts")
