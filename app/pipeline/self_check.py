"""Final self-check for user-facing responses."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.llm.runner import LLMRunner
from app.pipeline.base import BaseLLMStage
from app.prompts import PromptManager
from app.schemas import (
    AIContext,
    ArbitrationResult,
    AssessmentResult,
    DecisionStatus,
    ExtractedBrief,
    SelfCheckContext,
    SelfCheckPayload,
    SelfCheckResult,
    SelfCheckTechnicalInfo,
)
from app.tracing.tracing import TracingClient

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class SelfCheckError(RuntimeError):
    """Raised when the self-check cannot complete."""


def _risk_analysis_prompt_section(assessment_result: AssessmentResult) -> dict[str, Any]:
    """Build the legacy-shaped risk section from unified assessment output."""
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


def _evaluation_prompt_section(assessment_result: AssessmentResult) -> dict[str, Any]:
    """Build the legacy-shaped evaluation section from unified assessment output."""
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


class DeterministicValidator:
    """Perform deterministic checks before any optional LLM review."""

    def validate(self, context: SelfCheckContext) -> SelfCheckResult:
        """Validate the response using only deterministic rules."""
        issues: list[str] = []
        warnings: list[str] = []
        checked_fields: list[str] = []

        checked_fields.append("arbitration_result.final_status")
        response_status = self._extract_response_status(context.response_payload)
        if response_status is not None:
            checked_fields.append("response_payload.status")
            if response_status != context.arbitration_result.final_status:
                issues.append("Response status does not match the arbiter decision.")
        else:
            warnings.append(
                "Response payload does not expose a status field for direct comparison."
            )

        checked_fields.extend(
            [
                "brief_input",
                "extracted_brief",
                "completeness_result",
                "risk_analysis_result",
                "evaluation_result",
                "arbitration_result",
            ]
        )
        if context.retrieved_context:
            checked_fields.append("retrieved_context")

        facts = self._extract_response_facts(context.response_payload)
        if facts:
            checked_fields.append("response_payload.facts")
            unsupported = self._find_unsupported_facts(context, facts)
            for fact in unsupported:
                issues.append(f"Unsupported fact in response payload: {fact}")

        self._check_status_consistency(
            context=context,
            issues=issues,
            warnings=warnings,
            checked_fields=checked_fields,
        )

        checked_fields.extend(self._textual_checked_fields(context.response_text))
        issues.extend(
            self._check_text_contradictions(
                response_text=context.response_text,
                status=context.arbitration_result.final_status,
            )
        )

        checked_fields = list(dict.fromkeys(checked_fields))
        return SelfCheckResult(
            is_valid=not issues,
            issues=issues,
            warnings=warnings,
            checked_fields=checked_fields,
            technical_info=SelfCheckTechnicalInfo(
                deterministic_issues_count=len(issues),
                deterministic_warning_count=len(warnings),
                llm_invoked=False,
                attempts=0,
                prompt_name=None,
                trace_enabled=False,
                trace_name="self_check.brief",
                model_name=None,
                needs_llm_review=bool(context.response_text.strip()) and not issues,
                raw_response=None,
                recovered_errors=[],
            ),
        )

    def _check_status_consistency(
        self,
        context: SelfCheckContext,
        issues: list[str],
        warnings: list[str],
        checked_fields: list[str],
    ) -> None:
        """Validate questions and MVP plan placement against the arbiter status."""
        status = context.arbitration_result.final_status
        payload = context.response_payload or {}

        if status == DecisionStatus.clarify:
            checked_fields.append("clarification_result")
            questions = self._extract_questions(payload)
            if not questions and context.clarification_result is None:
                issues.append("CLARIFY responses must include clarification questions.")
        else:
            questions = self._extract_questions(payload)
            if questions:
                issues.append(
                    "Clarification questions are not allowed unless status is CLARIFY."
                )

        if status == DecisionStatus.simplify:
            checked_fields.append("mvp_planning_result")
            if self._extract_mvp_plan(payload) is None:
                if context.mvp_planning_result is None:
                    issues.append("SIMPLIFY responses must include an MVP plan.")
        else:
            if self._extract_mvp_plan(payload) is not None:
                issues.append("MVP plans are only allowed when status is SIMPLIFY.")

        if status == DecisionStatus.accept:
            checked_fields.append("acceptance_consistency")
        elif status == DecisionStatus.reject:
            checked_fields.append("rejection_consistency")
        elif status == DecisionStatus.clarify:
            checked_fields.append("clarification_consistency")
        elif status == DecisionStatus.simplify:
            checked_fields.append("simplification_consistency")
        else:
            checked_fields.append("mentor_review_consistency")

        if status != DecisionStatus.clarify and context.clarification_result:
            warnings.append(
                "Clarification questions were provided even though the final status is not CLARIFY."
            )
        if status != DecisionStatus.simplify and context.mvp_planning_result:
            warnings.append(
                "An MVP plan was provided even though the final status is not SIMPLIFY."
            )

    def _find_unsupported_facts(
        self,
        context: SelfCheckContext,
        facts: list[str],
    ) -> list[str]:
        """Find fact claims that are not supported by the available context."""
        corpus = self._build_support_corpus(context)
        unsupported: list[str] = []
        for fact in facts:
            normalized_fact = self._normalize_text(fact)
            if not any(
                self._fact_is_supported(normalized_fact, source) for source in corpus
            ):
                unsupported.append(fact)
        return unsupported

    @staticmethod
    def _fact_is_supported(fact: str, source: str) -> bool:
        """Check whether a fact is supported by a source fragment."""
        source = DeterministicValidator._normalize_text(source)
        return fact in source or source in fact

    @staticmethod
    def _build_support_corpus(context: SelfCheckContext) -> list[str]:
        """Build the textual evidence corpus available to the self-check."""
        corpus: list[str] = []
        corpus.extend(
            [
                context.brief_input.original_text,
                context.brief_input.normalized_text,
                context.response_text,
            ]
        )
        corpus.extend(DeterministicValidator._flatten_model_strings(context.extracted_brief))
        corpus.extend(DeterministicValidator._flatten_model_strings(context.completeness_result))
        corpus.extend(
            DeterministicValidator._flatten_model_strings(
                _risk_analysis_prompt_section(context.assessment_result)
            )
        )
        corpus.extend(
            DeterministicValidator._flatten_model_strings(
                _evaluation_prompt_section(context.assessment_result)
            )
        )
        corpus.extend(DeterministicValidator._flatten_model_strings(context.arbitration_result))
        if context.clarification_result is not None:
            corpus.extend(
                DeterministicValidator._flatten_model_strings(context.clarification_result)
            )
        if context.mvp_planning_result is not None:
            corpus.extend(
                DeterministicValidator._flatten_model_strings(context.mvp_planning_result)
            )
        for result in context.retrieved_context:
            corpus.extend(
                [
                    result.document.text,
                    result.document.metadata.source,
                    result.document.metadata.title or "",
                    result.document.metadata.document_type or "",
                    result.document.metadata.category or "",
                    result.document.metadata.version or "",
                ]
            )
            corpus.extend(
                str(value) for value in (result.document.metadata.model_extra or {}).values()
            )
        return [item for item in corpus if isinstance(item, str) and item.strip()]

    @staticmethod
    def _flatten_model_strings(model: Any) -> list[str]:
        """Collect string fragments from a Pydantic model or container."""
        fragments: list[str] = []
        if isinstance(model, str):
            return [model]
        if isinstance(model, list):
            for item in model:
                fragments.extend(DeterministicValidator._flatten_model_strings(item))
            return fragments
        if isinstance(model, dict):
            for value in model.values():
                fragments.extend(DeterministicValidator._flatten_model_strings(value))
            return fragments
        if hasattr(model, "model_dump"):
            return DeterministicValidator._flatten_model_strings(model.model_dump(mode="json"))
        if hasattr(model, "__dict__"):
            return DeterministicValidator._flatten_model_strings(vars(model))
        return fragments

    @staticmethod
    def _extract_response_status(payload: dict[str, Any] | None) -> DecisionStatus | None:
        """Extract a decision status from the response payload."""
        if not payload:
            return None
        raw_status = payload.get("status") or payload.get("final_status")
        if raw_status is None:
            return None
        try:
            return DecisionStatus(raw_status)
        except ValueError:
            return None

    @staticmethod
    def _extract_questions(payload: dict[str, Any]) -> list[str]:
        """Extract clarification questions from the response payload."""
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            return []

        questions: list[str] = []
        for item in raw_questions:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    questions.append(item)
            elif isinstance(item, dict):
                question = item.get("question")
                if isinstance(question, str) and question.strip():
                    questions.append(question.strip())
        return questions

    @staticmethod
    def _extract_mvp_plan(payload: dict[str, Any]) -> dict[str, Any] | None:
        """Extract an MVP plan from the response payload."""
        raw_plan = payload.get("mvp_plan")
        if isinstance(raw_plan, dict) and raw_plan:
            return raw_plan
        return None

    @staticmethod
    def _extract_response_facts(payload: dict[str, Any] | None) -> list[str]:
        """Extract explicit fact claims from the response payload."""
        if not payload:
            return []
        facts = payload.get("facts") or payload.get("claims") or []
        if not isinstance(facts, list):
            return []
        return [fact.strip() for fact in facts if isinstance(fact, str) and fact.strip()]

    @staticmethod
    def _check_text_contradictions(
        response_text: str,
        status: DecisionStatus,
    ) -> list[str]:
        """Find obvious status contradictions in the free-form response text."""
        normalized = DeterministicValidator._normalize_text(response_text)
        contradictions: dict[DecisionStatus, list[str]] = {
            DecisionStatus.accept: [
                "did not pass",
                "does not pass",
                "rejected",
                "failed criteria",
                "requires rejection",
            ],
            DecisionStatus.reject: [
                "passed",
                "accepted",
                "approved",
            ],
            DecisionStatus.clarify: [
                "all information is sufficient",
                "no questions needed",
            ],
            DecisionStatus.simplify: [
                "no simplification needed",
                "ready as is",
            ],
            DecisionStatus.mentor_review: [
                "no human review needed",
            ],
        }
        issues: list[str] = []
        for phrase in contradictions.get(status, []):
            if phrase in normalized:
                issues.append(f"Response text contradicts the arbiter status {status.value}.")
                break
        return issues

    @staticmethod
    def _textual_checked_fields(response_text: str) -> list[str]:
        """Return fields covered by text-level checks."""
        if not response_text.strip():
            return []
        return ["response_text"]

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize free-form text for heuristic comparison."""
        value = value.lower()
        value = re.sub(r"\s+", " ", value)
        return value.strip()


class LLMSelfChecker(BaseLLMStage):
    """Optional LLM-based semantic self-check for the final response."""

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
        """Initialize the LLM self-checker."""
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

    def check(
        self,
        context: SelfCheckContext,
        deterministic_result: SelfCheckResult,
    ) -> SelfCheckResult:
        """Run the LLM self-check after deterministic validation passes."""
        if deterministic_result.issues:
            return deterministic_result
        if not context.response_text.strip():
            return deterministic_result

        run_result = self._execute_structured_stage(
            trace_name="self_check.llm",
            span_name="self_check.llm_review",
            trace_input={
                "arbitration_status": context.arbitration_result.final_status.value,
            },
            messages=[
                {
                    "role": "system",
                    "content": self._render_system_prompt(
                        context=context,
                        deterministic_result=deterministic_result,
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        context=context,
                        deterministic_result=deterministic_result,
                    ),
                },
            ],
            response_model=SelfCheckPayload,
        )
        return SelfCheckResult(
            is_valid=run_result.payload.is_valid,
            issues=run_result.payload.issues,
            warnings=run_result.payload.warnings,
            checked_fields=run_result.payload.checked_fields,
            technical_info=SelfCheckTechnicalInfo(
                deterministic_issues_count=len(deterministic_result.issues),
                deterministic_warning_count=len(deterministic_result.warnings),
                llm_invoked=True,
                attempts=run_result.attempts,
                prompt_name=run_result.prompt_name,
                trace_enabled=run_result.trace_enabled,
                trace_name=run_result.trace_name,
                model_name=run_result.model_name,
                needs_llm_review=False,
                raw_response=run_result.raw_response,
                recovered_errors=run_result.recovered_errors,
            ),
        )

    def _build_user_prompt(
        self,
        context: SelfCheckContext,
        deterministic_result: SelfCheckResult,
    ) -> str:
        """Serialize the self-check context for the LLM."""
        return json.dumps(
            {
                "response_text": context.response_text,
                "response_payload": context.response_payload,
                "arbitration_result": context.arbitration_result.model_dump(mode="json"),
                "deterministic_result": deterministic_result.model_dump(mode="json"),
                "brief_input": context.brief_input.model_dump(mode="json"),
                "extracted_brief": context.extracted_brief.model_dump(mode="json"),
                "completeness_result": context.completeness_result.model_dump(mode="json"),
                "risk_analysis_result": _risk_analysis_prompt_section(
                    context.assessment_result
                ),
                "evaluation_result": _evaluation_prompt_section(
                    context.assessment_result
                ),
                "clarification_result": (
                    context.clarification_result.model_dump(mode="json")
                    if context.clarification_result is not None
                    else None
                ),
                "mvp_planning_result": (
                    context.mvp_planning_result.model_dump(mode="json")
                    if context.mvp_planning_result is not None
                    else None
                ),
                "retrieved_context": [
                    result.model_dump(mode="json") for result in context.retrieved_context
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    def _render_system_prompt(
        self,
        context: SelfCheckContext,
        deterministic_result: SelfCheckResult,
    ) -> str:
        """Render the self-check system prompt."""
        return self._render_prompt(
            {
                "review_context": {
                    "arbitration_status": context.arbitration_result.final_status.value,
                    "deterministic_issues": deterministic_result.issues,
                    "deterministic_warnings": deterministic_result.warnings,
                },
                "response_context": {
                    "response_text": context.response_text,
                },
            }
        ).system

    @staticmethod
    def _default_prompt_path() -> Path:
        """Return the default prompt path for the LLM review."""
        return Path(__file__).resolve().parents[2] / "prompts" / "self_check.md"

    def _build_failure_exception(
        self,
        attempts: int,
        last_error: Exception | None,
    ) -> Exception:
        """Build the self-check-specific failure exception."""
        return SelfCheckError(f"Unable to complete LLM self-check after {attempts} attempts")


class SelfChecker:
    """Orchestrate deterministic and optional LLM-based self-checks."""

    def __init__(
        self,
        llm_self_checker: LLMSelfChecker | None = None,
        deterministic_validator: DeterministicValidator | None = None,
    ) -> None:
        """Initialize the self-check orchestrator."""
        self._llm_self_checker = llm_self_checker
        self._deterministic_validator = (
            deterministic_validator or DeterministicValidator()
        )

    def check(self, context: SelfCheckContext) -> SelfCheckResult:
        """Run deterministic checks and optionally the LLM review."""
        deterministic_result = self._deterministic_validator.validate(context)
        if deterministic_result.issues:
            return deterministic_result
        if self._llm_self_checker is None:
            return deterministic_result

        return self._llm_self_checker.check(context, deterministic_result)

    def check_context(self, context: AIContext) -> AIContext:
        """Return context enriched with self-check output."""
        self_check_context = self._build_self_check_context(context)
        return context.with_self_check_result(self.check(self_check_context))

    def run_context(self, context: AIContext) -> AIContext:
        """Run this stage using the common AIContext pipeline contract."""
        return self.check_context(context)

    @staticmethod
    def _build_self_check_context(context: AIContext) -> SelfCheckContext:
        """Build self-check input from the shared AIContext."""
        if context.final_response_text is None:
            raise SelfCheckError("Self-check requires final_response_text in AIContext")
        if context.extracted_brief is None:
            raise SelfCheckError("Self-check requires extracted_brief in AIContext")
        if context.completeness_result is None:
            raise SelfCheckError("Self-check requires completeness_result in AIContext")
        if context.arbitration_result is None:
            raise SelfCheckError("Self-check requires arbitration_result in AIContext")

        if context.assessment_result is None:
            raise SelfCheckError(
                "Self-check requires assessment_result in AIContext"
            )

        return SelfCheckContext(
            response_text=context.final_response_text,
            response_payload=context.final_response_payload,
            brief_input=context.brief_input,
            extracted_brief=context.extracted_brief,
            completeness_result=context.completeness_result,
            assessment_result=context.assessment_result,
            arbitration_result=context.arbitration_result,
            clarification_result=context.clarification_result,
            mvp_planning_result=context.mvp_planning_result,
            retrieved_context=context.retrieved_context,
        )
