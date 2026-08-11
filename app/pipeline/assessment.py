"""Preparation contracts for the unified Assessment stage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.config import (
    CriteriaConfig,
    CriteriaConfigError,
    CriteriaLoader,
    Criterion,
    RiskType,
    get_criteria_config,
)
from app.llm.runner import LLMRunResult, LLMRunner
from app.pipeline.base import BaseLLMStage
from app.prompts import PromptManager, RenderedPrompt
from app.schemas.ai_context import AIContext
from app.schemas.assessment import (
    AssessmentEvidence,
    AssessmentPayload,
    AssessmentResult,
    AssessmentTechnicalInfo,
)
from app.schemas.evaluation import CriterionEvaluation
from app.schemas.knowledge import SearchResult
from app.schemas.risk import Risk
from app.tracing.tracing import TracingClient

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class AssessmentError(RuntimeError):
    """Raised when assessment input cannot be prepared."""


class AssessmentConfigError(AssessmentError):
    """Raised when assessment criteria configuration is missing or invalid."""


class AssessmentRetriever(Protocol):
    """Retriever contract used by the future Assessment component."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[SearchResult]:
        """Return relevant knowledge chunks for an assessment query."""


class AssessmentPreparedInput(BaseModel):
    """Validated input for the future BaseLLMStage-backed Assessment.

    The concrete Assessment stage should use this model as its stage input,
    ``AssessmentPayload`` as the LLM structured output model, and convert it to
    ``AssessmentResult`` during post-processing.
    """

    context: AIContext
    criteria_config: CriteriaConfig
    criteria: list[Criterion]
    risk_types: list[RiskType]
    retrieved_context: list[SearchResult] = Field(default_factory=list)
    retrieval_query: str | None = None
    metadata_filters: dict[str, object] | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @property
    def criteria_count(self) -> int:
        """Return the number of criteria available for assessment."""
        return len(self.criteria)

    @property
    def risk_types_count(self) -> int:
        """Return the number of configured risk types."""
        return len(self.risk_types)

    def to_prompt_context(self) -> dict[str, Any]:
        """Serialize prepared input for a future prompt renderer."""
        return {
            "brief": self.context.brief_input.model_dump(mode="json"),
            "extracted_brief": (
                self.context.extracted_brief.model_dump(mode="json")
                if self.context.extracted_brief is not None
                else None
            ),
            "completeness_result": (
                self.context.completeness_result.model_dump(mode="json")
                if self.context.completeness_result is not None
                else None
            ),
            "criteria_config": self.criteria_config.model_dump(mode="json"),
            "retrieved_context": [
                item.model_dump(mode="json") for item in self.retrieved_context
            ],
            "metadata": self.context.metadata,
        }


class AssessmentStage(
    BaseLLMStage[AssessmentPreparedInput, AssessmentPayload, AssessmentResult]
):
    """Unified LLM stage that evaluates criteria and risks."""

    output_model: ClassVar[type[AssessmentPayload]] = AssessmentPayload

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
        retriever: AssessmentRetriever | None = None,
        criteria_config: CriteriaConfig | None = None,
        criteria_path: str | Path | None = None,
    ) -> None:
        """Initialize Assessment with LLM transport and deterministic preparation."""
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
        self._preparation = AssessmentPreparation(
            retriever=retriever,
            criteria_config=criteria_config,
            criteria_path=criteria_path,
        )
        self._last_run_metadata: dict[str, Any] = {}

    def assess(
        self,
        context: AIContext,
        *,
        top_k: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> AIContext:
        """Return context enriched with a unified assessment result."""
        prepared = self._preparation.prepare(
            context,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        return prepared.context.with_assessment_result(self.run(prepared))

    def run_context(self, context: AIContext) -> AIContext:
        """Run this stage using the common AIContext pipeline contract."""
        return self.assess(context)

    @property
    def trace_name(self) -> str:
        """Return the Langfuse trace name for unified assessment."""
        return "assessment.brief"

    @property
    def span_name(self) -> str:
        """Return the Langfuse span name for the assessment LLM call."""
        return "assessment.llm"

    def build_prompt(self, stage_input: AssessmentPreparedInput) -> str:
        """Render the user prompt from deterministic assessment inputs."""
        rendered = self._render_assessment_prompt(stage_input)
        return rendered.user or rendered.system

    def build_system_prompt(self, stage_input: AssessmentPreparedInput) -> str | None:
        """Render the system prompt when the template has a user section."""
        rendered = self._render_assessment_prompt(stage_input)
        return rendered.system if rendered.user is not None else None

    def build_context(self, stage_input: AssessmentPreparedInput) -> None:
        """Do not duplicate transport context; prompt rendering owns input data."""
        return None

    def build_trace_input(self, stage_input: AssessmentPreparedInput) -> dict[str, Any]:
        """Build trace metadata without serializing the full brief or retrieved text."""
        metadata = {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "criteria_count": stage_input.criteria_count,
            "risk_types_count": stage_input.risk_types_count,
            "retriever_used": self._preparation.retriever_used,
            "retrieved_context_count": len(stage_input.retrieved_context),
            "is_complete": stage_input.context.completeness_result.is_complete,
            "completeness_level": stage_input.context.completeness_result.level.value,
        }
        self._last_run_metadata = metadata
        return metadata

    def postprocess(
        self,
        result: LLMRunResult[AssessmentPayload],
    ) -> AssessmentResult:
        """Convert the shared LLM result into unified assessment output."""
        payload = self._normalize_payload(result.payload)
        return AssessmentResult(
            criterion_evaluations=payload.criterion_evaluations,
            risks=payload.risks,
            evidence=payload.evidence,
            has_risks=payload.has_risks,
            recommendation=payload.recommendation,
            summary=self._strip_optional_text(payload.summary),
            confidence=payload.confidence,
            technical_info=AssessmentTechnicalInfo(
                attempts=result.attempts,
                prompt_name=self.prompt_name,
                trace_enabled=result.trace_enabled,
                trace_name=result.trace_name or self.trace_name,
                model_name=result.model_name,
                retriever_used=bool(self._last_run_metadata.get("retriever_used")),
                retrieved_context_count=int(
                    self._last_run_metadata.get("retrieved_context_count", 0)
                ),
                criteria_count=int(self._last_run_metadata.get("criteria_count", 0)),
                risk_types_count=int(
                    self._last_run_metadata.get("risk_types_count", 0)
                ),
                raw_response=result.raw_response,
                recovered_errors=list(result.recovered_errors),
                provider_metadata=dict(result.provider_metadata),
            ),
        )

    @staticmethod
    def _default_prompt_path() -> Path:
        """Return the default prompt template path."""
        return Path(__file__).resolve().parents[2] / "prompts" / "assessment.md"

    def _build_failure_exception(
        self,
        attempts: int,
        last_error: Exception | None,
    ) -> Exception:
        """Build the assessment-specific failure exception."""
        return AssessmentError(f"Unable to assess brief after {attempts} attempts")

    def _render_assessment_prompt(
        self,
        stage_input: AssessmentPreparedInput,
    ) -> RenderedPrompt:
        """Render the configured Assessment prompt through PromptManager."""
        return self._render_prompt(
            {
                "normalized_brief": stage_input.context.normalized_text,
                "extracted_brief": stage_input.context.extracted_brief,
                "completeness_result": stage_input.context.completeness_result,
                "criteria": [
                    item.model_dump(mode="json") for item in stage_input.criteria
                ],
                "risk_types": [
                    item.model_dump(mode="json") for item in stage_input.risk_types
                ],
                "retrieved_context": [
                    item.model_dump(mode="json")
                    for item in stage_input.retrieved_context
                ],
            }
        )

    def _normalize_payload(self, payload: AssessmentPayload) -> AssessmentPayload:
        """Apply minimal cleanup without adding deterministic business decisions."""
        return payload.model_copy(
            update={
                "criterion_evaluations": [
                    self._normalize_criterion(item)
                    for item in payload.criterion_evaluations
                ],
                "risks": [self._normalize_risk(item) for item in payload.risks],
                "evidence": [
                    self._normalize_evidence(item)
                    for item in payload.evidence
                    if item.quote.strip()
                ],
                "summary": self._strip_optional_text(payload.summary),
            }
        )

    @classmethod
    def _normalize_criterion(
        cls,
        item: CriterionEvaluation,
    ) -> CriterionEvaluation:
        """Trim criterion evaluation strings and evidence fragments."""
        return item.model_copy(
            update={
                "criterion": item.criterion.strip(),
                "criterion_title": cls._strip_optional_text(item.criterion_title),
                "evidence": cls._strip_text_list(item.evidence),
                "explanation": cls._strip_optional_text(item.explanation),
                "notes": cls._strip_optional_text(item.notes),
            }
        )

    @classmethod
    def _normalize_risk(cls, item: Risk) -> Risk:
        """Trim risk strings and evidence fragments."""
        return item.model_copy(
            update={
                "type": item.type.strip(),
                "description": item.description.strip(),
                "evidence": cls._strip_text_list(item.evidence),
                "notes": cls._strip_optional_text(item.notes),
            }
        )

    @classmethod
    def _normalize_evidence(cls, item: AssessmentEvidence) -> AssessmentEvidence:
        """Trim evidence strings and related identifier lists."""
        return item.model_copy(
            update={
                "source": item.source.strip(),
                "quote": item.quote.strip(),
                "related_criteria": cls._strip_text_list(item.related_criteria),
                "related_risks": cls._strip_text_list(item.related_risks),
            }
        )

    @staticmethod
    def _strip_optional_text(value: str | None) -> str | None:
        """Trim optional strings and normalize blank values to None."""
        if value is None:
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _strip_text_list(values: list[str]) -> list[str]:
        """Trim strings and remove blank values."""
        return [value.strip() for value in values if value and value.strip()]


class AssessmentPreparation:
    """Prepare all deterministic inputs required by the future Assessment stage."""

    def __init__(
        self,
        retriever: AssessmentRetriever | None = None,
        criteria_config: CriteriaConfig | None = None,
        criteria_path: str | Path | None = None,
    ) -> None:
        """Create assessment preparation with optional retriever and config source."""
        if criteria_config is not None and criteria_path is not None:
            raise ValueError("Pass either criteria_config or criteria_path, not both")

        self._retriever = retriever
        self._config = self._load_config(criteria_config, criteria_path)
        self._criteria = list(self._config.evaluation.criteria)
        self._risk_types = list(self._config.evaluation.risk_analysis.risk_types)

    @property
    def criteria_config(self) -> CriteriaConfig:
        """Return the validated criteria configuration."""
        return self._config

    @property
    def criteria(self) -> list[Criterion]:
        """Return configured criteria in deterministic order."""
        return list(self._criteria)

    @property
    def risk_types(self) -> list[RiskType]:
        """Return configured risk types in deterministic order."""
        return list(self._risk_types)

    @property
    def retriever_used(self) -> bool:
        """Return whether Assessment preparation has an active retriever."""
        return self._retriever is not None

    def prepare(
        self,
        context: AIContext,
        *,
        top_k: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> AssessmentPreparedInput:
        """Prepare context, criteria and optional retrieved knowledge for assessment."""
        self._validate_context(context)
        retrieval_query = self._build_retrieval_query(context)
        retrieved_context = self._retrieve_context(
            context=context,
            retrieval_query=retrieval_query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

        return AssessmentPreparedInput(
            context=context.with_retrieved_context(retrieved_context),
            criteria_config=self._config,
            criteria=self._criteria,
            risk_types=self._risk_types,
            retrieved_context=retrieved_context,
            retrieval_query=retrieval_query,
            metadata_filters=(
                dict(metadata_filters) if metadata_filters is not None else None
            ),
        )

    def _retrieve_context(
        self,
        context: AIContext,
        retrieval_query: str,
        top_k: int | None,
        metadata_filters: Mapping[str, object] | None,
    ) -> list[SearchResult]:
        """Retrieve fresh context or reuse context accumulated by earlier stages."""
        if self._retriever is None:
            return list(context.retrieved_context)

        return self._retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

    @staticmethod
    def _build_retrieval_query(context: AIContext) -> str:
        """Build a retrieval query from normalized brief and upstream signals."""
        parts: list[str] = [context.normalized_text]

        if context.extracted_brief is not None:
            extracted = context.extracted_brief
            if extracted.project_goal.value:
                parts.append(extracted.project_goal.value)
            parts.extend(item.value for item in extracted.tasks if item.value)
            parts.extend(item.value for item in extracted.technologies if item.value)
            parts.extend(item.value for item in extracted.integrations if item.value)

        if context.completeness_result is not None:
            parts.extend(
                item.field_key
                for item in context.completeness_result.missing_information
            )

        return "\n".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _validate_context(context: AIContext) -> None:
        """Ensure the upstream pipeline has produced required assessment inputs."""
        if context.extracted_brief is None:
            raise AssessmentError("Assessment requires extracted_brief in AIContext")
        if context.completeness_result is None:
            raise AssessmentError("Assessment requires completeness_result in AIContext")

    @staticmethod
    def _load_config(
        criteria_config: CriteriaConfig | None,
        criteria_path: str | Path | None,
    ) -> CriteriaConfig:
        """Load and validate criteria/risk definitions for Assessment."""
        if criteria_config is not None:
            config = criteria_config
        elif criteria_path is not None:
            try:
                config = CriteriaLoader.load(Path(criteria_path))
            except CriteriaConfigError as exc:
                raise AssessmentConfigError(str(exc)) from exc
        else:
            try:
                config = get_criteria_config()
            except CriteriaConfigError as exc:
                raise AssessmentConfigError(str(exc)) from exc

        if not config.evaluation.criteria:
            raise AssessmentConfigError("criteria configuration has no criteria")

        risk_analysis = config.evaluation.risk_analysis
        if risk_analysis is None:
            raise AssessmentConfigError(
                "criteria configuration is missing risk_analysis section"
            )
        if not risk_analysis.risk_types:
            raise AssessmentConfigError(
                "risk_analysis configuration has no risk_types"
            )

        return config
