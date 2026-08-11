"""Shared immutable-by-copy AI pipeline context model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.criteria import CriteriaConfig
from app.schemas.assessment import AssessmentResult
from app.schemas.brief import BriefInput
from app.schemas.completeness import CompletenessResult
from app.schemas.decision import ArbitrationResult
from app.schemas.extraction import ExtractionResult, ExtractedBrief
from app.schemas.knowledge import SearchResult
from app.schemas.mvp import MVPPlanningResult
from app.schemas.question import QuestionGenerationResult
from app.schemas.self_check import SelfCheckResult


class PipelineInputState(BaseModel):
    """Initial user input carried through the pipeline."""

    brief_input: BriefInput

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def original_text(self) -> str:
        """Return the original brief text."""
        return self.brief_input.original_text

    @property
    def normalized_text(self) -> str:
        """Return the normalized brief text."""
        return self.brief_input.normalized_text


class RetrievalState(BaseModel):
    """Retrieved knowledge context accumulated by pipeline stages."""

    results: list[SearchResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class PipelineResults(BaseModel):
    """Structured outputs produced by pipeline stages."""

    extracted_brief: ExtractedBrief | None = None
    extraction_result: ExtractionResult | None = None
    completeness_result: CompletenessResult | None = None
    assessment_result: AssessmentResult | None = None
    arbitration_result: ArbitrationResult | None = None
    clarification_result: QuestionGenerationResult | None = None
    mvp_planning_result: MVPPlanningResult | None = None
    self_check_result: SelfCheckResult | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ResponseState(BaseModel):
    """Final user-facing response produced by the pipeline."""

    text: str | None = None
    payload: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        """Normalize optional final response text."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("final response text must not be empty")
        return value


class PipelineTechnicalState(BaseModel):
    """Non-business metadata and technical details for pipeline execution."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: CriteriaConfig | None = None
    stage_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


class AIContext(BaseModel):
    """Primary immutable-by-copy state object passed between pipeline stages."""

    inputs: PipelineInputState
    retrieval: RetrievalState = Field(default_factory=RetrievalState)
    results: PipelineResults = Field(default_factory=PipelineResults)
    response: ResponseState = Field(default_factory=ResponseState)
    technical: PipelineTechnicalState = Field(default_factory=PipelineTechnicalState)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def brief_input(self) -> BriefInput:
        """Return the normalized brief input."""
        return self.inputs.brief_input

    @property
    def original_text(self) -> str:
        """Return the original brief text."""
        return self.inputs.original_text

    @property
    def normalized_text(self) -> str:
        """Return the normalized brief text."""
        return self.inputs.normalized_text

    @property
    def extracted_brief(self) -> ExtractedBrief | None:
        """Return extracted structured facts, if available."""
        return self.results.extracted_brief

    @property
    def extraction_result(self) -> ExtractionResult | None:
        """Return extractor output, if available."""
        return self.results.extraction_result

    @property
    def completeness_result(self) -> CompletenessResult | None:
        """Return completeness output, if available."""
        return self.results.completeness_result

    @property
    def assessment_result(self) -> AssessmentResult | None:
        """Return unified assessment output, if available."""
        return self.results.assessment_result

    @property
    def arbitration_result(self) -> ArbitrationResult | None:
        """Return deterministic arbitration output, if available."""
        return self.results.arbitration_result

    @property
    def clarification_result(self) -> QuestionGenerationResult | None:
        """Return clarification-question output, if available."""
        return self.results.clarification_result

    @property
    def mvp_planning_result(self) -> MVPPlanningResult | None:
        """Return MVP planning output, if available."""
        return self.results.mvp_planning_result

    @property
    def final_response_text(self) -> str | None:
        """Return the final user-facing response text, if available."""
        return self.response.text

    @property
    def final_response_payload(self) -> dict[str, Any] | None:
        """Return the final user-facing response payload, if available."""
        return self.response.payload

    @property
    def self_check_result(self) -> SelfCheckResult | None:
        """Return self-check output, if available."""
        return self.results.self_check_result

    @property
    def retrieved_context(self) -> list[SearchResult]:
        """Return retrieved knowledge context."""
        return self.retrieval.results

    @property
    def metadata(self) -> dict[str, Any]:
        """Return request-level metadata."""
        return self.technical.metadata

    @property
    def configuration(self) -> CriteriaConfig | None:
        """Return pipeline configuration snapshot, if available."""
        return self.technical.configuration

    @property
    def stage_metadata(self) -> dict[str, dict[str, Any]]:
        """Return technical metadata collected by stages."""
        return self.technical.stage_metadata

    @classmethod
    def from_brief(
        cls,
        brief_input: BriefInput,
        *,
        metadata: dict[str, Any] | None = None,
        configuration: CriteriaConfig | None = None,
    ) -> "AIContext":
        """Create a context from the normalized brief input."""
        return cls(
            inputs=PipelineInputState(brief_input=brief_input),
            technical=PipelineTechnicalState(
                metadata=metadata or {},
                configuration=configuration,
            ),
        )

    def with_extraction_result(self, result: ExtractionResult) -> "AIContext":
        """Return a context updated with extraction output."""
        return self._with_results(
            update={
                "extraction_result": result,
                "extracted_brief": result.extracted_brief,
            }
        )

    def with_extracted_brief(self, extracted_brief: ExtractedBrief) -> "AIContext":
        """Return a context updated with extracted structured data."""
        return self._with_results(update={"extracted_brief": extracted_brief})

    def with_completeness_result(self, result: CompletenessResult) -> "AIContext":
        """Return a context updated with completeness output."""
        return self._with_results(update={"completeness_result": result})

    def with_assessment_result(self, result: AssessmentResult) -> "AIContext":
        """Return a context updated with unified assessment output."""
        return self._with_results(update={"assessment_result": result})

    def with_assessment(self, result: AssessmentResult) -> "AIContext":
        """Return a context updated with unified assessment output."""
        return self.with_assessment_result(result)

    def with_arbitration_result(self, result: ArbitrationResult) -> "AIContext":
        """Return a context updated with deterministic arbitration output."""
        return self._with_results(update={"arbitration_result": result})

    def with_clarification_result(
        self,
        result: QuestionGenerationResult,
    ) -> "AIContext":
        """Return a context updated with clarification questions."""
        return self._with_results(update={"clarification_result": result})

    def with_mvp_planning_result(self, result: MVPPlanningResult) -> "AIContext":
        """Return a context updated with MVP planning output."""
        return self._with_results(update={"mvp_planning_result": result})

    def with_retrieved_context(self, results: list[SearchResult]) -> "AIContext":
        """Return a context with retrieved knowledge context replaced."""
        return self.model_copy(
            update={"retrieval": RetrievalState(results=list(results))}
        )

    def append_retrieved_context(self, results: list[SearchResult]) -> "AIContext":
        """Return a context with additional retrieved knowledge context."""
        return self.model_copy(
            update={
                "retrieval": RetrievalState(
                    results=[*self.retrieved_context, *results]
                )
            }
        )

    def with_final_response(
        self,
        response_text: str,
        response_payload: dict[str, Any] | None = None,
    ) -> "AIContext":
        """Return a context updated with the final user-facing response."""
        return self.model_copy(
            update={
                "response": ResponseState(
                    text=response_text,
                    payload=response_payload,
                )
            }
        )

    def with_self_check_result(self, result: SelfCheckResult) -> "AIContext":
        """Return a context updated with self-check output."""
        return self._with_results(update={"self_check_result": result})

    def with_metadata(self, **metadata: Any) -> "AIContext":
        """Return a context with merged metadata."""
        return self.model_copy(
            update={
                "technical": self.technical.model_copy(
                    update={"metadata": {**self.metadata, **metadata}}
                )
            }
        )

    def with_stage_metadata(self, stage_name: str, **metadata: Any) -> "AIContext":
        """Return a context with merged technical metadata for one stage."""
        stage_name = stage_name.strip()
        if not stage_name:
            raise ValueError("stage_name must not be empty")

        existing = self.stage_metadata.get(stage_name, {})
        return self.model_copy(
            update={
                "technical": self.technical.model_copy(
                    update={
                        "stage_metadata": {
                            **self.stage_metadata,
                            stage_name: {**existing, **metadata},
                        }
                    }
                )
            }
        )

    def _with_results(self, update: dict[str, Any]) -> "AIContext":
        """Return a context with updated stage results."""
        return self.model_copy(
            update={"results": self.results.model_copy(update=update)}
        )
