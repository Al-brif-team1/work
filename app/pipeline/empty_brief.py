"""Deterministic rejection for empty or obvious nonsense briefs."""

from __future__ import annotations

from app.pipeline.contracts import BaseStage
from app.pipeline.response_writer import ResponseWriterStage
from app.schemas import (
    AIContext,
    ArbitrationResult,
    ArbitrationRuleHit,
    AssessmentRecommendation,
    AssessmentResult,
    AssessmentTechnicalInfo,
    CompletenessResult,
    DecisionStatus,
    ExtractedBrief,
    ExtractedFact,
    ExtractionResult,
    ExtractorTechnicalInfo,
    FactStatus,
)
from app.tracing.tracing import NoOpTracingClient, TracingClient

EMPTY_OR_NONSENSE_BRIEF_REASON_CODE = "empty_or_nonsense_brief"
EMPTY_OR_NONSENSE_BRIEF_RULE_KEY = "reject_empty_or_nonsense_brief"


class EmptyBriefRejectionStage(BaseStage[AIContext, AIContext]):
    """Reject empty or obvious nonsense briefs before any LLM call."""

    def __init__(self, tracing_client: TracingClient | None = None) -> None:
        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or NoOpTracingClient(),
        )
        self._response_writer = ResponseWriterStage(tracing_client=tracing_client)

    def run_context(self, context: AIContext) -> AIContext:
        return self.run(context)

    def _run(self, stage_input: AIContext) -> AIContext:
        if not is_empty_or_obvious_nonsense(stage_input.original_text):
            return stage_input

        context = (
            stage_input.with_extraction_result(_build_extraction_result())
            .with_completeness_result(CompletenessResult(is_complete=True))
            .with_assessment_result(_build_assessment_result())
            .with_arbitration_result(_build_arbitration_result())
            .with_stage_metadata(
                self.__class__.__name__,
                reason_code=EMPTY_OR_NONSENSE_BRIEF_REASON_CODE,
                short_circuit_pipeline=True,
            )
        )
        return self._response_writer.write_context(context)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        return exc


def is_empty_or_obvious_nonsense(text: str) -> bool:
    """Conservative detector; semantic non-briefs are intentionally out of scope."""
    normalized = _normalize_detector_text(text)
    if not normalized:
        return True

    alnum_chars = [char for char in normalized if char.isalnum()]
    if not alnum_chars:
        return True
    if len(alnum_chars) == 1:
        return True

    return False


def _normalize_detector_text(text: str) -> str:
    return text.strip().lower()


def _missing_fact() -> ExtractedFact:
    return ExtractedFact(status=FactStatus.missing, value=None, evidence=[])


def _build_extraction_result() -> ExtractionResult:
    extracted = ExtractedBrief(
        project_goal=_missing_fact(),
        tasks=[],
        project_type=_missing_fact(),
        project_direction=_missing_fact(),
        expected_result=_missing_fact(),
    )
    return ExtractionResult(
        extracted_brief=extracted,
        technical_info=ExtractorTechnicalInfo(
            attempts=0,
            prompt_name="empty_brief_rejection",
            trace_enabled=False,
            trace_name="empty_brief_rejection",
        ),
    )


def _build_assessment_result() -> AssessmentResult:
    return AssessmentResult(
        criterion_evaluations=[],
        risks=[],
        evidence=[],
        has_risks=False,
        recommendation=AssessmentRecommendation.high_risk_review,
        summary="Бриф не содержит осмысленной проектной заявки.",
        confidence=1.0,
        technical_info=AssessmentTechnicalInfo(
            attempts=0,
            prompt_name="empty_brief_rejection",
            trace_enabled=False,
            trace_name="empty_brief_rejection",
            model_name=None,
            criteria_count=0,
            risk_types_count=0,
            raw_response=None,
            recovered_errors=[],
        ),
    )


def _build_arbitration_result() -> ArbitrationResult:
    hit = ArbitrationRuleHit(
        rule_key=EMPTY_OR_NONSENSE_BRIEF_RULE_KEY,
        title="Reject empty or nonsense brief",
        status=DecisionStatus.reject,
        conditions=[
            f"reason_code == {EMPTY_OR_NONSENSE_BRIEF_REASON_CODE}",
        ],
        evidence=[],
        explanation="Бриф не содержит осмысленного проектного запроса.",
        confidence=1.0,
        metadata={"reason_code": EMPTY_OR_NONSENSE_BRIEF_REASON_CODE},
    )
    return ArbitrationResult(
        final_status=DecisionStatus.reject,
        reasons=["Бриф не содержит осмысленного проектного запроса."],
        evidence=[],
        triggered_rules=[hit],
        confidence=1.0,
        metadata={
            "matched_rule": EMPTY_OR_NONSENSE_BRIEF_RULE_KEY,
            "reason_code": EMPTY_OR_NONSENSE_BRIEF_REASON_CODE,
        },
    )
