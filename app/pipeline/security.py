"""Security gate for brief analysis pipeline."""

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
    PipelineInputState,
)
from app.security import InjectionDetector, PIISanitizer, SecurityPipeline
from app.security.pipeline import SecurityPipelineResult
from app.tracing.tracing import NoOpTracingClient, TracingClient

PROMPT_INJECTION_REASON_CODE = "prompt_injection_detected"
PROMPT_INJECTION_RULE_KEY = "reject_prompt_injection"


class SecurityGateStage(BaseStage[AIContext, AIContext]):
    """Run security checks before the first LLM stage."""

    def __init__(
        self,
        injection_detector: InjectionDetector | None = None,
        tracing_client: TracingClient | None = None,
    ) -> None:
        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or NoOpTracingClient(),
        )
        self._injection_detector = injection_detector or InjectionDetector()
        self._response_writer = ResponseWriterStage(tracing_client=tracing_client)

    def run_context(self, context: AIContext) -> AIContext:
        return self.run(context)

    def _run(self, stage_input: AIContext) -> AIContext:
        security_result = SecurityPipeline(
            pii_sanitizer=PIISanitizer(),
            injection_detector=self._injection_detector,
        ).process(stage_input.normalized_text)

        context = self._with_security_metadata(stage_input, security_result)
        if security_result.status == "blocked":
            return self._short_circuit(context, security_result)

        return self._with_sanitized_brief_input(context, security_result.sanitized_text)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        return exc

    def _with_security_metadata(
        self,
        context: AIContext,
        security_result: SecurityPipelineResult,
    ) -> AIContext:
        return context.with_stage_metadata(
            self.__class__.__name__,
            status=security_result.status,
            safe=security_result.safe,
            risk_level=security_result.injection_result.risk_level,
            found_patterns=list(security_result.injection_result.found_patterns),
            warnings=list(security_result.warnings),
            error=security_result.error,
            restoration_map=security_result.restoration_map,
        )

    @staticmethod
    def _with_sanitized_brief_input(context: AIContext, sanitized_text: str) -> AIContext:
        brief_input = context.brief_input.model_copy(
            update={"normalized_text": sanitized_text}
        )
        return context.model_copy(
            update={
                "inputs": PipelineInputState(brief_input=brief_input),
            }
        )

    def _short_circuit(
        self,
        context: AIContext,
        security_result: SecurityPipelineResult,
    ) -> AIContext:
        context = (
            context.with_extraction_result(_build_extraction_result())
            .with_completeness_result(CompletenessResult(is_complete=True))
            .with_assessment_result(_build_assessment_result(security_result))
            .with_arbitration_result(_build_arbitration_result(security_result))
            .with_stage_metadata(
                self.__class__.__name__,
                reason_code=PROMPT_INJECTION_REASON_CODE,
                short_circuit_pipeline=True,
            )
        )
        return self._response_writer.write_context(context)


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
            prompt_name=PROMPT_INJECTION_REASON_CODE,
            trace_enabled=False,
            trace_name=PROMPT_INJECTION_REASON_CODE,
        ),
    )


def _build_assessment_result(
    security_result: SecurityPipelineResult,
) -> AssessmentResult:
    return AssessmentResult(
        criterion_evaluations=[],
        risks=[],
        evidence=[],
        has_risks=False,
        recommendation=AssessmentRecommendation.high_risk_review,
        summary="Бриф содержит инструкцию, похожую на попытку вмешаться в работу системы.",
        confidence=1.0,
        technical_info=AssessmentTechnicalInfo(
            attempts=0,
            prompt_name=PROMPT_INJECTION_REASON_CODE,
            trace_enabled=False,
            trace_name=PROMPT_INJECTION_REASON_CODE,
            model_name=None,
            retriever_used=False,
            retrieved_context_count=0,
            criteria_count=0,
            risk_types_count=0,
            raw_response={
                "risk_level": security_result.injection_result.risk_level,
                "found_patterns": list(security_result.injection_result.found_patterns),
            },
            recovered_errors=[],
        ),
    )


def _build_arbitration_result(
    security_result: SecurityPipelineResult,
) -> ArbitrationResult:
    hit = ArbitrationRuleHit(
        rule_key=PROMPT_INJECTION_RULE_KEY,
        title="Reject prompt injection",
        status=DecisionStatus.reject,
        conditions=[f"reason_code == {PROMPT_INJECTION_REASON_CODE}"],
        evidence=list(security_result.injection_result.found_patterns),
        explanation="Бриф содержит инструкцию, похожую на попытку вмешаться в работу системы.",
        confidence=1.0,
        metadata={
            "reason_code": PROMPT_INJECTION_REASON_CODE,
            "risk_level": security_result.injection_result.risk_level,
        },
    )
    return ArbitrationResult(
        final_status=DecisionStatus.reject,
        reasons=["Бриф содержит инструкцию, похожую на попытку вмешаться в работу системы."],
        evidence=list(security_result.injection_result.found_patterns),
        triggered_rules=[hit],
        confidence=1.0,
        metadata={
            "matched_rule": PROMPT_INJECTION_RULE_KEY,
            "reason_code": PROMPT_INJECTION_REASON_CODE,
        },
    )
