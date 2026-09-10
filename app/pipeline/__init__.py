"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from app.pipeline.assessment import (
    AssessmentConfigError,
    AssessmentError,
    AssessmentPreparation,
    AssessmentPreparedInput,
    AssessmentStage,
)
from app.pipeline.base import BaseLLMStage, LLMStageRunResult
from app.pipeline.contracts import BaseStage, PipelineStage, StageExecutionError
from app.pipeline.arbiter import (
    ArbitrationConfigError,
    ArbitrationError,
    DeterministicArbiterStage,
)
from app.pipeline.question_generator import (
    QuestionGenerationError,
    QuestionGeneratorConfigError,
    TemplateQuestionGeneratorStage,
)
from app.pipeline.mvp_planner import MVPPlannerError, MVPPlannerStage
from app.pipeline.orchestrator import BriefAnalysisPipeline, BriefAnalysisPipelineError
from app.pipeline.response_writer import ResponseWriterError, ResponseWriterStage
from app.pipeline.security import (
    PROMPT_INJECTION_REASON_CODE,
    PROMPT_INJECTION_RULE_KEY,
    SecurityGateStage,
)
from app.pipeline.result_builder import (
    BriefAnalysisResultBuilder,
    BriefAnalysisResultError,
)
from app.pipeline.completeness import (
    CompletenessCheckStage,
    CompletenessConfigError,
    CompletenessError,
)
from app.pipeline.empty_brief import (
    EMPTY_OR_NONSENSE_BRIEF_REASON_CODE,
    EMPTY_OR_NONSENSE_BRIEF_RULE_KEY,
    EmptyBriefRejectionStage,
    is_empty_or_obvious_nonsense,
)
from app.pipeline.extractor import Extractor, ExtractorError

__all__ = [
    "AssessmentConfigError",
    "AssessmentError",
    "AssessmentPreparation",
    "AssessmentPreparedInput",
    "AssessmentStage",
    "BaseLLMStage",
    "BaseStage",
    "LLMStageRunResult",
    "PipelineStage",
    "StageExecutionError",
    "CompletenessCheckStage",
    "CompletenessConfigError",
    "CompletenessError",
    "EMPTY_OR_NONSENSE_BRIEF_REASON_CODE",
    "EMPTY_OR_NONSENSE_BRIEF_RULE_KEY",
    "EmptyBriefRejectionStage",
    "is_empty_or_obvious_nonsense",
    "ArbitrationConfigError",
    "ArbitrationError",
    "DeterministicArbiterStage",
    "QuestionGenerationError",
    "QuestionGeneratorConfigError",
    "TemplateQuestionGeneratorStage",
    "MVPPlannerStage",
    "MVPPlannerError",
    "BriefAnalysisPipeline",
    "BriefAnalysisPipelineError",
    "BriefAnalysisResultBuilder",
    "BriefAnalysisResultError",
    "ResponseWriterError",
    "ResponseWriterStage",
    "PROMPT_INJECTION_REASON_CODE",
    "PROMPT_INJECTION_RULE_KEY",
    "SecurityGateStage",
    "Extractor",
    "ExtractorError",
]
