"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from app.pipeline.assessment import (
    AssessmentConfigError,
    AssessmentError,
    AssessmentPreparation,
    AssessmentPreparedInput,
    AssessmentRetriever,
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
from app.pipeline.result_builder import (
    BriefAnalysisResultBuilder,
    BriefAnalysisResultError,
)
from app.pipeline.self_check import (
    DeterministicValidator,
    LLMSelfChecker,
    SelfCheckError,
    SelfChecker,
)
from app.pipeline.completeness import (
    CompletenessCheckStage,
    CompletenessConfigError,
    CompletenessError,
)
from app.pipeline.extractor import Extractor, ExtractorError

__all__ = [
    "AssessmentConfigError",
    "AssessmentError",
    "AssessmentPreparation",
    "AssessmentPreparedInput",
    "AssessmentRetriever",
    "AssessmentStage",
    "BaseLLMStage",
    "BaseStage",
    "LLMStageRunResult",
    "PipelineStage",
    "StageExecutionError",
    "CompletenessCheckStage",
    "CompletenessConfigError",
    "CompletenessError",
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
    "DeterministicValidator",
    "LLMSelfChecker",
    "SelfCheckError",
    "SelfChecker",
    "Extractor",
    "ExtractorError",
]
