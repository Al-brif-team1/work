"""Модуль структур данных для конвейера анализа брифов. Эти модели помогают хранить информацию аккуратно, чтобы этапы не перепутали факты, статусы и технические детали."""

from app.schemas.assessment import (
    AssessmentEvidence,
    AssessmentPayload,
    AssessmentRecommendation,
    AssessmentResult,
    AssessmentTechnicalInfo,
)
from app.schemas.ai_context import (
    AIContext,
    PipelineInputState,
    PipelineResults,
    PipelineTechnicalState,
    ResponseState,
    RetrievalState,
)
from app.schemas.brief import BriefInput, BriefInputMetadata
from app.schemas.evaluation import (
    CriterionEvaluation,
    CriterionEvaluationStatus,
)
from app.schemas.decision import ArbitrationResult, ArbitrationRuleHit, DecisionStatus
from app.schemas.completeness import (
    CompletenessLevel,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    CompletenessTechnicalInfo,
)
from app.schemas.question import (
    ClarificationQuestion,
    QuestionGenerationResult,
    QuestionGenerationTechnicalInfo,
)
from app.schemas.mvp import MVPPlan, MVPPlanningResult, MVPPlanningTechnicalInfo
from app.schemas.final_result import (
    BriefAnalysisResult,
    BriefAssessmentSummary,
    BriefExtractedFields,
    DirectionValue,
)
from app.schemas.self_check import (
    SelfCheckContext,
    SelfCheckPayload,
    SelfCheckResult,
    SelfCheckTechnicalInfo,
)
from app.schemas.extraction import (
    ExtractedBrief,
    ExtractedFact,
    ExtractionResult,
    ExtractorTechnicalInfo,
    FactStatus,
)
from app.schemas.knowledge import Document, DocumentMetadata, SearchResult
from app.schemas.risk import (
    Risk,
    RiskSeverity,
)
from app.schemas.traffic_light import (
    TrafficLightMatch,
    TrafficLightResult,
    TrafficLightStatus,
)

__all__ = [
    "AIContext",
    "PipelineInputState",
    "PipelineResults",
    "PipelineTechnicalState",
    "ResponseState",
    "RetrievalState",
    "AssessmentEvidence",
    "AssessmentPayload",
    "AssessmentRecommendation",
    "AssessmentResult",
    "AssessmentTechnicalInfo",
    "BriefInput",
    "BriefInputMetadata",
    "CriterionEvaluation",
    "CriterionEvaluationStatus",
    "ArbitrationResult",
    "ArbitrationRuleHit",
    "DecisionStatus",
    "ClarificationQuestion",
    "QuestionGenerationResult",
    "QuestionGenerationTechnicalInfo",
    "MVPPlan",
    "MVPPlanningResult",
    "MVPPlanningTechnicalInfo",
    "BriefAnalysisResult",
    "BriefAssessmentSummary",
    "BriefExtractedFields",
    "DirectionValue",
    "SelfCheckContext",
    "SelfCheckPayload",
    "SelfCheckResult",
    "SelfCheckTechnicalInfo",
    "CompletenessItem",
    "CompletenessLevel",
    "CompletenessResult",
    "CompletenessStatus",
    "CompletenessTechnicalInfo",
    "ExtractedBrief",
    "ExtractedFact",
    "ExtractionResult",
    "ExtractorTechnicalInfo",
    "FactStatus",
    "Document",
    "DocumentMetadata",
    "SearchResult",
    "Risk",
    "RiskSeverity",
    "TrafficLightMatch",
    "TrafficLightResult",
    "TrafficLightStatus",
]
