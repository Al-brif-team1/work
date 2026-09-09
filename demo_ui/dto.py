"""Demo UI DTO adapter for completed brief analysis contexts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.schemas import AIContext, RiskSeverity

_RISK_SEVERITY_RANK = {
    RiskSeverity.low.value: 1,
    RiskSeverity.medium.value: 2,
    RiskSeverity.high.value: 3,
    RiskSeverity.critical.value: 4,
}

_SENSITIVE_METADATA_KEYS = {
    "restoration_map",
}

_RAW_TECHNICAL_KEYS = {
    "raw_response",
}


def build_demo_response(context: AIContext) -> dict[str, Any]:
    """Build a JSON-ready DTO for Demo UI from an already completed AIContext."""
    short_circuit, short_circuit_stage = _short_circuit_info(context)
    return {
        "ok": True,
        "decision": _decision_section(
            context,
            short_circuit=short_circuit,
            short_circuit_stage=short_circuit_stage,
        ),
        "extraction": _extraction_section(context, synthetic=short_circuit),
        "completeness": _completeness_section(context, synthetic=short_circuit),
        "assessment": _assessment_section(context, synthetic=short_circuit),
        "traffic_light": _traffic_light_section(context, synthetic=short_circuit),
        "risks": _risks_section(context, synthetic=short_circuit),
        "arbiter": _arbiter_section(context),
        "questions": _questions_section(context),
        "mvp": _mvp_section(context),
        "customer_response": {
            "text": context.final_response_text,
        },
        "technical": _technical_section(context),
    }


def _decision_section(
    context: AIContext,
    *,
    short_circuit: bool,
    short_circuit_stage: str | None,
) -> dict[str, Any]:
    arbitration = context.arbitration_result
    public_payload = context.final_response_payload or {}
    if arbitration is None:
        return {
            "available": False,
            "final_status": None,
            "summary": public_payload.get("summary"),
            "reasons": [],
            "evidence": [],
            "confidence": None,
            "matched_rule": None,
            "signals": {},
            "triggered_rules": [],
            "short_circuit": short_circuit,
            "short_circuit_stage": short_circuit_stage,
        }

    metadata = _json_ready(arbitration.metadata)
    return {
        "available": True,
        "final_status": arbitration.final_status.value,
        "summary": public_payload.get("summary"),
        "reasons": list(arbitration.reasons),
        "evidence": list(arbitration.evidence),
        "confidence": arbitration.confidence,
        "matched_rule": metadata.get("matched_rule") if isinstance(metadata, dict) else None,
        "signals": metadata.get("signals", {}) if isinstance(metadata, dict) else {},
        "triggered_rules": [_model_dump(rule) for rule in arbitration.triggered_rules],
        "short_circuit": short_circuit,
        "short_circuit_stage": short_circuit_stage,
    }


def _extraction_section(context: AIContext, *, synthetic: bool) -> dict[str, Any]:
    result = context.extraction_result
    if result is None:
        return {
            "available": False,
            "synthetic": False,
            "extracted_brief": None,
            "technical_info": None,
        }
    return {
        "available": True,
        "synthetic": synthetic,
        "extracted_brief": _model_dump(result.extracted_brief),
        "technical_info": _technical_info(result.technical_info),
    }


def _completeness_section(context: AIContext, *, synthetic: bool) -> dict[str, Any]:
    result = context.completeness_result
    if result is None:
        return {
            "available": False,
            "synthetic": False,
            "is_complete": None,
            "level": None,
            "missing_information": [],
            "critical_missing_information": [],
            "optional_missing_information": [],
            "present_information": [],
            "clarification_information": [],
            "warnings": [],
            "technical_info": None,
        }
    return {
        "available": True,
        "synthetic": synthetic,
        "is_complete": result.is_complete,
        "level": result.level.value,
        "missing_information": [_model_dump(item) for item in result.missing_information],
        "critical_missing_information": [
            _model_dump(item) for item in result.critical_missing_information
        ],
        "optional_missing_information": [
            _model_dump(item) for item in result.optional_missing_information
        ],
        "present_information": [_model_dump(item) for item in result.present_information],
        "clarification_information": [
            _model_dump(item) for item in result.clarification_information
        ],
        "warnings": list(result.warnings),
        "technical_info": _technical_info(result.technical_info),
    }


def _assessment_section(context: AIContext, *, synthetic: bool) -> dict[str, Any]:
    result = context.assessment_result
    if result is None:
        return {
            "available": False,
            "synthetic": False,
            "criterion_evaluations": [],
            "evidence": [],
            "recommendation": None,
            "summary": None,
            "confidence": None,
            "technical_info": None,
        }
    return {
        "available": True,
        "synthetic": synthetic,
        "criterion_evaluations": [
            _model_dump(item) for item in result.criterion_evaluations
        ],
        "evidence": [_model_dump(item) for item in result.evidence],
        "recommendation": result.recommendation.value,
        "summary": result.summary,
        "confidence": result.confidence,
        "technical_info": _technical_info(result.technical_info),
    }


def _traffic_light_section(context: AIContext, *, synthetic: bool) -> dict[str, Any]:
    assessment = context.assessment_result
    if assessment is None:
        return {
            "available": False,
            "synthetic": False,
            "status": None,
            "direction": None,
            "specialization": None,
            "reason": None,
            "matches": [],
        }

    result = assessment.traffic_light
    return {
        "available": True,
        "synthetic": synthetic,
        "status": result.status.value,
        "direction": result.direction,
        "specialization": result.specialization,
        "reason": result.reason,
        "matches": [_model_dump(match) for match in result.matches],
    }


def _risks_section(context: AIContext, *, synthetic: bool) -> dict[str, Any]:
    risks = list(context.assessment_result.risks) if context.assessment_result else []
    items = [_model_dump(risk) for risk in risks]
    return {
        "available": context.assessment_result is not None,
        "synthetic": synthetic if context.assessment_result is not None else False,
        "items": items,
        "count": len(items),
        "max_severity": _max_risk_severity(items),
    }


def _arbiter_section(context: AIContext) -> dict[str, Any]:
    result = context.arbitration_result
    if result is None:
        return {
            "available": False,
            "final_status": None,
            "reasons": [],
            "evidence": [],
            "confidence": None,
            "metadata": {},
            "triggered_rules": [],
        }
    return {
        "available": True,
        "final_status": result.final_status.value,
        "reasons": list(result.reasons),
        "evidence": list(result.evidence),
        "confidence": result.confidence,
        "metadata": _json_ready(result.metadata),
        "triggered_rules": [_model_dump(rule) for rule in result.triggered_rules],
    }


def _questions_section(context: AIContext) -> dict[str, Any]:
    result = context.clarification_result
    if result is None:
        return {
            "available": False,
            "items": [],
            "summary": None,
            "technical_info": None,
        }
    return {
        "available": True,
        "items": [_model_dump(question) for question in result.questions],
        "summary": result.summary,
        "technical_info": _technical_info(result.technical_info),
    }


def _mvp_section(context: AIContext) -> dict[str, Any]:
    result = context.mvp_planning_result
    if result is None:
        return {
            "available": False,
            "plan": None,
            "technical_info": None,
            "skipped_reason": None,
        }
    return {
        "available": True,
        "plan": _model_dump(result.plan) if result.plan is not None else None,
        "technical_info": _technical_info(result.technical_info),
        "skipped_reason": result.technical_info.skipped_reason,
    }


def _technical_section(context: AIContext) -> dict[str, Any]:
    return {
        "stage_metadata": _safe_stage_metadata(context.stage_metadata),
        "llm": {
            "extractor": _technical_info(
                context.extraction_result.technical_info
                if context.extraction_result is not None
                else None
            ),
            "assessment": _technical_info(
                context.assessment_result.technical_info
                if context.assessment_result is not None
                else None
            ),
            "mvp_planner": _technical_info(
                context.mvp_planning_result.technical_info
                if context.mvp_planning_result is not None
                else None
            ),
        },
    }


def _short_circuit_info(context: AIContext) -> tuple[bool, str | None]:
    for stage_name, metadata in context.stage_metadata.items():
        if metadata.get("short_circuit_pipeline") is True:
            return True, stage_name
    return False, None


def _max_risk_severity(items: list[dict[str, Any]]) -> str | None:
    severities = [
        item.get("severity")
        for item in items
        if item.get("severity") in _RISK_SEVERITY_RANK
    ]
    if not severities:
        return None
    return max(severities, key=lambda severity: _RISK_SEVERITY_RANK[severity])


def _technical_info(value: BaseModel | None) -> dict[str, Any] | None:
    if value is None:
        return None
    dumped = _model_dump(value)
    for key in _RAW_TECHNICAL_KEYS:
        dumped.pop(key, None)
    return dumped


def _safe_stage_metadata(stage_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for stage_name, metadata in stage_metadata.items():
        safe[stage_name] = {
            key: _json_ready(value)
            for key, value in metadata.items()
            if key not in _SENSITIVE_METADATA_KEYS
        }
    return safe


def _model_dump(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value
