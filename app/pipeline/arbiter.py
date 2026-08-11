"""Deterministic arbitration for project brief decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import (
    ArbitrationConfiguration,
    ArbitrationCondition,
    ArbitrationRule,
    CriteriaConfig,
    CriteriaConfigError,
    CriteriaLoader,
    get_criteria_config,
)
from app.pipeline.contracts import BaseStage
from app.schemas import (
    AIContext,
    ArbitrationResult,
    ArbitrationRuleHit,
    AssessmentResult,
    CompletenessResult,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    DecisionStatus,
    Risk,
    RiskSeverity,
)
from app.tracing.tracing import NoOpTracingClient, TracingClient


class ArbitrationError(RuntimeError):
    """Raised when deterministic arbitration cannot be completed."""


class ArbitrationConfigError(ArbitrationError):
    """Raised when arbitration configuration is missing or invalid."""


class DeterministicArbiterStage(BaseStage[AIContext, AIContext]):
    """Produce a final deterministic decision from prior pipeline results."""

    def __init__(
        self,
        criteria_config: CriteriaConfig | None = None,
        criteria_path: str | Path | None = None,
        tracing_client: TracingClient | None = None,
    ) -> None:
        """Initialize the arbiter with a validated configuration."""
        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or NoOpTracingClient(),
        )
        if criteria_config is not None and criteria_path is not None:
            raise ValueError("Pass either criteria_config or criteria_path, not both")

        if criteria_config is not None:
            config = criteria_config
        elif criteria_path is not None:
            try:
                config = CriteriaLoader.load(Path(criteria_path))
            except CriteriaConfigError as exc:
                raise ArbitrationConfigError(str(exc)) from exc
        else:
            try:
                config = get_criteria_config()
            except CriteriaConfigError as exc:
                raise ArbitrationConfigError(str(exc)) from exc

        arbitration = config.evaluation.arbitration
        if arbitration is None:
            raise ArbitrationConfigError(
                "criteria configuration is missing arbitration section"
            )
        if not arbitration.rules:
            raise ArbitrationConfigError("arbitration configuration has no rules")

        self._config = config
        self._arbitration = arbitration
        self._supported_signals = self._build_supported_signals()
        self._validate_config()

    def _arbitrate_from_parts(
        self,
        completeness_result: CompletenessResult,
        risks: list[Risk],
        criterion_evaluations: list[CriterionEvaluation],
    ) -> ArbitrationResult:
        """Return a deterministic final decision from active signal parts."""
        signals = self._build_signals(
            completeness_result=completeness_result,
            risks=risks,
            criterion_evaluations=criterion_evaluations,
        )
        evidence_map = self._build_evidence_map(
            completeness_result=completeness_result,
            risks=risks,
            criterion_evaluations=criterion_evaluations,
        )

        for rule in self._arbitration.rules:
            if self._rule_matches(rule, signals):
                return self._build_result(
                    rule=rule,
                    signals=signals,
                    evidence_map=evidence_map,
                )

        default_status = self._parse_status(self._arbitration.default_status)
        return ArbitrationResult(
            final_status=default_status,
            reasons=[
                "No arbitration rule matched; default status selected "
                "from configuration."
            ],
            evidence=[],
            triggered_rules=[],
            confidence=None,
            metadata={
                "configuration_version": self._arbitration.version,
                "signals": signals,
            },
        )

    def arbitrate_assessment(
        self,
        completeness_result: CompletenessResult,
        assessment_result: AssessmentResult,
    ) -> ArbitrationResult:
        """Return a deterministic decision from unified Assessment output."""
        return self._arbitrate_from_parts(
            completeness_result=completeness_result,
            risks=assessment_result.risks,
            criterion_evaluations=assessment_result.criterion_evaluations,
        )

    def arbitrate_context(self, context: AIContext) -> AIContext:
        """Return context enriched with arbitration based on available signals."""
        if context.completeness_result is None:
            raise ArbitrationError(
                "Arbitration requires completeness_result in AIContext"
            )

        if context.assessment_result is None:
            raise ArbitrationError(
                "Arbitration requires assessment_result in AIContext"
            )

        result = self.arbitrate_assessment(
            completeness_result=context.completeness_result,
            assessment_result=context.assessment_result,
        )
        return context.with_arbitration_result(result)

    def run_context(self, context: AIContext) -> AIContext:
        """Run this stage using the common AIContext pipeline contract."""
        return self.run(context)

    def _run(self, stage_input: AIContext) -> AIContext:
        """Run deterministic arbitration through the BaseStage lifecycle."""
        return self.arbitrate_context(stage_input)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Preserve arbitration-specific exceptions at the stage boundary."""
        return exc

    def _build_trace_input(self, stage_input: AIContext) -> dict[str, Any]:
        """Build safe arbitration trace metadata."""
        return {
            "has_assessment_result": stage_input.assessment_result is not None,
            "has_completeness_result": stage_input.completeness_result is not None,
        }

    def _build_trace_output(self, stage_output: AIContext) -> dict[str, Any]:
        """Build safe arbitration trace output."""
        result = stage_output.arbitration_result
        return {
            "status": "success",
            "final_status": result.final_status.value if result else None,
        }

    def _build_result(
        self,
        rule: ArbitrationRule,
        signals: dict[str, Any],
        evidence_map: dict[str, list[str]],
    ) -> ArbitrationResult:
        """Convert a matched rule into a final arbitration result."""
        evidence = self._collect_evidence(rule, evidence_map)
        condition_texts = [
            self._render_condition(condition) for condition in rule.conditions
        ]
        hit = ArbitrationRuleHit(
            rule_key=rule.key,
            title=rule.title,
            status=self._parse_status(rule.status),
            conditions=condition_texts,
            evidence=evidence,
            explanation=rule.description,
            confidence=rule.confidence,
            metadata={
                "signals": {
                    condition.field: signals.get(condition.field)
                    for condition in rule.conditions
                }
            },
        )
        return ArbitrationResult(
            final_status=hit.status,
            reasons=self._build_reasons(rule, signals),
            evidence=evidence,
            triggered_rules=[hit],
            confidence=rule.confidence,
            metadata={
                "configuration_version": self._arbitration.version,
                "matched_rule": rule.key,
                "signals": signals,
            },
        )

    def _build_reasons(
        self,
        rule: ArbitrationRule,
        signals: dict[str, Any],
    ) -> list[str]:
        """Build human-readable reasons for a matched rule."""
        rendered_conditions = [
            self._render_condition(condition) for condition in rule.conditions
        ]
        reasons = [rule.description]
        reasons.append(f"Matched conditions: {'; '.join(rendered_conditions)}")
        reasons.append(f"Signals: {self._format_signals(rule.conditions, signals)}")
        return reasons

    def _rule_matches(
        self,
        rule: ArbitrationRule,
        signals: dict[str, Any],
    ) -> bool:
        """Check whether all conditions in a rule match the signal snapshot."""
        return all(
            self._condition_matches(condition, signals.get(condition.field))
            for condition in rule.conditions
        )

    def _condition_matches(
        self,
        condition: ArbitrationCondition,
        signal_value: Any,
    ) -> bool:
        """Evaluate a single arbitration condition."""
        operator = condition.operator.lower()
        target = condition.value

        if operator == "exists":
            return signal_value is not None
        if operator == "not_exists":
            return signal_value is None

        left = self._normalize_value(signal_value, condition.case_sensitive)
        right = self._normalize_value(target, condition.case_sensitive)

        if operator == "eq":
            return left == right
        if operator == "ne":
            return left != right
        if operator == "gt":
            return self._compare_numbers(left, right, ">")
        if operator == "gte":
            return self._compare_numbers(left, right, ">=")
        if operator == "lt":
            return self._compare_numbers(left, right, "<")
        if operator == "lte":
            return self._compare_numbers(left, right, "<=")
        if operator == "in":
            return self._membership(left, right)
        if operator == "not_in":
            return not self._membership(left, right)
        if operator == "contains":
            return self._contains(left, right)
        if operator == "any_in":
            return self._any_in(left, right)
        if operator == "all_in":
            return self._all_in(left, right)

        raise ArbitrationConfigError(f"Unsupported arbitration operator: {operator}")

    def _build_signals(
        self,
        completeness_result: CompletenessResult,
        risks: list[Risk],
        criterion_evaluations: list[CriterionEvaluation],
    ) -> dict[str, Any]:
        """Build the deterministic signal snapshot used by the rule engine."""
        risk_counts = self._risk_counts(risks)
        evaluation_counts = self._evaluation_counts(criterion_evaluations)
        completeness_counts = self._completeness_counts(completeness_result)

        signals: dict[str, Any] = {
            "completeness.is_complete": completeness_result.is_complete,
            "completeness.missing_count": completeness_counts["missing_count"],
            "completeness.clarification_count": completeness_counts[
                "clarification_count"
            ],
            "completeness.present_count": completeness_counts["present_count"],
            "risk.has_risks": bool(risks),
            "risk.total_count": risk_counts["total_count"],
            "risk.low_count": risk_counts["low_count"],
            "risk.medium_count": risk_counts["medium_count"],
            "risk.high_count": risk_counts["high_count"],
            "risk.critical_count": risk_counts["critical_count"],
            "risk.max_severity": risk_counts["max_severity"],
            "evaluation.total_count": evaluation_counts["total_count"],
            "evaluation.met_count": evaluation_counts["met_count"],
            "evaluation.not_met_count": evaluation_counts["not_met_count"],
            "evaluation.insufficient_information_count": evaluation_counts[
                "insufficient_information_count"
            ],
            "evaluation.risk_detected_count": evaluation_counts[
                "risk_detected_count"
            ],
        }
        return signals

    def _build_evidence_map(
        self,
        completeness_result: CompletenessResult,
        risks: list[Risk],
        criterion_evaluations: list[CriterionEvaluation],
    ) -> dict[str, list[str]]:
        """Build evidence fragments for each supported signal."""
        evidence_map: dict[str, list[str]] = {
            "completeness.is_complete": self._present_field_evidence(
                completeness_result
            ),
            "completeness.missing_count": [
                f"{item.title}: {item.reason or item.field_path}"
                for item in completeness_result.missing_information
            ],
            "completeness.clarification_count": [
                f"{item.title}: {item.reason or item.field_path}"
                for item in completeness_result.clarification_information
            ],
            "completeness.present_count": [
                item.title for item in completeness_result.present_information
            ],
            "risk.has_risks": [
                f"{risk.type}: {risk.description}"
                for risk in risks
            ],
            "risk.total_count": self._risk_evidence(
                risks,
                {"low", "medium", "high", "critical"},
            ),
            "risk.low_count": self._risk_evidence(risks, {"low"}),
            "risk.medium_count": self._risk_evidence(risks, {"medium"}),
            "risk.high_count": self._risk_evidence(risks, {"high"}),
            "risk.critical_count": self._risk_evidence(
                risks,
                {"critical"},
            ),
            "risk.max_severity": self._risk_evidence(
                risks,
                {self._risk_max_severity(risks)},
            ),
            "evaluation.total_count": [
                f"{item.criterion}: {item.explanation or item.criterion}"
                for item in criterion_evaluations
            ],
            "evaluation.met_count": self._evaluation_evidence(
                criterion_evaluations,
                {CriterionEvaluationStatus.met},
            ),
            "evaluation.not_met_count": self._evaluation_evidence(
                criterion_evaluations,
                {CriterionEvaluationStatus.not_met},
            ),
            "evaluation.insufficient_information_count": self._evaluation_evidence(
                criterion_evaluations,
                {CriterionEvaluationStatus.insufficient_information},
            ),
            "evaluation.risk_detected_count": self._evaluation_evidence(
                criterion_evaluations,
                {CriterionEvaluationStatus.risk_detected},
            ),
        }
        return evidence_map

    def _validate_config(self) -> None:
        """Validate arbitration configuration semantics."""
        self._parse_status(self._arbitration.default_status)

        for rule in self._arbitration.rules:
            self._parse_status(rule.status)
            for condition in rule.conditions:
                if condition.field not in self._supported_signals:
                    raise ArbitrationConfigError(
                        f"Unsupported arbitration signal: {condition.field}"
                    )

    def _build_supported_signals(self) -> set[str]:
        """Return the fixed set of signals produced by this arbiter."""
        return {
            "completeness.is_complete",
            "completeness.missing_count",
            "completeness.clarification_count",
            "completeness.present_count",
            "risk.has_risks",
            "risk.total_count",
            "risk.low_count",
            "risk.medium_count",
            "risk.high_count",
            "risk.critical_count",
            "risk.max_severity",
            "evaluation.total_count",
            "evaluation.met_count",
            "evaluation.not_met_count",
            "evaluation.insufficient_information_count",
            "evaluation.risk_detected_count",
        }

    def _collect_evidence(
        self,
        rule: ArbitrationRule,
        evidence_map: dict[str, list[str]],
    ) -> list[str]:
        """Collect deduplicated evidence for a matched rule."""
        collected: list[str] = []
        for condition in rule.conditions:
            for fragment in evidence_map.get(condition.field, []):
                if fragment not in collected:
                    collected.append(fragment)
        return collected

    @staticmethod
    def _format_signals(
        conditions: list[ArbitrationCondition],
        signals: dict[str, Any],
    ) -> str:
        """Render the signals referenced by a rule."""
        parts = []
        for condition in conditions:
            parts.append(f"{condition.field}={signals.get(condition.field)!r}")
        return ", ".join(parts)

    @staticmethod
    def _render_condition(condition: ArbitrationCondition) -> str:
        """Render a human-readable condition string."""
        return (
            f"{condition.field} {condition.operator} "
            f"{condition.value!r}"
        )

    @staticmethod
    def _normalize_value(value: Any, case_sensitive: bool) -> Any:
        """Normalize values before comparison."""
        if isinstance(value, str) and not case_sensitive:
            return value.strip().lower()
        if isinstance(value, list):
            return [
                DeterministicArbiterStage._normalize_value(item, case_sensitive)
                for item in value
            ]
        return value

    @staticmethod
    def _compare_numbers(left: Any, right: Any, operator: str) -> bool:
        """Compare numeric values using the requested operator."""
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return False
        if operator == ">":
            return left > right
        if operator == ">=":
            return left >= right
        if operator == "<":
            return left < right
        if operator == "<=":
            return left <= right
        return False

    @staticmethod
    def _membership(left: Any, right: Any) -> bool:
        """Check whether the left value is a member of the right sequence."""
        if not isinstance(right, list):
            right = [right]

        if isinstance(left, list):
            return any(item in right for item in left)

        return left in right

    @staticmethod
    def _contains(left: Any, right: Any) -> bool:
        """Check containment for strings and lists."""
        if isinstance(left, str) and isinstance(right, str):
            return right in left
        if isinstance(left, list):
            return right in left
        return False

    @staticmethod
    def _any_in(left: Any, right: Any) -> bool:
        """Check whether any item from left exists in right."""
        if not isinstance(left, list):
            return False
        if not isinstance(right, list):
            right = [right]
        return any(item in right for item in left)

    @staticmethod
    def _all_in(left: Any, right: Any) -> bool:
        """Check whether all items from left exist in right."""
        if not isinstance(left, list):
            return False
        if not isinstance(right, list):
            right = [right]
        return all(item in right for item in left)

    @staticmethod
    def _parse_status(status: str) -> DecisionStatus:
        """Parse a configured status string into the enum."""
        try:
            return DecisionStatus(status)
        except ValueError as exc:
            raise ArbitrationConfigError(
                f"Unsupported decision status: {status}"
            ) from exc

    @staticmethod
    def _risk_counts(risks: list[Risk]) -> dict[str, Any]:
        """Count risk severities."""
        counts = {
            "total_count": len(risks),
            "low_count": 0,
            "medium_count": 0,
            "high_count": 0,
            "critical_count": 0,
            "max_severity": "none",
        }
        severity_rank = {
            RiskSeverity.low: 1,
            RiskSeverity.medium: 2,
            RiskSeverity.high: 3,
            RiskSeverity.critical: 4,
        }
        highest = 0
        for risk in risks:
            counts[f"{risk.severity.value}_count"] += 1
            rank = severity_rank[risk.severity]
            if rank > highest:
                highest = rank
                counts["max_severity"] = risk.severity.value
        return counts

    @staticmethod
    def _risk_max_severity(risks: list[Risk]) -> str:
        """Return the maximum risk severity as a string."""
        if not risks:
            return "none"

        severity_rank = {
            RiskSeverity.low: 1,
            RiskSeverity.medium: 2,
            RiskSeverity.high: 3,
            RiskSeverity.critical: 4,
        }
        highest = max(
            risks,
            key=lambda risk: severity_rank[risk.severity],
        )
        return highest.severity.value

    @staticmethod
    def _risk_evidence(
        risks: list[Risk],
        severities: set[str],
    ) -> list[str]:
        """Collect evidence fragments for risks matching the given severities."""
        evidence: list[str] = []
        for risk in risks:
            if risk.severity.value not in severities:
                continue
            for fragment in risk.evidence:
                if fragment not in evidence:
                    evidence.append(fragment)
        return evidence

    @staticmethod
    def _evaluation_counts(
        criterion_evaluations: list[CriterionEvaluation],
    ) -> dict[str, int]:
        """Count criterion evaluation outcomes."""
        counts = {
            "total_count": len(criterion_evaluations),
            "met_count": 0,
            "not_met_count": 0,
            "insufficient_information_count": 0,
            "risk_detected_count": 0,
        }
        for item in criterion_evaluations:
            counts[f"{item.status.value}_count"] += 1
        return counts

    @staticmethod
    def _evaluation_evidence(
        criterion_evaluations: list[CriterionEvaluation],
        statuses: set[CriterionEvaluationStatus],
    ) -> list[str]:
        """Collect evidence fragments for matching criterion evaluations."""
        evidence: list[str] = []
        for item in criterion_evaluations:
            if item.status not in statuses:
                continue
            fragments = item.evidence or []
            if not fragments:
                fragments = [item.explanation or item.criterion]
            for fragment in fragments:
                if fragment not in evidence:
                    evidence.append(fragment)
        return evidence

    @staticmethod
    def _completeness_counts(
        completeness_result: CompletenessResult,
    ) -> dict[str, int]:
        """Count completeness states."""
        return {
            "present_count": len(completeness_result.present_information),
            "missing_count": len(completeness_result.missing_information),
            "clarification_count": len(completeness_result.clarification_information),
        }

    @staticmethod
    def _present_field_evidence(
        completeness_result: CompletenessResult,
    ) -> list[str]:
        """Collect evidence-like references for present completeness fields."""
        return [item.title for item in completeness_result.present_information]
