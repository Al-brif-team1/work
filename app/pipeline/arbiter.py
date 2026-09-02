"""Модуль этапа конвейера ИИ-ассистента для анализа проектных брифов. Здесь код работает как участок большого завода: каждый класс отвечает за свою роль и передает результат дальше."""

from __future__ import annotations

import sys
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
    CompletenessStatus,
    CriterionEvaluation,
    CriterionEvaluationStatus,
    DecisionStatus,
    Risk,
    RiskSeverity,
)
from app.tracing.tracing import NoOpTracingClient, TracingClient


class ArbitrationError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class ArbitrationConfigError(ArbitrationError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class DeterministicArbiterStage(BaseStage[AIContext, AIContext]):
    """[РОЛЬ В КОНВЕЙЕРЕ] Этот класс - чертеж конкретного робота-сотрудника: Робот-судья. Он обычным детерминированным кодом применяет правила и выносит вердикт, чтобы исключить галлюцинации ИИ. Этот этап работает как детерминированный робот: обычный код, без творческих догадок ИИ. [НАСЛЕДОВАНИЕ] Этот робот строится на базе общего шаблона BaseStage, поэтому он умеет работать в нашем конвейере."""

    def __init__(
        self,
        criteria_config: CriteriaConfig | None = None,
        criteria_path: str | Path | None = None,
        tracing_client: TracingClient | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
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
        self._known_risk_types = {
            risk_type.key
            for risk_type in self._config.evaluation.risk_analysis.risk_types
        }
        self._supported_signals = self._build_supported_signals()
        self._validate_config()

    def _arbitrate_from_parts(
        self,
        completeness_result: CompletenessResult,
        risks: list[Risk],
        criterion_evaluations: list[CriterionEvaluation],
    ) -> ArbitrationResult:
        """Выполняет шаг «arbitrate from parts». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        diagnostics = self._build_arbitration_diagnostics(
            completeness_result=completeness_result,
            signals=signals,
        )
        self._print_arbitration_diagnostics(
            diagnostics=diagnostics,
            matched_rule_key=None,
        )

        for rule in self._arbitration.rules:
            if self._rule_matches(rule, signals):
                self._print_arbitration_diagnostics(
                    diagnostics=diagnostics,
                    matched_rule_key=rule.key,
                )
                return self._build_result(
                    rule=rule,
                    signals=signals,
                    evidence_map=evidence_map,
                )

        default_status = self._parse_status(self._arbitration.default_status)
        self._print_arbitration_diagnostics(
            diagnostics=diagnostics,
            matched_rule_key=None,
            default_status=default_status.value,
        )
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
        """Выполняет шаг «arbitrate assessment». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._arbitrate_from_parts(
            completeness_result=completeness_result,
            risks=assessment_result.risks,
            criterion_evaluations=assessment_result.criterion_evaluations,
        )

    def arbitrate_context(self, context: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
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
        """[ЗАПУСК РОБОТА] Запускает этап на общем AIContext. Так каждый робот получает одну и ту же коробку с деталями конструктора, добавляет свой результат и передает ее дальше."""
        return self.run(context)

    def _run(self, stage_input: AIContext) -> AIContext:
        """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""
        return self.arbitrate_context(stage_input)

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return exc

    def _build_trace_input(self, stage_input: AIContext) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return {
            "has_assessment_result": stage_input.assessment_result is not None,
            "has_completeness_result": stage_input.completeness_result is not None,
        }

    def _build_trace_output(self, stage_output: AIContext) -> dict[str, Any]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
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
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
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
            reasons=self._build_reasons(rule),
            evidence=evidence,
            triggered_rules=[hit],
            confidence=rule.confidence,
            metadata={
                "configuration_version": self._arbitration.version,
                "matched_rule": rule.key,
                "signals": signals,
            },
        )

    @staticmethod
    def _build_reasons(rule: ArbitrationRule) -> list[str]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        # Условия и сигналы сюда не дублируем: они сохранены структурно
        # в ArbitrationRuleHit.conditions и в metadata["signals"], а плоская
        # строка нужна была только для печати в письме заказчику.
        return [rule.description]

    def _rule_matches(
        self,
        rule: ArbitrationRule,
        signals: dict[str, Any],
    ) -> bool:
        """Выполняет шаг «rule matches». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return all(
            self._condition_matches(condition, signals.get(condition.field))
            for condition in rule.conditions
        )

    def _build_arbitration_diagnostics(
        self,
        completeness_result: CompletenessResult,
        signals: dict[str, Any],
    ) -> dict[str, Any]:
        """Build temporary arbitration diagnostics without changing arbitration logic."""
        actuals = {
            "completeness.is_complete": signals.get("completeness.is_complete"),
            "completeness.missing_count": signals.get("completeness.missing_count"),
            "completeness.blocking_missing_count": signals.get(
                "completeness.blocking_missing_count"
            ),
            "completeness.critical_missing_count": len(
                completeness_result.critical_missing_information
            ),
            "completeness.clarification_count": signals.get(
                "completeness.clarification_count"
            ),
            "completeness.blocking_clarification_count": signals.get(
                "completeness.blocking_clarification_count"
            ),
            "completeness.optional_missing_count": signals.get(
                "completeness.optional_missing_count"
            ),
            "completeness.optional_clarification_count": signals.get(
                "completeness.optional_clarification_count"
            ),
            "risk.total_count": signals.get("risk.total_count"),
            "risk.max_severity": signals.get("risk.max_severity"),
            "risk.low_count": signals.get("risk.low_count"),
            "risk.medium_count": signals.get("risk.medium_count"),
            "risk.high_count": signals.get("risk.high_count"),
            "risk.critical_count": signals.get("risk.critical_count"),
            "risk.types": signals.get("risk.types"),
            "risk.unknown_type_count": signals.get("risk.unknown_type_count"),
            "risk.unknown_types": signals.get("risk.unknown_types"),
            "evaluation.total_count": signals.get("evaluation.total_count"),
            "evaluation.met_count": signals.get("evaluation.met_count"),
            "evaluation.not_met_count": signals.get("evaluation.not_met_count"),
            "evaluation.insufficient_information_count": signals.get(
                "evaluation.insufficient_information_count"
            ),
            "evaluation.risk_detected_count": signals.get(
                "evaluation.risk_detected_count"
            ),
        }
        rules = []
        for rule in self._arbitration.rules:
            condition_results = []
            failed_conditions = []
            for condition in rule.conditions:
                actual = signals.get(condition.field)
                passed = self._condition_matches(condition, actual)
                condition_result = {
                    "field": condition.field,
                    "operator": condition.operator,
                    "expected": condition.value,
                    "actual": actual,
                    "passed": passed,
                }
                condition_results.append(condition_result)
                if not passed:
                    failed_conditions.append(condition_result)
            rules.append(
                {
                    "key": rule.key,
                    "target_status": rule.status,
                    "conditions": condition_results,
                    "rejected_because": failed_conditions,
                }
            )
        return {
            "actuals": actuals,
            "rules": rules,
            "default_status": self._arbitration.default_status,
        }

    def _print_arbitration_diagnostics(
        self,
        diagnostics: dict[str, Any],
        matched_rule_key: str | None,
        default_status: str | None = None,
    ) -> None:
        """Print temporary arbitration diagnostics to stderr."""
        stream = sys.stderr
        if matched_rule_key is not None:
            print(
                "[ARBITRATION DIAGNOSTICS] result: "
                f"matched_rule={matched_rule_key}",
                file=stream,
                flush=True,
            )
            return

        if default_status is not None:
            print(
                "[ARBITRATION DIAGNOSTICS] result: no rule matched",
                file=stream,
                flush=True,
            )
            for rule in diagnostics["rules"]:
                failed_conditions = rule["rejected_because"]
                if not failed_conditions:
                    continue
                reasons = [
                    (
                        f"{condition['field']} "
                        f"actual={self._format_diagnostic_value(condition['actual'])} "
                        f"operator={condition['operator']} "
                        f"expected={self._format_diagnostic_value(condition['expected'])}"
                    )
                    for condition in failed_conditions
                ]
                print(
                    "  "
                    f"rule={rule['key']} rejected: "
                    f"{'; '.join(reasons)}",
                    file=stream,
                    flush=True,
                )
            print(
                "[ARBITRATION DIAGNOSTICS] default_status used: "
                f"{default_status} because no arbitration rule matched",
                file=stream,
                flush=True,
            )
            return

        print(
            "[ARBITRATION DIAGNOSTICS] actual values used by arbiter:",
            file=stream,
            flush=True,
        )
        for field, value in diagnostics["actuals"].items():
            print(
                "  "
                f"{field}: actual={self._format_diagnostic_value(value)}",
                file=stream,
                flush=True,
            )

        print("[ARBITRATION DIAGNOSTICS] rules:", file=stream, flush=True)
        for rule in diagnostics["rules"]:
            print(
                "  "
                f"rule={rule['key']} target_status={rule['target_status']}",
                file=stream,
                flush=True,
            )
            if rule["key"] == "accept_ready":
                print(
                    "    accept_ready detailed conditions:",
                    file=stream,
                    flush=True,
                )
            for condition in rule["conditions"]:
                result = "PASS" if condition["passed"] else "FAIL"
                print(
                    "    "
                    f"{condition['field']}: "
                    f"actual={self._format_diagnostic_value(condition['actual'])}, "
                    f"expected={self._format_diagnostic_value(condition['expected'])}, "
                    f"operator={condition['operator']} -> {result}",
                    file=stream,
                    flush=True,
                )

    def _condition_matches(
        self,
        condition: ArbitrationCondition,
        signal_value: Any,
    ) -> bool:
        """Выполняет шаг «condition matches». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        risk_counts = self._risk_counts(risks)
        evaluation_counts = self._evaluation_counts(criterion_evaluations)
        completeness_counts = self._completeness_counts(completeness_result)
        unknown_risk_types = [
            risk.type
            for risk in risks
            if risk.type not in self._known_risk_types
        ]

        signals: dict[str, Any] = {
            "completeness.is_complete": completeness_result.is_complete,
            "completeness.missing_count": completeness_counts["missing_count"],
            "completeness.blocking_missing_count": completeness_counts[
                "blocking_missing_count"
            ],
            "completeness.clarification_count": completeness_counts[
                "clarification_count"
            ],
            "completeness.blocking_clarification_count": completeness_counts[
                "blocking_clarification_count"
            ],
            "completeness.optional_missing_count": completeness_counts[
                "optional_missing_count"
            ],
            "completeness.optional_clarification_count": completeness_counts[
                "optional_clarification_count"
            ],
            "completeness.present_count": completeness_counts["present_count"],
            "risk.has_risks": bool(risks),
            "risk.total_count": risk_counts["total_count"],
            "risk.low_count": risk_counts["low_count"],
            "risk.medium_count": risk_counts["medium_count"],
            "risk.high_count": risk_counts["high_count"],
            "risk.critical_count": risk_counts["critical_count"],
            "risk.max_severity": risk_counts["max_severity"],
            "risk.types": [risk.type for risk in risks],
            "risk.unknown_type_count": len(unknown_risk_types),
            "risk.unknown_types": unknown_risk_types,
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
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        evidence_map: dict[str, list[str]] = {
            "completeness.is_complete": self._present_field_evidence(
                completeness_result
            ),
            "completeness.missing_count": [
                f"{item.title}: {item.reason or item.field_path}"
                for item in completeness_result.missing_information
            ],
            "completeness.blocking_missing_count": [
                f"{item.title}: {item.reason or item.field_path}"
                for item in completeness_result.missing_information
            ],
            "completeness.clarification_count": [
                f"{item.title}: {item.reason or item.field_path}"
                for item in completeness_result.clarification_information
            ],
            "completeness.blocking_clarification_count": [
                f"{item.title}: {item.reason or item.field_path}"
                for item in completeness_result.clarification_information
            ],
            "completeness.optional_missing_count": [
                f"{item.title}: {item.reason or item.field_path}"
                for item in completeness_result.optional_missing_information
                if item.status is CompletenessStatus.missing
            ],
            "completeness.optional_clarification_count": [
                f"{item.title}: {item.reason or item.field_path}"
                for item in completeness_result.optional_missing_information
                if item.status is CompletenessStatus.clarification
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
            "risk.types": [
                f"{risk.type}: {risk.description}"
                for risk in risks
            ],
            "risk.unknown_type_count": [
                f"{risk.type}: {risk.description}"
                for risk in risks
                if risk.type not in self._known_risk_types
            ],
            "risk.unknown_types": [
                f"{risk.type}: {risk.description}"
                for risk in risks
                if risk.type not in self._known_risk_types
            ],
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
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        self._parse_status(self._arbitration.default_status)

        for rule in self._arbitration.rules:
            self._parse_status(rule.status)
            for condition in rule.conditions:
                if condition.field not in self._supported_signals:
                    raise ArbitrationConfigError(
                        f"Unsupported arbitration signal: {condition.field}"
                    )

    def _build_supported_signals(self) -> set[str]:
        """Собирает вспомогательные данные для следующего шага. Такие методы не принимают решений сами, а готовят детали для основного процесса."""
        return {
            "completeness.is_complete",
            "completeness.missing_count",
            "completeness.blocking_missing_count",
            "completeness.clarification_count",
            "completeness.blocking_clarification_count",
            "completeness.optional_missing_count",
            "completeness.optional_clarification_count",
            "completeness.present_count",
            "risk.has_risks",
            "risk.total_count",
            "risk.low_count",
            "risk.medium_count",
            "risk.high_count",
            "risk.critical_count",
            "risk.max_severity",
            "risk.types",
            "risk.unknown_type_count",
            "risk.unknown_types",
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
        """Выполняет шаг «collect evidence». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        collected: list[str] = []
        for condition in rule.conditions:
            for fragment in evidence_map.get(condition.field, []):
                if fragment not in collected:
                    collected.append(fragment)
        return collected

    @staticmethod
    def _render_condition(condition: ArbitrationCondition) -> str:
        """Готовит человекочитаемый текст из внутренних данных. Это нужно для промптов, объяснений или финального ответа."""
        return (
            f"{condition.field} {condition.operator} "
            f"{condition.value!r}"
        )

    @staticmethod
    def _format_diagnostic_value(value: Any) -> str:
        """Format temporary arbitration diagnostics for stdout."""
        if isinstance(value, bool):
            return str(value).lower()
        return repr(value)

    @staticmethod
    def _normalize_value(value: Any, case_sensitive: bool) -> Any:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
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
        """Выполняет шаг «compare numbers». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «membership». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if not isinstance(right, list):
            right = [right]

        if isinstance(left, list):
            return any(item in right for item in left)

        return left in right

    @staticmethod
    def _contains(left: Any, right: Any) -> bool:
        """Выполняет шаг «contains». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if isinstance(left, str) and isinstance(right, str):
            return right in left
        if isinstance(left, list):
            return right in left
        return False

    @staticmethod
    def _any_in(left: Any, right: Any) -> bool:
        """Выполняет шаг «any in». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if not isinstance(left, list):
            return False
        if not isinstance(right, list):
            right = [right]
        return any(item in right for item in left)

    @staticmethod
    def _all_in(left: Any, right: Any) -> bool:
        """Выполняет шаг «all in». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if not isinstance(left, list):
            return False
        if not isinstance(right, list):
            right = [right]
        return all(item in right for item in left)

    @staticmethod
    def _parse_status(status: str) -> DecisionStatus:
        """Разбирает текстовое значение и превращает его в программный объект. Так код дальше работает не с произвольной строкой, а с понятной структурой."""
        try:
            return DecisionStatus(status)
        except ValueError as exc:
            raise ArbitrationConfigError(
                f"Unsupported decision status: {status}"
            ) from exc

    @staticmethod
    def _risk_counts(risks: list[Risk]) -> dict[str, Any]:
        """Выполняет шаг «risk counts». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «risk max severity». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «risk evidence». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «evaluation counts». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «evaluation evidence». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «completeness counts». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return {
            "present_count": len(completeness_result.present_information),
            "missing_count": len(completeness_result.missing_information),
            "blocking_missing_count": len(completeness_result.missing_information),
            "clarification_count": len(completeness_result.clarification_information),
            "blocking_clarification_count": len(
                completeness_result.clarification_information
            ),
            "optional_missing_count": sum(
                1
                for item in completeness_result.optional_missing_information
                if item.status is CompletenessStatus.missing
            ),
            "optional_clarification_count": sum(
                1
                for item in completeness_result.optional_missing_information
                if item.status is CompletenessStatus.clarification
            ),
        }

    @staticmethod
    def _present_field_evidence(
        completeness_result: CompletenessResult,
    ) -> list[str]:
        """Выполняет шаг «present field evidence». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return [item.title for item in completeness_result.present_information]
