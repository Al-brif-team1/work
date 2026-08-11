"""Completeness checking for extracted project briefs."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from app.config import (
    CriteriaConfig,
    CriteriaConfigError,
    CriteriaLoader,
    RequiredField,
    get_criteria_config,
)
from app.pipeline.contracts import BaseStage
from app.schemas import (
    AIContext,
    CompletenessLevel,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    CompletenessTechnicalInfo,
    ExtractedFact,
    ExtractedBrief,
    FactStatus,
)
from app.tracing.tracing import NoOpTracingClient, TracingClient


class CompletenessError(RuntimeError):
    """Raised when completeness checking fails."""


class CompletenessConfigError(CompletenessError):
    """Raised when completeness configuration is invalid."""


class CompletenessCheckStage(BaseStage[AIContext, AIContext]):
    """Determine which brief data is present, missing, or ambiguous."""

    def __init__(
        self,
        criteria_config: CriteriaConfig | None = None,
        criteria_path: str | Path | None = None,
        tracing_client: TracingClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the checker from a config object or YAML file."""
        super().__init__(
            stage_name=self.__class__.__name__,
            tracing_client=tracing_client or NoOpTracingClient(),
            logger=logger,
        )
        if criteria_config is not None and criteria_path is not None:
            raise ValueError("Pass either criteria_config or criteria_path, not both")

        if criteria_config is not None:
            config = criteria_config
        elif criteria_path is not None:
            try:
                config = CriteriaLoader.load(Path(criteria_path))
            except CriteriaConfigError as exc:
                raise CompletenessConfigError(str(exc)) from exc
        else:
            try:
                config = get_criteria_config()
            except CriteriaConfigError as exc:
                raise CompletenessConfigError(str(exc)) from exc

        self._config = self._validate_config(config)
        self._project_type_registry = self._build_project_type_registry(
            self._config.evaluation.project_types
        )

    def check(self, extracted_brief: ExtractedBrief) -> CompletenessResult:
        """Return completeness information for an extracted brief."""
        present_information: list[CompletenessItem] = []
        missing_information: list[CompletenessItem] = []
        critical_missing_information: list[CompletenessItem] = []
        clarification_information: list[CompletenessItem] = []
        warnings: list[str] = []
        required_fields_count = 0
        optional_fields_count = 0

        for field_def in self._config.evaluation.required_fields:
            if field_def.required:
                required_fields_count += 1
            else:
                optional_fields_count += 1

            item = self._evaluate_field(field_def, extracted_brief)
            if item is None:
                continue

            if item.status is CompletenessStatus.present:
                present_information.append(item)
                continue

            if item.status is CompletenessStatus.missing:
                if field_def.required:
                    missing_information.append(item)
                    critical_missing_information.append(item)
                continue

            clarification_information.append(item)
            if item.reason:
                warnings.append(item.reason)

        level = self._build_level(
            missing_information=missing_information,
            clarification_information=clarification_information,
        )
        return CompletenessResult(
            is_complete=not missing_information and not clarification_information,
            level=level,
            missing_information=missing_information,
            critical_missing_information=critical_missing_information,
            present_information=present_information,
            clarification_information=clarification_information,
            warnings=warnings,
            technical_info=CompletenessTechnicalInfo(
                checked_fields_count=(
                    len(present_information)
                    + len(missing_information)
                    + len(clarification_information)
                ),
                required_fields_count=required_fields_count,
                optional_fields_count=optional_fields_count,
                present_count=len(present_information),
                missing_count=len(missing_information),
                critical_missing_count=len(critical_missing_information),
                clarification_count=len(clarification_information),
            ),
        )

    def _run(self, stage_input: AIContext) -> AIContext:
        """Run deterministic completeness checking for pipeline context."""
        return self.check_context(stage_input)

    def check_context(self, context: AIContext) -> AIContext:
        """Return context enriched with completeness output."""
        if context.extracted_brief is None:
            raise CompletenessError(
                "Completeness check requires extracted_brief in AIContext"
            )
        return context.with_completeness_result(self.check(context.extracted_brief))

    def run_context(self, context: AIContext) -> AIContext:
        """Run this stage using the common AIContext pipeline contract."""
        return self.run(context)

    def _evaluate_field(
        self,
        field_def: RequiredField,
        extracted_brief: ExtractedBrief,
    ) -> CompletenessItem | None:
        """Evaluate a single configured field against the extracted brief."""
        resolved_value = self._resolve_field_path(extracted_brief, field_def.field_path)
        status, value, reason = self._classify_value(
            field_def=field_def,
            resolved_value=resolved_value,
        )

        if status is CompletenessStatus.missing and not field_def.required:
            return None

        return CompletenessItem(
            field_key=field_def.key,
            field_path=field_def.field_path,
            title=field_def.title,
            status=status,
            value=value,
            reason=reason,
        )

    def _classify_value(
        self,
        field_def: RequiredField,
        resolved_value: Any,
    ) -> tuple[CompletenessStatus, Any | None, str | None]:
        """Classify a resolved field value into completeness categories."""
        if isinstance(resolved_value, ExtractedFact):
            return self._classify_fact(field_def, resolved_value)

        if isinstance(resolved_value, list) and self._is_fact_list(resolved_value):
            return self._classify_fact_list(field_def, resolved_value)

        if resolved_value is None:
            return CompletenessStatus.missing, None, None

        if isinstance(resolved_value, str):
            if resolved_value.strip():
                return CompletenessStatus.present, resolved_value, None
            return CompletenessStatus.missing, None, None

        if isinstance(resolved_value, BaseModel):
            return (
                CompletenessStatus.present,
                resolved_value.model_dump(mode="json"),
                None,
            )

        if isinstance(resolved_value, list):
            if resolved_value:
                return CompletenessStatus.present, resolved_value, None
            return CompletenessStatus.missing, None, None

        return CompletenessStatus.present, resolved_value, None

    def _classify_fact(
        self,
        field_def: RequiredField,
        fact: ExtractedFact,
    ) -> tuple[CompletenessStatus, Any | None, str | None]:
        """Classify a single extracted fact."""
        if fact.status is FactStatus.explicit and fact.value is not None:
            if (
                self._is_project_type_field(field_def)
                and not self._is_known_project_type(fact.value)
            ):
                return (
                    CompletenessStatus.clarification,
                    fact.value,
                    f"Unknown project type: {fact.value}",
                )

            return CompletenessStatus.present, fact.value, None

        if fact.status is FactStatus.uncertain:
            return (
                CompletenessStatus.clarification,
                fact.value,
                f"{field_def.title} requires clarification",
            )

        return CompletenessStatus.missing, None, None

    def _classify_fact_list(
        self,
        field_def: RequiredField,
        facts: list[ExtractedFact],
    ) -> tuple[CompletenessStatus, Any | None, str | None]:
        """Classify a list of extracted facts."""
        if not facts:
            return CompletenessStatus.missing, None, None

        explicit_values = [
            fact.value
            for fact in facts
            if fact.status is FactStatus.explicit and fact.value
        ]
        uncertain_values = [
            fact.value
            for fact in facts
            if fact.status is FactStatus.uncertain and fact.value
        ]
        missing_count = sum(1 for fact in facts if fact.status is FactStatus.missing)

        if explicit_values and not uncertain_values and missing_count == 0:
            return CompletenessStatus.present, explicit_values, None

        if explicit_values or uncertain_values:
            reason = f"{field_def.title} requires clarification"
            return (
                CompletenessStatus.clarification,
                explicit_values or uncertain_values or None,
                reason,
            )

        return CompletenessStatus.missing, None, None

    def _resolve_field_path(self, obj: Any, field_path: str) -> Any:
        """Resolve a dotted field path against a Pydantic model or mapping."""
        current: Any = obj
        for part in field_path.split("."):
            if isinstance(current, BaseModel):
                if part not in type(current).model_fields:
                    raise CompletenessConfigError(
                        f"Unknown field path in criteria configuration: {field_path}"
                    )
                current = getattr(current, part)
                continue

            if isinstance(current, dict):
                if part not in current:
                    raise CompletenessConfigError(
                        f"Unknown field path in criteria configuration: {field_path}"
                    )
                current = current[part]
                continue

            raise CompletenessConfigError(
                f"Unknown field path in criteria configuration: {field_path}"
            )

        return current

    def _validate_config(self, config: CriteriaConfig) -> CriteriaConfig:
        """Validate config semantics that are not covered by the YAML schema."""
        if not config.evaluation.required_fields:
            raise CompletenessConfigError(
                "criteria configuration has no required fields"
            )

        for field_def in config.evaluation.required_fields:
            if not field_def.field_path.strip():
                raise CompletenessConfigError(
                    f"Required field {field_def.key} has an empty field_path"
                )

            self._validate_field_path(field_def.field_path)

        return config

    def _validate_field_path(self, field_path: str) -> None:
        """Ensure the configured path exists on ExtractedBrief."""
        model: type[BaseModel] | None = ExtractedBrief
        for part in field_path.split("."):
            if model is None:
                raise CompletenessConfigError(
                    f"Unknown field path in criteria configuration: {field_path}"
                )

            field_info = model.model_fields.get(part)
            if field_info is None:
                raise CompletenessConfigError(
                    f"Unknown field path in criteria configuration: {field_path}"
                )

            model = self._extract_model_type(field_info.annotation)

    def _extract_model_type(self, annotation: Any) -> type[BaseModel] | None:
        """Return a nested BaseModel type from a field annotation if present."""
        origin = get_origin(annotation)
        if origin is list:
            for candidate in get_args(annotation):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                    return candidate
            return None

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation

        return None

    def _build_project_type_registry(
        self,
        project_types: list[Any],
    ) -> set[str]:
        """Build a normalized registry of known project type names."""
        registry: set[str] = set()
        for project_type in project_types:
            registry.add(self._normalize_text(project_type.key))
            registry.add(self._normalize_text(project_type.title))
            for alias in project_type.aliases:
                registry.add(self._normalize_text(alias))
        return registry

    def _is_project_type_field(self, field_def: RequiredField) -> bool:
        """Return whether the field represents the project type."""
        return field_def.field_path == "project_type"

    def _is_known_project_type(self, value: str) -> bool:
        """Check whether a project type value exists in the configured registry."""
        return self._normalize_text(value) in self._project_type_registry

    @staticmethod
    def _is_fact_list(value: list[Any]) -> bool:
        """Check whether a list contains ExtractedFact items."""
        return all(isinstance(item, ExtractedFact) for item in value)

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize text for registry lookups."""
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _build_level(
        *,
        missing_information: list[CompletenessItem],
        clarification_information: list[CompletenessItem],
    ) -> CompletenessLevel:
        """Build the aggregate deterministic completeness level."""
        if missing_information:
            return CompletenessLevel.incomplete
        if clarification_information:
            return CompletenessLevel.needs_clarification
        return CompletenessLevel.complete

    def _build_stage_exception(self, exc: Exception) -> Exception:
        """Preserve completeness-specific errors at the stage boundary."""
        return exc

    def _build_trace_input(self, stage_input: AIContext) -> dict[str, Any]:
        """Build safe trace metadata for completeness checking."""
        return {
            "has_extraction_result": stage_input.extraction_result is not None,
            "has_extracted_brief": stage_input.extracted_brief is not None,
        }

    def _build_trace_output(self, stage_output: AIContext) -> dict[str, Any]:
        """Build safe trace metadata for the completeness result."""
        result = stage_output.completeness_result
        return {
            "status": "success",
            "is_complete": result.is_complete if result is not None else None,
            "level": result.level.value if result is not None else None,
        }

