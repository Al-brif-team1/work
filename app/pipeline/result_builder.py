"""Build the product-facing JSON result from AIContext."""

from __future__ import annotations

from app.schemas import (
    AIContext,
    BriefAnalysisResult,
    BriefAssessmentSummary,
    BriefExtractedFields,
    DecisionStatus,
    ExtractedFact,
    RiskSeverity,
)


class BriefAnalysisResultError(RuntimeError):
    """Raised when the final product-facing result cannot be built."""


class BriefAnalysisResultBuilder:
    """Convert internal pipeline models into the required public JSON shape."""

    _STATUS_MAP = {
        DecisionStatus.accept: "accept",
        DecisionStatus.clarify: "clarify",
        DecisionStatus.simplify: "simplify",
        DecisionStatus.mentor_review: "mentor_review",
        DecisionStatus.reject: "reject",
    }

    def build(self, context: AIContext) -> BriefAnalysisResult:
        """Build a validated result for one analyzed brief."""
        self._validate_context(context)
        extracted = context.extracted_brief
        assessment = context.assessment_result
        arbitration = context.arbitration_result

        assert extracted is not None
        assert assessment is not None
        assert arbitration is not None

        return BriefAnalysisResult(
            summary=self._build_summary(context),
            extracted_fields=BriefExtractedFields(
                goal=self._fact_value(extracted.project_goal),
                expected_result=self._fact_value(extracted.expected_result),
                tasks=self._fact_values(extracted.tasks),
                domain=self._fact_value(extracted.project_type),
                direction=self._normalize_direction(
                    self._fact_value(extracted.project_direction)
                ),
                available_materials=self._deduplicate(
                    [
                        *self._fact_values(extracted.materials),
                        *self._fact_values(extracted.existing_resources),
                        *self._fact_values(extracted.integrations),
                    ]
                ),
                missing_information=[
                    item.title for item in context.completeness_result.missing_information
                ]
                if context.completeness_result is not None
                else [],
                complexity_factors=self._deduplicate(
                    [
                        *self._fact_values(extracted.constraints),
                        *[
                            risk.description
                            for risk in assessment.risks
                            if risk.severity in {RiskSeverity.high, RiskSeverity.critical}
                        ],
                    ]
                ),
            ),
            assessment=BriefAssessmentSummary(
                recommendation=self._STATUS_MAP[arbitration.final_status],
                confidence=self._confidence_label(
                    arbitration.confidence or assessment.confidence
                ),
                reasons=self._deduplicate(
                    [
                        *arbitration.reasons,
                        *[
                            item.explanation
                            for item in assessment.criterion_evaluations
                            if item.explanation
                        ],
                    ]
                ),
                risks=[risk.description for risk in assessment.risks],
            ),
            clarifying_questions=[
                item.question
                for item in (
                    context.clarification_result.questions
                    if context.clarification_result is not None
                    else []
                )
            ],
            mvp_suggestion=self._format_mvp_suggestion(context),
            customer_response_draft=context.final_response_text or "",
        )

    @staticmethod
    def _validate_context(context: AIContext) -> None:
        """Ensure all mandatory pipeline stages have populated the context."""
        if context.extracted_brief is None:
            raise BriefAnalysisResultError("Final result requires extracted_brief")
        if context.completeness_result is None:
            raise BriefAnalysisResultError("Final result requires completeness_result")
        if context.assessment_result is None:
            raise BriefAnalysisResultError("Final result requires assessment_result")
        if context.arbitration_result is None:
            raise BriefAnalysisResultError("Final result requires arbitration_result")
        if context.final_response_text is None:
            raise BriefAnalysisResultError("Final result requires final_response_text")

        status = context.arbitration_result.final_status
        if status is DecisionStatus.clarify and not (
            context.clarification_result
            and context.clarification_result.questions
        ):
            raise BriefAnalysisResultError(
                "CLARIFY final result requires clarification questions"
            )
        if status is DecisionStatus.simplify and not (
            context.mvp_planning_result and context.mvp_planning_result.plan
        ):
            raise BriefAnalysisResultError(
                "SIMPLIFY final result requires an MVP plan"
            )

    @staticmethod
    def _fact_value(fact: ExtractedFact) -> str | None:
        """Return a normalized value from an extracted fact."""
        if fact.value is None:
            return None
        value = fact.value.strip()
        return value or None

    @classmethod
    def _fact_values(cls, facts: list[ExtractedFact]) -> list[str]:
        """Return normalized non-empty values from extracted facts."""
        return cls._deduplicate(
            [value for fact in facts if (value := cls._fact_value(fact))]
        )

    @staticmethod
    def _normalize_direction(value: str | None) -> str | None:
        """Normalize common direction names without rejecting model wording."""
        if value is None:
            return None
        normalized = " ".join(value.lower().split())
        aliases = {
            "разработка": "development",
            "development": "development",
            "дизайн": "design",
            "design": "design",
            "аналитика": "analytics",
            "analytics": "analytics",
            "маркетинг": "marketing",
            "marketing": "marketing",
            "ии": "ai",
            "ai": "ai",
            "искусственный интеллект": "ai",
            "образование": "education",
            "education": "education",
            "смешанный проект": "mixed",
            "mixed": "mixed",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _confidence_label(value: float | None) -> str:
        """Convert numeric confidence into the public low/medium/high scale."""
        if value is None:
            return "medium"
        if value < 0.45:
            return "low"
        if value < 0.75:
            return "medium"
        return "high"

    def _build_summary(self, context: AIContext) -> str:
        """Build a compact project summary from assessment or extracted facts."""
        if context.assessment_result and context.assessment_result.summary:
            return context.assessment_result.summary

        assert context.extracted_brief is not None
        goal = self._fact_value(context.extracted_brief.project_goal)
        result = self._fact_value(context.extracted_brief.expected_result)
        if goal and result:
            return f"{goal}. Ожидаемый результат: {result}."
        if goal:
            return goal
        if result:
            return f"Ожидаемый результат: {result}."
        return "Краткое описание проекта не удалось извлечь из брифа."

    @staticmethod
    def _format_mvp_suggestion(context: AIContext) -> str:
        """Return a readable MVP suggestion if the MVP planner produced one."""
        if (
            context.arbitration_result is None
            or context.arbitration_result.final_status is not DecisionStatus.simplify
        ):
            return ""

        planning = context.mvp_planning_result
        if planning is None or planning.plan is None:
            return ""

        plan = planning.plan
        parts = [f"Цель MVP: {plan.core_goal}"]
        if plan.keep:
            parts.append("Оставить: " + "; ".join(plan.keep))
        if plan.simplify:
            parts.append("Упростить: " + "; ".join(plan.simplify))
        if plan.remove:
            parts.append("Исключить из первой версии: " + "; ".join(plan.remove))
        if plan.mvp_scope:
            parts.append("Состав первой версии: " + "; ".join(plan.mvp_scope))
        return "\n".join(parts)

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        """Deduplicate strings while preserving order."""
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result
