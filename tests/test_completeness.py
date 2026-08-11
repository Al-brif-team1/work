"""Tests for the completeness check stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

from app.input import BriefInputFactory
from app.pipeline import (
    BaseStage,
    CompletenessCheckStage,
    CompletenessConfigError,
    CompletenessError,
)
from app.schemas import (
    AIContext,
    CompletenessItem,
    CompletenessLevel,
    CompletenessResult,
    CompletenessStatus,
    ExtractedBrief,
    ExtractedFact,
    FactStatus,
)


def fact(value: str | None, status: FactStatus = FactStatus.explicit) -> ExtractedFact:
    """Create a compact extracted fact for tests."""
    return ExtractedFact(
        status=status,
        value=value,
        evidence=[value] if value else [],
        confidence=0.9 if status is FactStatus.explicit else None,
        notes=None,
    )


def make_brief(
    *,
    project_goal: ExtractedFact,
    tasks: list[ExtractedFact],
    project_type: ExtractedFact,
    project_direction: ExtractedFact,
    expected_result: ExtractedFact,
    technologies: list[ExtractedFact] | None = None,
) -> ExtractedBrief:
    """Create a minimal extracted brief for completeness tests."""
    return ExtractedBrief(
        project_goal=project_goal,
        tasks=tasks,
        project_type=project_type,
        project_direction=project_direction,
        technologies=technologies or [],
        stack=[],
        materials=[],
        expected_result=expected_result,
        constraints=[],
        deadlines=[],
        existing_resources=[],
        integrations=[],
        other_facts=[],
    )


def write_criteria_yaml(path: Path, project_type_key: str = "web_app") -> None:
    """Write a temporary criteria YAML file for tests."""
    path.write_text(
        dedent(
            f"""
            evaluation:
              version: "1"
              description: Test criteria configuration.
              project_types:
                - key: {project_type_key}
                  title: {project_type_key}
                  description: Recognized project type for tests.
                  task_types:
                    - implementation
                  aliases:
                    - {project_type_key}_alias
              task_types:
                - key: implementation
                  title: implementation
                  description: Implementation task type.
                  criteria:
                    - goal
              criteria:
                - key: goal
                  title: Project goal
                  description: Project goal criterion.
                  allowed_values:
                    - placeholder
                  status_signals:
                    - placeholder
              required_fields:
                - key: project_goal
                  field_path: project_goal
                  title: Project goal
                  description: Required project goal.
                  required: true
                - key: tasks
                  field_path: tasks
                  title: Tasks
                  description: Required tasks.
                  required: true
                - key: project_type
                  field_path: project_type
                  title: Project type
                  description: Required project type.
                  required: true
                - key: project_direction
                  field_path: project_direction
                  title: Project direction
                  description: Required project direction.
                  required: true
                - key: expected_result
                  field_path: expected_result
                  title: Expected result
                  description: Required expected result.
                  required: true
                - key: technologies
                  field_path: technologies
                  title: Technologies
                  description: Optional technologies.
                  required: false
              decision_thresholds:
                - min_score: 0
                  max_score: 1
                  conditions:
                    - placeholder
                  description: Placeholder threshold.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


class TestCompletenessCheckStage(unittest.TestCase):
    """Unit tests for completeness checking."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.criteria_path = Path(self.tmpdir.name) / "criteria.yaml"
        write_criteria_yaml(self.criteria_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_complete_brief_is_marked_complete(self) -> None:
        checker = CompletenessCheckStage(criteria_path=self.criteria_path)
        brief = make_brief(
            project_goal=fact("Build a support bot"),
            tasks=[fact("Implement bot")],
            project_type=fact("web_app"),
            project_direction=fact("support automation"),
            expected_result=fact("Working support bot"),
            technologies=[fact("Python")],
        )

        result = checker.check(brief)

        self.assertTrue(result.is_complete)
        self.assertEqual(result.level, CompletenessLevel.complete)
        self.assertEqual(len(result.missing_information), 0)
        self.assertEqual(len(result.critical_missing_information), 0)
        self.assertEqual(len(result.clarification_information), 0)
        self.assertGreaterEqual(len(result.present_information), 5)
        self.assertEqual(result.technical_info.required_fields_count, 5)
        self.assertEqual(result.technical_info.optional_fields_count, 1)

    def test_missing_goal_is_reported(self) -> None:
        checker = CompletenessCheckStage(criteria_path=self.criteria_path)
        brief = make_brief(
            project_goal=fact(None, status=FactStatus.missing),
            tasks=[fact("Implement bot")],
            project_type=fact("web_app"),
            project_direction=fact("support automation"),
            expected_result=fact("Working support bot"),
        )

        result = checker.check(brief)

        self.assertFalse(result.is_complete)
        self.assertEqual(result.level, CompletenessLevel.incomplete)
        self.assertEqual(len(result.missing_information), 1)
        self.assertEqual(len(result.critical_missing_information), 1)
        self.assertEqual(result.missing_information[0].field_key, "project_goal")
        self.assertEqual(
            result.critical_missing_information[0].field_key,
            "project_goal",
        )
        self.assertEqual(
            result.missing_information[0].status,
            CompletenessStatus.missing,
        )

    def test_missing_tasks_are_reported(self) -> None:
        checker = CompletenessCheckStage(criteria_path=self.criteria_path)
        brief = make_brief(
            project_goal=fact("Build a support bot"),
            tasks=[],
            project_type=fact("web_app"),
            project_direction=fact("support automation"),
            expected_result=fact("Working support bot"),
        )

        result = checker.check(brief)

        self.assertFalse(result.is_complete)
        self.assertEqual(len(result.missing_information), 1)
        self.assertEqual(result.missing_information[0].field_key, "tasks")

    def test_only_optional_fields_missing_keeps_brief_complete(self) -> None:
        checker = CompletenessCheckStage(criteria_path=self.criteria_path)
        brief = make_brief(
            project_goal=fact("Build a support bot"),
            tasks=[fact("Implement bot")],
            project_type=fact("web_app"),
            project_direction=fact("support automation"),
            expected_result=fact("Working support bot"),
            technologies=[],
        )

        result = checker.check(brief)

        self.assertTrue(result.is_complete)
        self.assertEqual(result.level, CompletenessLevel.complete)
        self.assertEqual(result.missing_information, [])
        self.assertEqual(result.critical_missing_information, [])

    def test_multiple_missing_fields_are_reported(self) -> None:
        checker = CompletenessCheckStage(criteria_path=self.criteria_path)
        brief = make_brief(
            project_goal=fact(None, status=FactStatus.missing),
            tasks=[],
            project_type=fact("web_app"),
            project_direction=fact(None, status=FactStatus.missing),
            expected_result=fact(None, status=FactStatus.missing),
        )

        result = checker.check(brief)

        self.assertFalse(result.is_complete)
        self.assertEqual(
            {item.field_key for item in result.missing_information},
            {"project_goal", "tasks", "project_direction", "expected_result"},
        )
        self.assertEqual(
            {item.field_key for item in result.critical_missing_information},
            {"project_goal", "tasks", "project_direction", "expected_result"},
        )

    def test_unknown_project_type_requires_clarification(self) -> None:
        checker = CompletenessCheckStage(criteria_path=self.criteria_path)
        brief = make_brief(
            project_goal=fact("Build a support bot"),
            tasks=[fact("Implement bot")],
            project_type=fact("mobile_app"),
            project_direction=fact("support automation"),
            expected_result=fact("Working support bot"),
        )

        result = checker.check(brief)

        self.assertFalse(result.is_complete)
        self.assertEqual(result.level, CompletenessLevel.needs_clarification)
        self.assertEqual(len(result.clarification_information), 1)
        self.assertEqual(result.clarification_information[0].field_key, "project_type")
        self.assertIn("Unknown project type", result.warnings[0])

    def test_almost_empty_brief_reports_all_critical_missing_fields(self) -> None:
        checker = CompletenessCheckStage(criteria_path=self.criteria_path)
        brief = make_brief(
            project_goal=fact(None, status=FactStatus.missing),
            tasks=[],
            project_type=fact(None, status=FactStatus.missing),
            project_direction=fact(None, status=FactStatus.missing),
            expected_result=fact(None, status=FactStatus.missing),
        )

        result = checker.check(brief)

        self.assertFalse(result.is_complete)
        self.assertEqual(result.level, CompletenessLevel.incomplete)
        self.assertEqual(result.technical_info.critical_missing_count, 5)

    def test_stage_uses_base_stage_lifecycle_and_updates_context(self) -> None:
        stage = CompletenessCheckStage(criteria_path=self.criteria_path)
        brief = make_brief(
            project_goal=fact("Build a support bot"),
            tasks=[fact("Implement bot")],
            project_type=fact("web_app"),
            project_direction=fact("support automation"),
            expected_result=fact("Working support bot"),
        )
        context = AIContext.from_brief(
            BriefInputFactory().from_text("Build a support bot")
        ).with_extracted_brief(brief)

        updated = stage.run(context)

        self.assertIsInstance(stage, BaseStage)
        self.assertIsNone(context.completeness_result)
        self.assertIsNotNone(updated.completeness_result)
        self.assertTrue(updated.completeness_result.is_complete)

    def test_stage_requires_extraction_result_or_extracted_brief(self) -> None:
        stage = CompletenessCheckStage(criteria_path=self.criteria_path)
        context = AIContext.from_brief(
            BriefInputFactory().from_text("Build a support bot")
        )

        with self.assertRaisesRegex(CompletenessError, "extracted_brief"):
            stage.run(context)

    def test_completeness_result_model_keeps_assessment_ready_counts(self) -> None:
        item = CompletenessItem(
            field_key="project_goal",
            field_path="project_goal",
            title="Project goal",
            status=CompletenessStatus.missing,
        )

        result = CompletenessResult(
            is_complete=False,
            level=CompletenessLevel.incomplete,
            missing_information=[item],
            critical_missing_information=[item],
        )

        self.assertEqual(result.critical_missing_information[0].field_key, "project_goal")
        self.assertEqual(result.technical_info.checked_fields_count, 0)

    def test_invalid_configuration_is_rejected(self) -> None:
        bad_path = Path(self.tmpdir.name) / "bad_criteria.yaml"
        bad_path.write_text(
            dedent(
                """
                evaluation:
                  version: "1"
                  description: Invalid test criteria configuration.
                  project_types:
                    - key: web_app
                      title: web_app
                      description: Recognized project type.
                      task_types:
                        - implementation
                      aliases:
                        - web_app_alias
                  task_types:
                    - key: implementation
                      title: implementation
                      description: Implementation task type.
                      criteria:
                        - goal
                  criteria:
                    - key: goal
                      title: Project goal
                      description: Project goal criterion.
                      allowed_values:
                        - placeholder
                      status_signals:
                        - placeholder
                  required_fields:
                    - key: project_goal
                      field_path: does_not_exist
                      title: Project goal
                      description: Invalid field path.
                      required: true
                  decision_thresholds:
                    - min_score: 0
                      max_score: 1
                      conditions:
                        - placeholder
                      description: Placeholder threshold.
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaises(CompletenessConfigError):
            CompletenessCheckStage(criteria_path=bad_path)


if __name__ == "__main__":
    unittest.main()
