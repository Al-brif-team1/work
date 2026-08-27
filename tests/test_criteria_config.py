import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import (
    CriteriaConfigError,
    CriteriaLoader,
    EvaluationConfiguration,
    get_criteria_config,
)


class TestCriteriaConfig(unittest.TestCase):
    """Класс «TestCriteriaConfig» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def tearDown(self) -> None:
        get_criteria_config.cache_clear()

    def test_loads_default_criteria_config(self) -> None:
        config = CriteriaLoader.load()

        self.assertIsInstance(config.evaluation, EvaluationConfiguration)
        self.assertEqual(config.evaluation.project_types[0].key, "development")
        self.assertEqual(config.evaluation.criteria[0].key, "request_eligibility")
        self.assertEqual(config.evaluation.required_fields[0].key, "project_goal")

    def test_eligibility_gate_is_configured(self) -> None:
        config = CriteriaLoader.load()

        criteria_keys = [item.key for item in config.evaluation.criteria]
        risk_keys = [
            item.key for item in config.evaluation.risk_analysis.risk_types
        ]
        rules = config.evaluation.arbitration.rules

        self.assertEqual(criteria_keys[0], "request_eligibility")
        self.assertIn("out_of_scope_request", risk_keys)
        self.assertEqual(rules[0].key, "reject_out_of_scope")
        self.assertEqual(rules[0].status, "REJECT")
        self.assertEqual(rules[0].conditions[0].field, "risk.types")
        self.assertEqual(rules[0].conditions[0].operator, "any_in")
        self.assertIn("out_of_scope_request", rules[0].conditions[0].value)

    def test_get_criteria_config_is_cached(self) -> None:
        sentinel = CriteriaLoader.load()

        with patch.object(CriteriaLoader, "load", return_value=sentinel) as mocked:
            first = get_criteria_config()
            second = get_criteria_config()

        self.assertIs(first, second)
        self.assertEqual(mocked.call_count, 1)

    def test_invalid_yaml_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "broken.yaml"
            path.write_text(
                "evaluation:\n  version \"1\"\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                CriteriaConfigError,
                "Invalid YAML syntax in criteria configuration",
            ):
                CriteriaLoader.load(path)

    def test_invalid_schema_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid.yaml"
            path.write_text("evaluation:\n  version: '1'\n", encoding="utf-8")

            with self.assertRaisesRegex(
                CriteriaConfigError,
                "Invalid criteria configuration schema",
            ):
                CriteriaLoader.load(path)


if __name__ == "__main__":
    unittest.main()
