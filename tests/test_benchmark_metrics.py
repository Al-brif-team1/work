"""Tests for benchmark classification metrics."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark.metrics import BenchmarkMetricsError, compute_metrics, format_metrics
from benchmark.runner import SKIPPED_EMPTY_GOLD_CLASS


class TestBenchmarkMetrics(unittest.TestCase):
    def test_all_correct_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "ACCEPT", "accept", "true"),
                    self._row("2", "REJECT", "reject", "true"),
                ],
            )

            metrics = compute_metrics(path)

            self.assertEqual(metrics.total_rows, 2)
            self.assertEqual(metrics.evaluated, 2)
            self.assertEqual(metrics.correct, 2)
            self.assertEqual(metrics.incorrect, 0)
            self.assertEqual(metrics.accuracy, 1.0)
            self.assertEqual(metrics.per_class["accept"].precision, 1.0)
            self.assertEqual(metrics.per_class["accept"].recall, 1.0)
            self.assertEqual(metrics.per_class["accept"].f1, 1.0)
            self.assertEqual(metrics.transitions, {})

    def test_wrong_predictions_accuracy_class_metrics_matrix_and_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "ACCEPT", "accept", "true"),
                    self._row("2", "ACCEPT", "clarify", "false"),
                    self._row("3", "CLARIFY", "clarify", "true"),
                    self._row("4", "REJECT", "clarify", "false"),
                ],
            )

            metrics = compute_metrics(path)

            self.assertEqual(metrics.evaluated, 4)
            self.assertEqual(metrics.correct, 2)
            self.assertEqual(metrics.incorrect, 2)
            self.assertEqual(metrics.accuracy, 0.5)
            self.assertEqual(metrics.confusion_matrix["accept"]["accept"], 1)
            self.assertEqual(metrics.confusion_matrix["accept"]["clarify"], 1)
            self.assertEqual(metrics.confusion_matrix["reject"]["clarify"], 1)
            self.assertEqual(
                metrics.transitions,
                {
                    ("accept", "clarify"): 1,
                    ("reject", "clarify"): 1,
                },
            )
            self.assertAlmostEqual(metrics.per_class["accept"].precision, 1.0)
            self.assertAlmostEqual(metrics.per_class["accept"].recall, 0.5)
            self.assertAlmostEqual(metrics.per_class["accept"].f1, 2 / 3)
            self.assertAlmostEqual(metrics.per_class["clarify"].precision, 1 / 3)
            self.assertAlmostEqual(metrics.per_class["clarify"].recall, 1.0)
            self.assertAlmostEqual(metrics.per_class["clarify"].f1, 0.5)

    def test_missing_class_does_not_break_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(path, [self._row("1", "ACCEPT", "accept", "true")])

            metrics = compute_metrics(path)

            mentor = metrics.per_class["mentor_review"]
            self.assertEqual(mentor.support, 0)
            self.assertEqual(mentor.precision, 0.0)
            self.assertEqual(mentor.recall, 0.0)
            self.assertEqual(mentor.f1, 0.0)
            self.assertEqual(
                sum(metrics.confusion_matrix["mentor_review"].values()),
                0,
            )

    def test_skipped_and_error_rows_are_excluded_from_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "", "", "", SKIPPED_EMPTY_GOLD_CLASS),
                    self._row("2", "REJECT", "", "false", "RuntimeError: failed"),
                    self._row("3", "REJECT", "reject", "true", ""),
                ],
            )

            metrics = compute_metrics(path)

            self.assertEqual(metrics.total_rows, 3)
            self.assertEqual(metrics.skipped, 1)
            self.assertEqual(metrics.errors, 1)
            self.assertEqual(metrics.evaluated, 1)
            self.assertEqual(metrics.correct, 1)
            self.assertEqual(metrics.accuracy, 1.0)

    def test_empty_predictions_file_with_no_evaluated_rows_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(path, [])

            metrics = compute_metrics(path)

            self.assertEqual(metrics.total_rows, 0)
            self.assertEqual(metrics.evaluated, 0)
            self.assertEqual(metrics.correct, 0)
            self.assertEqual(metrics.incorrect, 0)
            self.assertEqual(metrics.accuracy, 0.0)
            self.assertEqual(metrics.per_class["reject"].f1, 0.0)

    def test_row_without_valid_prediction_is_excluded_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(path, [self._row("1", "REJECT", "", "false", "")])

            metrics = compute_metrics(path)

            self.assertEqual(metrics.errors, 1)
            self.assertEqual(metrics.evaluated, 0)
            self.assertEqual(metrics.accuracy, 0.0)

    def test_unknown_non_empty_gold_class_is_data_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(path, [self._row("1", "ACCEPTT", "accept", "false")])

            with self.assertRaisesRegex(BenchmarkMetricsError, "ACCEPTT"):
                compute_metrics(path)

    def test_unknown_non_empty_predicted_class_is_data_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(path, [self._row("1", "ACCEPT", "ACCEPTT", "false")])

            with self.assertRaisesRegex(BenchmarkMetricsError, "ACCEPTT"):
                compute_metrics(path)

    def test_missing_required_column_is_data_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=("id", "gold_class"))
                writer.writeheader()

            with self.assertRaisesRegex(BenchmarkMetricsError, "predicted_class"):
                compute_metrics(path)

    def test_format_metrics_labels_confusion_matrix_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(path, [self._row("1", "REJECT", "clarify", "false")])

            report = format_metrics(compute_metrics(path))

            self.assertIn("accuracy: 0.000", report)
            self.assertIn("rows = gold_class, columns = predicted_class", report)
            self.assertIn("REJECT -> CLARIFY: 1", report)
            self.assertIn("MENTOR_REVIEW", report)

    def test_schema_validation_error_lowers_schema_valid_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "ACCEPT", "accept", "true"),
                    self._row("2", "REJECT", "", "false", "ValidationError: bad payload", "ValidationError"),
                ],
            )

            metrics = compute_metrics(path)

            self.assertEqual(metrics.schema_invalid, 1)
            self.assertEqual(metrics.schema_valid_rate, 0.5)
            self.assertEqual(metrics.pipeline_success_rate, 0.5)

    def test_non_schema_error_keeps_schema_valid_rate_at_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "ACCEPT", "accept", "true"),
                    self._row("2", "REJECT", "", "false", "RuntimeError: timeout", "RuntimeError"),
                ],
            )

            metrics = compute_metrics(path)

            self.assertEqual(metrics.schema_invalid, 0)
            self.assertEqual(metrics.schema_valid_rate, 1.0)
            self.assertEqual(metrics.pipeline_success_rate, 0.5)

    def test_skipped_rows_do_not_affect_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "", "", "", SKIPPED_EMPTY_GOLD_CLASS),
                    self._row("2", "ACCEPT", "accept", "true"),
                ],
            )

            metrics = compute_metrics(path)

            self.assertEqual(metrics.schema_valid_rate, 1.0)
            self.assertEqual(metrics.pipeline_success_rate, 1.0)

    def test_llm_stats_are_aggregated_from_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "id",
                        "gold_class",
                        "predicted_class",
                        "correct",
                        "error",
                        "error_type",
                        "tokens_extractor",
                        "tokens_assessment",
                        "tokens_mvp",
                        "tokens_total",
                        "latency_total",
                        "attempts_total",
                        "llm_calls",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "id": "1",
                            "gold_class": "ACCEPT",
                            "predicted_class": "accept",
                            "correct": "true",
                            "error": "",
                            "error_type": "",
                            "tokens_extractor": "100",
                            "tokens_assessment": "200",
                            "tokens_mvp": "",
                            "tokens_total": "300",
                            "latency_total": "4.000",
                            "attempts_total": "2",
                            "llm_calls": "2",
                        },
                        {
                            "id": "2",
                            "gold_class": "REJECT",
                            "predicted_class": "reject",
                            "correct": "true",
                            "error": "",
                            "error_type": "",
                            "tokens_extractor": "100",
                            "tokens_assessment": "400",
                            "tokens_mvp": "",
                            "tokens_total": "500",
                            "latency_total": "6.000",
                            "attempts_total": "3",
                            "llm_calls": "2",
                        },
                    ]
                )

            metrics = compute_metrics(path)

            self.assertEqual(metrics.tokens_total, 800)
            self.assertEqual(metrics.tokens_per_brief, 400.0)
            self.assertEqual(metrics.tokens_assessment_avg, 300.0)
            self.assertEqual(metrics.tokens_mvp_avg, 0.0)
            self.assertEqual(metrics.latency_avg, 5.0)
            self.assertEqual(metrics.retry_rate, 0.25)

    def test_missing_llm_columns_leave_stats_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(path, [self._row("1", "ACCEPT", "accept", "true")])

            metrics = compute_metrics(path)

            self.assertEqual(metrics.tokens_total, 0)
            self.assertEqual(metrics.tokens_per_brief, 0.0)
            self.assertEqual(metrics.latency_p95, 0.0)        


    @staticmethod
    def _row(
        case_id: str,
        gold_class: str,
        predicted_class: str,
        correct: str,
        error: str = "",
        error_type: str = "",
    ) -> dict[str, str]:
        return {
            "id": case_id,
            "gold_class": gold_class,
            "predicted_class": predicted_class,
            "correct": correct,
            "error": error,
            "error_type": error_type,
        }

    @staticmethod
    def _write_results(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("id", "gold_class", "predicted_class", "correct", "error", "error_type"),
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
