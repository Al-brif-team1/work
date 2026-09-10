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

    def test_f1_macro_averages_over_classes_present_in_gold_or_predictions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "ACCEPT", "accept", "true"),
                    self._row("2", "ACCEPT", "clarify", "false"),
                    self._row("3", "CLARIFY", "clarify", "true"),
                    self._row("4", "REJECT", "reject", "true"),
                ],
            )

            metrics = compute_metrics(path)

            # accept: P=1.0 R=0.5 F1=2/3; clarify: P=0.5 R=1.0 F1=2/3; reject: F1=1.0.
            # Три оставшихся класса не встречаются ни в разметке, ни в предсказаниях,
            # поэтому в среднее не входят: (2/3 + 2/3 + 1) / 3.
            self.assertAlmostEqual(metrics.f1_macro, 7 / 9)

    def test_f1_macro_counts_a_class_that_was_only_predicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "ACCEPT", "accept", "true"),
                    self._row("2", "REJECT", "simplify", "false"),
                ],
            )

            metrics = compute_metrics(path)

            # SIMPLIFY нет в разметке, но модель его предсказала, и это ошибка,
            # которую метрика обязана учесть: accept=1.0, reject=0.0, simplify=0.0.
            self.assertAlmostEqual(metrics.f1_macro, 1 / 3)

    def test_error_rate_counts_failed_rows_against_attempted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(
                path,
                [
                    self._row("1", "ACCEPT", "accept", "true"),
                    self._row("2", "REJECT", "", "false", "ExtractorError: boom"),
                    self._row("3", "", "", "", SKIPPED_EMPTY_GOLD_CLASS),
                ],
            )

            metrics = compute_metrics(path)

            # Пропущенная строка в знаменатель не идет: пайплайн ее не запускал.
            self.assertEqual(metrics.errors, 1)
            self.assertEqual(metrics.evaluated, 1)
            self.assertAlmostEqual(metrics.error_rate, 0.5)

    def test_cost_columns_are_aggregated_and_empty_cells_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results_with_cost(
                path,
                [
                    self._cost_row("1", "ACCEPT", "accept", "true", "1.0", "2", "100"),
                    self._cost_row("2", "CLARIFY", "clarify", "true", "2.0", "2", "200"),
                    self._cost_row("3", "REJECT", "reject", "true", "6.0", "3", "300"),
                    self._cost_row("4", "", "", "", "", "", "", SKIPPED_EMPTY_GOLD_CLASS),
                ],
            )

            metrics = compute_metrics(path)

            self.assertAlmostEqual(metrics.latency_mean, 3.0)
            # Медиана нужна ровно затем, чтобы одна долгая строка не выдавала
            # себя за типичное время прогона.
            self.assertAlmostEqual(metrics.latency_median, 2.0)
            self.assertAlmostEqual(metrics.total_tokens_mean, 200.0)
            self.assertAlmostEqual(metrics.llm_calls_mean, 7 / 3)

    def test_results_without_cost_columns_leave_cost_metrics_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results(path, [self._row("1", "ACCEPT", "accept", "true")])

            metrics = compute_metrics(path)
            report = format_metrics(metrics)

            self.assertIsNone(metrics.latency_mean)
            self.assertIsNone(metrics.total_tokens_mean)
            self.assertNotIn("latency_mean_seconds", report)

    def test_format_metrics_reports_f1_macro_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            self._write_results_with_cost(
                path,
                [self._cost_row("1", "ACCEPT", "accept", "true", "1.5", "2", "120")],
            )

            report = format_metrics(compute_metrics(path))

            self.assertIn("f1_macro: 1.000", report)
            self.assertIn("error_rate: 0.000", report)
            self.assertIn("latency_mean_seconds: 1.500", report)
            self.assertIn("llm_calls_per_brief: 2.000", report)
            self.assertIn("total_tokens_per_brief: 120.000", report)

    @staticmethod
    def _row(
        case_id: str,
        gold_class: str,
        predicted_class: str,
        correct: str,
        error: str = "",
    ) -> dict[str, str]:
        return {
            "id": case_id,
            "gold_class": gold_class,
            "predicted_class": predicted_class,
            "correct": correct,
            "error": error,
        }

    @staticmethod
    def _cost_row(
        case_id: str,
        gold_class: str,
        predicted_class: str,
        correct: str,
        latency_seconds: str,
        llm_calls: str,
        total_tokens: str,
        error: str = "",
    ) -> dict[str, str]:
        return {
            "id": case_id,
            "gold_class": gold_class,
            "predicted_class": predicted_class,
            "correct": correct,
            "error": error,
            "latency_seconds": latency_seconds,
            "llm_calls": llm_calls,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _write_results(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("id", "gold_class", "predicted_class", "correct", "error"),
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_results_with_cost(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "id",
                    "gold_class",
                    "predicted_class",
                    "correct",
                    "error",
                    "latency_seconds",
                    "llm_calls",
                    "total_tokens",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
