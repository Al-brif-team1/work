"""Tests for the CSV benchmark runner."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark.runner import (
    SKIPPED_EMPTY_GOLD_CLASS,
    BenchmarkRunnerError,
    run_benchmark,
)


class FakeAssessment:
    def __init__(self, recommendation: str) -> None:
        self.recommendation = recommendation


class FakeResult:
    def __init__(self, recommendation: str) -> None:
        self.assessment = FakeAssessment(recommendation)


class FakePipeline:
    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.briefs: list[str] = []

    def analyze_text(self, text: str) -> FakeResult:
        self.briefs.append(text)
        if not self._responses:
            raise AssertionError("Unexpected benchmark pipeline call")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResult(response)


class TestBenchmarkRunner(unittest.TestCase):
    def test_runs_limited_rows_and_writes_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "benchmark.csv"
            output_path = Path(tmpdir) / "predictions.csv"
            self._write_input(
                input_path,
                [
                    {
                        "id": "1",
                        "source": "real",
                        "brief": "Нужно сделать сайт.",
                        "gold_class": "ACCEPT",
                        "comment": "extra fields should not matter",
                    },
                    {
                        "id": "2",
                        "source": "real",
                        "brief": "Нужно уточнить задачу.",
                        "gold_class": "clarify",
                        "comment": "",
                    },
                ],
            )
            pipeline = FakePipeline(["accept", "clarify"])

            stats = run_benchmark(
                input_csv=input_path,
                output_csv=output_path,
                pipeline=pipeline,
                limit=1,
            )

            self.assertEqual(stats.total, 2)
            self.assertEqual(stats.processed, 1)
            self.assertEqual(stats.skipped, 0)
            self.assertEqual(stats.errors, 0)
            self.assertEqual(pipeline.briefs, ["Нужно сделать сайт."])
            self.assertEqual(
                self._read_output(output_path),
                [
                    {
                        "id": "1",
                        "gold_class": "ACCEPT",
                        "predicted_class": "accept",
                        "correct": "true",
                        "error": "",
                        "error_type": "",
                    }
                ],
            )

    def test_row_error_is_recorded_and_next_row_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "benchmark.csv"
            output_path = Path(tmpdir) / "predictions.csv"
            self._write_input(
                input_path,
                [
                    {"id": "1", "brief": "bad", "gold_class": "reject"},
                    {"id": "2", "brief": "good", "gold_class": "clarify"},
                ],
                fieldnames=("id", "brief", "gold_class"),
            )
            pipeline = FakePipeline([RuntimeError("provider failed"), "clarify"])

            stats = run_benchmark(
                input_csv=input_path,
                output_csv=output_path,
                pipeline=pipeline,
            )

            rows = self._read_output(output_path)
            self.assertEqual(stats.total, 2)
            self.assertEqual(stats.processed, 2)
            self.assertEqual(stats.skipped, 0)
            self.assertEqual(stats.errors, 1)
            self.assertEqual(rows[0]["id"], "1")
            self.assertEqual(rows[0]["predicted_class"], "")
            self.assertEqual(rows[0]["correct"], "false")
            self.assertIn("RuntimeError: provider failed", rows[0]["error"])
            self.assertEqual(rows[1]["predicted_class"], "clarify")
            self.assertEqual(rows[1]["correct"], "true")
            self.assertEqual(rows[1]["error"], "")

    def test_empty_gold_class_is_skipped_without_pipeline_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "benchmark.csv"
            output_path = Path(tmpdir) / "predictions.csv"
            self._write_input(
                input_path,
                [
                    {"id": "BRIEF_074", "brief": "Класс уточним позже.", "gold_class": "   "},
                    {"id": "BRIEF_075", "brief": "Готовый бриф.", "gold_class": "ACCEPT"},
                ],
                fieldnames=("id", "brief", "gold_class"),
            )
            pipeline = FakePipeline(["accept"])

            stats = run_benchmark(
                input_csv=input_path,
                output_csv=output_path,
                pipeline=pipeline,
            )

            rows = self._read_output(output_path)
            self.assertEqual(stats.total, 2)
            self.assertEqual(stats.processed, 1)
            self.assertEqual(stats.skipped, 1)
            self.assertEqual(stats.errors, 0)
            self.assertEqual(pipeline.briefs, ["Готовый бриф."])
            self.assertEqual(rows[0]["id"], "BRIEF_074")
            self.assertEqual(rows[0]["predicted_class"], "")
            self.assertEqual(rows[0]["correct"], "")
            self.assertEqual(rows[0]["error"], SKIPPED_EMPTY_GOLD_CLASS)
            self.assertEqual(rows[1]["correct"], "true")

    def test_empty_optional_fields_do_not_skip_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "benchmark.csv"
            output_path = Path(tmpdir) / "predictions.csv"
            self._write_input(
                input_path,
                [
                    {
                        "id": "1",
                        "brief": "Нужно сделать сайт.",
                        "gold_class": "ACCEPT",
                        "missing_fields": "",
                        "reason_codes": "",
                        "label_confidence": "",
                        "comment": "",
                    }
                ],
                fieldnames=(
                    "id",
                    "brief",
                    "gold_class",
                    "missing_fields",
                    "reason_codes",
                    "label_confidence",
                    "comment",
                ),
            )
            pipeline = FakePipeline(["accept"])

            stats = run_benchmark(
                input_csv=input_path,
                output_csv=output_path,
                pipeline=pipeline,
            )

            self.assertEqual(stats.processed, 1)
            self.assertEqual(stats.skipped, 0)
            self.assertEqual(pipeline.briefs, ["Нужно сделать сайт."])
            self.assertEqual(self._read_output(output_path)[0]["correct"], "true")

    def test_unknown_non_empty_gold_class_is_data_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "benchmark.csv"
            output_path = Path(tmpdir) / "predictions.csv"
            self._write_input(
                input_path,
                [{"id": "1", "brief": "Нужно сделать сайт.", "gold_class": "ACCEPTT"}],
                fieldnames=("id", "brief", "gold_class"),
            )
            pipeline = FakePipeline(["accept"])

            with self.assertRaisesRegex(BenchmarkRunnerError, "ACCEPTT"):
                run_benchmark(
                    input_csv=input_path,
                    output_csv=output_path,
                    pipeline=pipeline,
                )

            self.assertEqual(pipeline.briefs, [])

    def test_missing_required_column_fails_before_pipeline_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "benchmark.csv"
            output_path = Path(tmpdir) / "predictions.csv"
            self._write_input(
                input_path,
                [{"id": "1", "brief": "text"}],
                fieldnames=("id", "brief"),
            )
            pipeline = FakePipeline(["accept"])

            with self.assertRaisesRegex(BenchmarkRunnerError, "gold_class"):
                run_benchmark(
                    input_csv=input_path,
                    output_csv=output_path,
                    pipeline=pipeline,
                )

            self.assertEqual(pipeline.briefs, [])

    @staticmethod
    def _write_input(
        path: Path,
        rows: list[dict[str, str]],
        fieldnames: tuple[str, ...] = ("id", "source", "brief", "gold_class", "comment"),
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_output(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))


if __name__ == "__main__":
    unittest.main()
