"""Minimal CSV benchmark runner for the production brief analysis pipeline."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from app.config import Config
from app.llm.factory import LLMClientFactory
from app.pipeline import BriefAnalysisPipeline
from app.schemas import BriefAnalysisResult, DecisionStatus
from benchmark.telemetry import UsageCollector, UsageSnapshot


REQUIRED_COLUMNS = frozenset({"id", "brief", "gold_class"})
OUTPUT_COLUMNS = (
    "id",
    "gold_class",
    "predicted_class",
    "correct",
    "error",
    "latency_seconds",
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
SKIPPED_EMPTY_GOLD_CLASS = "skipped_empty_gold_class"
PUBLIC_RECOMMENDATIONS = frozenset(
    {
        "accept",
        "clarify",
        "simplify",
        "mentor_review",
        "accept_with_clarifications",
        "reject",
    }
)


class BenchmarkRunnerError(RuntimeError):
    """Raised when benchmark input or arguments are invalid."""


@dataclass(frozen=True)
class BenchmarkRunStats:
    total: int
    processed: int
    skipped: int
    errors: int


class BriefPipeline(Protocol):
    def analyze_text(self, text: str) -> BriefAnalysisResult:
        """Analyze one normalized text brief and return the public result."""


def build_production_pipeline(
    *,
    usage_collector: UsageCollector | None = None,
) -> BriefAnalysisPipeline:
    """Build the same production pipeline used by the application CLI."""
    settings = Config.load()
    llm_client = LLMClientFactory.create(settings, usage_observer=usage_collector)
    return BriefAnalysisPipeline.from_llm_client(llm_client, settings=settings)


def run_benchmark(
    *,
    input_csv: Path,
    output_csv: Path,
    pipeline: BriefPipeline,
    limit: int | None = None,
    usage_collector: UsageCollector | None = None,
) -> BenchmarkRunStats:
    """Run benchmark rows through the production pipeline and write predictions."""
    if limit is not None and limit < 0:
        raise BenchmarkRunnerError("--limit must be greater than or equal to 0")

    processed = 0
    skipped = 0
    errors = 0
    with input_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        _validate_columns(reader.fieldnames)
        rows = list(reader)
        _validate_gold_classes(rows)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            output_file.flush()

            for row in rows:
                if limit is not None and processed >= limit:
                    break

                result_row, was_processed, was_skipped, had_error = _run_row(
                    row,
                    pipeline,
                    usage_collector,
                )
                writer.writerow(result_row)
                output_file.flush()
                if was_processed:
                    processed += 1
                if was_skipped:
                    skipped += 1
                if had_error:
                    errors += 1

    return BenchmarkRunStats(
        total=len(rows),
        processed=processed,
        skipped=skipped,
        errors=errors,
    )


def _validate_columns(fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        raise BenchmarkRunnerError("Input CSV has no header")

    missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
    if missing:
        raise BenchmarkRunnerError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )


def _validate_gold_classes(rows: Sequence[dict[str, str]]) -> None:
    for index, row in enumerate(rows, start=2):
        gold_class = row.get("gold_class", "")
        if not gold_class.strip():
            continue
        if _normalize_class(gold_class) in PUBLIC_RECOMMENDATIONS:
            continue
        case_id = row.get("id", "").strip() or f"line {index}"
        raise BenchmarkRunnerError(
            f"Unknown gold_class for {case_id}: {gold_class!r}"
        )


def _run_row(
    row: dict[str, str],
    pipeline: BriefPipeline,
    usage_collector: UsageCollector | None = None,
) -> tuple[dict[str, str], bool, bool, bool]:
    case_id = row.get("id", "")
    gold_class = row.get("gold_class", "")
    brief = row.get("brief", "")

    if not gold_class.strip():
        return (
            {
                "id": case_id,
                "gold_class": gold_class,
                "predicted_class": "",
                "correct": "",
                "error": SKIPPED_EMPTY_GOLD_CLASS,
            },
            False,
            True,
            False,
        )

    if usage_collector is not None:
        usage_collector.reset()

    started_at = time.perf_counter()
    try:
        result = pipeline.analyze_text(brief)
        predicted_class = result.assessment.recommendation
        error = ""
    except Exception as exc:
        predicted_class = ""
        error = f"{exc.__class__.__name__}: {exc}"
    # Время пишем и для упавших строк: отвалившийся по таймауту бриф стоил
    # столько же, сколько успешный, и это должно быть видно.
    latency_seconds = time.perf_counter() - started_at

    result_row = {
        "id": case_id,
        "gold_class": gold_class,
        "predicted_class": predicted_class,
        "correct": _format_correct(gold_class, predicted_class),
        "error": error,
        "latency_seconds": f"{latency_seconds:.3f}",
    }
    if usage_collector is not None:
        usage = usage_collector.snapshot()
        result_row.update(
            llm_calls=usage.llm_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    return result_row, True, False, bool(error)


def _format_correct(gold_class: str, predicted_class: str) -> str:
    if not predicted_class:
        return "false"
    return str(_normalize_class(gold_class) == _normalize_class(predicted_class)).lower()


def _normalize_class(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return ""

    public_value = normalized.lower()
    if public_value in PUBLIC_RECOMMENDATIONS:
        return public_value

    try:
        return DecisionStatus(normalized.upper()).name
    except ValueError:
        return public_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_runner",
        description="Run a CSV benchmark through the production brief pipeline.",
    )
    parser.add_argument("input_csv", type=Path, help="Path to benchmark CSV.")
    parser.add_argument("output_csv", type=Path, help="Path to output predictions CSV.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of benchmark briefs to process.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        usage_collector = UsageCollector()
        pipeline = build_production_pipeline(usage_collector=usage_collector)
        stats = run_benchmark(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            pipeline=pipeline,
            limit=args.limit,
            usage_collector=usage_collector,
        )
    except BenchmarkRunnerError as exc:
        parser.error(str(exc))
        return 2

    print(
        f"model={Config.load().llm_model} "
        f"total={stats.total} "
        f"processed={stats.processed} "
        f"skipped={stats.skipped} "
        f"errors={stats.errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
