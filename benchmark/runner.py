"""Minimal CSV benchmark runner for the production brief analysis pipeline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.config import Config
from app.llm.factory import LLMClientFactory
from app.pipeline import BriefAnalysisPipeline, BriefAnalysisPipelineError
from app.schemas import BriefAnalysisResult, DecisionStatus


REQUIRED_COLUMNS = frozenset({"id", "brief", "gold_class"})
OUTPUT_COLUMNS = (
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


@dataclass(frozen=True)
class BriefLLMStats:
    """Technical statistics collected from one brief run."""

    tokens_extractor: int | None = None
    tokens_assessment: int | None = None
    tokens_mvp: int | None = None
    tokens_total: int | None = None
    latency_total: float | None = None
    attempts_total: int = 0
    llm_calls: int = 0


class BriefPipeline(Protocol):
    def analyze_text_context(self, text: str) -> Any:
        """Analyze one text brief and return the full pipeline context."""


def build_production_pipeline() -> BriefAnalysisPipeline:
    """Build the same production pipeline used by the application CLI."""
    settings = Config.load()
    llm_client = LLMClientFactory.create(settings)
    return BriefAnalysisPipeline.from_llm_client(llm_client, settings=settings)


def run_benchmark(
    *,
    input_csv: Path,
    output_csv: Path,
    pipeline: BriefPipeline,
    limit: int | None = None,
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
                "error_type": "",
                **_format_llm_stats(BriefLLMStats()),
            },
            False,
            True,
            False,
        )

    try:
        context = pipeline.analyze_text_context(brief)
        if context.final_response_payload is None:
            raise BriefAnalysisPipelineError("Pipeline did not produce final payload")
        result = BriefAnalysisResult.model_validate(context.final_response_payload)
        predicted_class = result.assessment.recommendation
        stats = _collect_llm_stats(context)
        error = ""
        error_type = ""
    except Exception as exc:
        predicted_class = ""
        stats = BriefLLMStats()
        error = f"{exc.__class__.__name__}: {exc}"
        error_type = exc.__class__.__name__

    return (
        {
            "id": case_id,
            "gold_class": gold_class,
            "predicted_class": predicted_class,
            "correct": _format_correct(gold_class, predicted_class),
            "error": error,
            "error_type": error_type,
            **_format_llm_stats(stats),
        },
        True,
        False,
        bool(error),
    )


def _stage_tokens(technical_info: Any) -> int | None:
    """Return total tokens for one stage, falling back to the estimate."""
    usage = getattr(technical_info, "token_usage", None)
    if usage is None:
        return None
    return usage.total_tokens or usage.total_tokens_estimate


def _collect_llm_stats(context: Any) -> BriefLLMStats:
    """Aggregate token usage and latency across all LLM stages of one brief."""
    stages = [
        context.extraction_result,
        context.assessment_result,
        context.mvp_planning_result,
    ]
    tokens: list[int | None] = []
    latency = 0.0
    attempts = 0
    calls = 0

    for stage in stages:
        info = getattr(stage, "technical_info", None) if stage is not None else None
        if info is None:
            tokens.append(None)
            continue
        stage_tokens = _stage_tokens(info)
        tokens.append(stage_tokens)
        latency += getattr(info, "latency_seconds", None) or 0.0
        attempts += getattr(info, "attempts", 0) or 0
        if stage_tokens is not None:
            calls += 1

    known = [value for value in tokens if value is not None]
    return BriefLLMStats(
        tokens_extractor=tokens[0],
        tokens_assessment=tokens[1],
        tokens_mvp=tokens[2],
        tokens_total=sum(known) if known else None,
        latency_total=latency or None,
        attempts_total=attempts,
        llm_calls=calls,
    )


def _format_llm_stats(stats: BriefLLMStats) -> dict[str, str]:
    """Render technical statistics as CSV-safe strings."""

    def _num(value: int | float | None) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    return {
        "tokens_extractor": _num(stats.tokens_extractor),
        "tokens_assessment": _num(stats.tokens_assessment),
        "tokens_mvp": _num(stats.tokens_mvp),
        "tokens_total": _num(stats.tokens_total),
        "latency_total": _num(stats.latency_total),
        "attempts_total": _num(stats.attempts_total),
        "llm_calls": _num(stats.llm_calls),
    }


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
        pipeline = build_production_pipeline()
        stats = run_benchmark(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            pipeline=pipeline,
            limit=args.limit,
        )
    except BenchmarkRunnerError as exc:
        parser.error(str(exc))
        return 2

    print(
        f"total={stats.total} "
        f"processed={stats.processed} "
        f"skipped={stats.skipped} "
        f"errors={stats.errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())