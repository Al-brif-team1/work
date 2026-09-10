"""Compute basic classification metrics for benchmark prediction CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from benchmark.runner import PUBLIC_RECOMMENDATIONS, SKIPPED_EMPTY_GOLD_CLASS


REQUIRED_COLUMNS = frozenset(
    {"id", "gold_class", "predicted_class", "correct", "error", "error_type"}
)
CLASS_ORDER = (
    "accept",
    "accept_with_clarifications",
    "clarify",
    "simplify",
    "mentor_review",
    "reject",
)
CLASS_LABELS = {
    "accept": "ACCEPT",
    "accept_with_clarifications": "ACCEPT_WITH_CLARIFICATIONS",
    "clarify": "CLARIFY",
    "simplify": "SIMPLIFY",
    "mentor_review": "MENTOR_REVIEW",
    "reject": "REJECT",
}


class BenchmarkMetricsError(RuntimeError):
    """Raised when benchmark results cannot be scored safely."""


@dataclass(frozen=True)
class ClassMetrics:
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class BenchmarkMetrics:
    total_rows: int
    evaluated: int
    skipped: int
    errors: int
    correct: int
    incorrect: int
    accuracy: float
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    transitions: dict[tuple[str, str], int] = field(default_factory=dict)
    schema_invalid: int = 0
    schema_valid_rate: float = 0.0
    pipeline_success_rate: float = 0.0
    tokens_total: int = 0
    tokens_per_brief: float = 0.0
    tokens_extractor_avg: float = 0.0
    tokens_assessment_avg: float = 0.0
    tokens_mvp_avg: float = 0.0
    latency_avg: float = 0.0
    latency_p95: float = 0.0
    retry_rate: float = 0.0


def compute_metrics(results_csv: Path) -> BenchmarkMetrics:
    """Read benchmark predictions and compute classification metrics."""
    with results_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        _validate_columns(reader.fieldnames)
        rows = list(reader)

    evaluated_pairs: list[tuple[str, str]] = []
    skipped = 0
    errors = 0
    schema_invalid = 0
    stats_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=2):
        gold_class = row.get("gold_class", "")
        predicted_class = row.get("predicted_class", "")
        error = row.get("error", "")
        error_type = row.get("error_type", "")

        if not gold_class.strip():
            skipped += 1
            continue
        gold = _parse_class(gold_class, row=row, column="gold_class", line_number=index)

        if error.strip():
            if error.strip() == SKIPPED_EMPTY_GOLD_CLASS:
                skipped += 1
            else:
                errors += 1
                if error_type == "ValidationError":
                    schema_invalid += 1
            continue

        if not predicted_class.strip():
            errors += 1
            continue
        predicted = _parse_class(
            predicted_class,
            row=row,
            column="predicted_class",
            line_number=index,
        )

        evaluated_pairs.append((gold, predicted))
        stats_rows.append(row)

    correct = sum(1 for gold, predicted in evaluated_pairs if gold == predicted)
    evaluated = len(evaluated_pairs)
    processed = len(rows) - skipped
    incorrect = evaluated - correct
    accuracy = _safe_divide(correct, evaluated)
    confusion_matrix = _build_confusion_matrix(evaluated_pairs)

    llm_stats = _build_llm_stats(stats_rows)

    return BenchmarkMetrics(
        total_rows=len(rows),
        evaluated=evaluated,
        skipped=skipped,
        errors=errors,
        correct=correct,
        incorrect=incorrect,
        accuracy=accuracy,
        per_class=_build_per_class_metrics(confusion_matrix),
        confusion_matrix=confusion_matrix,
        transitions=_build_transitions(evaluated_pairs),
        schema_invalid=schema_invalid,
        schema_valid_rate=_safe_divide(processed - schema_invalid, processed),
        pipeline_success_rate=_safe_divide(processed - errors, processed),
        **llm_stats,
    )


def _validate_columns(fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        raise BenchmarkMetricsError("Results CSV has no header")

    missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
    if missing:
        raise BenchmarkMetricsError(
            "Results CSV is missing required columns: " + ", ".join(missing)
        )


def _parse_class(
    value: str,
    *,
    row: dict[str, str],
    column: str,
    line_number: int,
) -> str:
    normalized = value.strip().lower()
    if normalized in PUBLIC_RECOMMENDATIONS:
        return normalized

    case_id = row.get("id", "").strip() or f"line {line_number}"
    raise BenchmarkMetricsError(
        f"Unknown {column} for {case_id}: {value!r}"
    )


def _build_confusion_matrix(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    matrix = {
        gold: {predicted: 0 for predicted in CLASS_ORDER}
        for gold in CLASS_ORDER
    }
    for gold, predicted in pairs:
        matrix[gold][predicted] += 1
    return matrix


def _build_per_class_metrics(
    matrix: dict[str, dict[str, int]],
) -> dict[str, ClassMetrics]:
    metrics: dict[str, ClassMetrics] = {}
    for class_name in CLASS_ORDER:
        true_positive = matrix[class_name][class_name]
        false_positive = sum(
            matrix[gold][class_name]
            for gold in CLASS_ORDER
            if gold != class_name
        )
        false_negative = sum(
            matrix[class_name][predicted]
            for predicted in CLASS_ORDER
            if predicted != class_name
        )
        support = sum(matrix[class_name].values())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        metrics[class_name] = ClassMetrics(
            support=support,
            precision=precision,
            recall=recall,
            f1=f1,
        )
    return metrics


def _build_transitions(
    pairs: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    counter: Counter[tuple[str, str]] = Counter(
        (gold, predicted)
        for gold, predicted in pairs
        if gold != predicted
    )
    return dict(sorted(counter.items()))


def _build_llm_stats(rows: Sequence[dict[str, str]]) -> dict[str, float | int]:
    def _values(column: str) -> list[float | int]:
        result = []
        for row in rows:
            raw = row.get(column, "").strip()
            if raw:
                result.append(float(raw))
        return result

    totals = _values("tokens_total")
    latencies = _values("latency_total")
    attempts = _values("attempts_total")
    calls = _values("llm_calls")

    return {
        "tokens_total": int(sum(totals)),
        "tokens_per_brief": _safe_divide(sum(totals), len(totals)),
        "tokens_extractor_avg": _safe_divide(sum(_values("tokens_extractor")), len(totals)),
        "tokens_assessment_avg": _safe_divide(sum(_values("tokens_assessment")), len(totals)),
        "tokens_mvp_avg": _safe_divide(sum(_values("tokens_mvp")), len(totals)),
        "latency_avg": _safe_divide(sum(latencies), len(latencies)),
        "latency_p95": _percentile(latencies, 95),
        "retry_rate": _safe_divide(sum(attempts) - sum(calls), sum(calls)),
    }


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * percentile / 100), len(ordered) - 1)
    return ordered[index]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_metrics(metrics: BenchmarkMetrics) -> str:
    """Render metrics as a compact console report."""
    lines = [
        "Benchmark metrics",
        "",
        "Summary:",
        f"  total_rows: {metrics.total_rows}",
        f"  evaluated: {metrics.evaluated}",
        f"  skipped: {metrics.skipped}",
        f"  errors: {metrics.errors}",
        f"  correct: {metrics.correct}",
        f"  incorrect: {metrics.incorrect}",
        f"  accuracy: {_format_float(metrics.accuracy)}",
        f"  schema_invalid: {metrics.schema_invalid}",
        f"  schema_valid_rate: {_format_float(metrics.schema_valid_rate)}",
        f"  pipeline_success_rate: {_format_float(metrics.pipeline_success_rate)}",
        "",
        "LLM usage:",
        f"  tokens_total: {metrics.tokens_total}",
        f"  tokens_per_brief: {_format_float(metrics.tokens_per_brief)}",
        f"  tokens_extractor_avg: {_format_float(metrics.tokens_extractor_avg)}",
        f"  tokens_assessment_avg: {_format_float(metrics.tokens_assessment_avg)}",
        f"  tokens_mvp_avg: {_format_float(metrics.tokens_mvp_avg)}",
        f"  latency_avg: {_format_float(metrics.latency_avg)}",
        f"  latency_p95: {_format_float(metrics.latency_p95)}",
        f"  retry_rate: {_format_float(metrics.retry_rate)}",
        "",
        "Per-class metrics:",
        _format_table(
            ["class", "support", "precision", "recall", "f1"],
            [
                [
                    CLASS_LABELS[class_name],
                    str(class_metrics.support),
                    _format_float(class_metrics.precision),
                    _format_float(class_metrics.recall),
                    _format_float(class_metrics.f1),
                ]
                for class_name, class_metrics in metrics.per_class.items()
            ],
        ),
        "",
        "Confusion matrix (rows = gold_class, columns = predicted_class):",
        _format_confusion_matrix(metrics.confusion_matrix),
        "",
        "Classification errors (gold_class -> predicted_class):",
    ]
    if metrics.transitions:
        lines.extend(
            f"  {CLASS_LABELS[gold]} -> {CLASS_LABELS[predicted]}: {count}"
            for (gold, predicted), count in metrics.transitions.items()
        )
    else:
        lines.append("  none")
    return "\n".join(lines)


def _format_confusion_matrix(matrix: dict[str, dict[str, int]]) -> str:
    headers = ["gold \\ predicted", *[CLASS_LABELS[name] for name in CLASS_ORDER]]
    rows = [
        [CLASS_LABELS[gold], *[str(matrix[gold][predicted]) for predicted in CLASS_ORDER]]
        for gold in CLASS_ORDER
    ]
    return _format_table(headers, rows)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(row[index])) for row in (headers, *rows))
        for index in range(len(headers))
    ]
    rendered = [
        "  "
        + "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(headers)),
        "  "
        + "  ".join("-" * width for width in widths),
    ]
    rendered.extend(
        "  "
        + "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(rendered)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_metrics",
        description="Compute benchmark classification metrics from predictions CSV.",
    )
    parser.add_argument("results_csv", type=Path, help="Path to benchmark predictions CSV.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        metrics = compute_metrics(args.results_csv)
    except BenchmarkMetricsError as exc:
        parser.error(str(exc))
        return 2

    print(format_metrics(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
