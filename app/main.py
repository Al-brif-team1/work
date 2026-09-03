"""Точка входа CLI-приложения. Она принимает бриф пользователя, запускает конвейер анализа и печатает итоговый ответ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from app.config import Config
from app.input import BriefInputError, BriefInputFactory
from app.llm.factory import LLMClientFactory
from app.pipeline import AssessmentStage, BriefAnalysisPipeline, BriefAnalysisPipelineError
from app.schemas import AIContext, AssessmentResult, BriefAnalysisResult


def print_traffic_light_diagnostics(result: AssessmentResult) -> None:
    """Print CLI-only traffic-light diagnostics without changing public JSON."""
    traffic_light = result.traffic_light
    print("\n[TRAFFIC LIGHT DIAGNOSTICS]", file=sys.stderr)
    print(f"status={traffic_light.status.value}", file=sys.stderr)
    print(f"direction={traffic_light.direction}", file=sys.stderr)
    print(f"specialization={traffic_light.specialization}", file=sys.stderr)
    if not traffic_light.matches:
        print("matches: []", file=sys.stderr)
        return

    print("matches:", file=sys.stderr)
    for match in traffic_light.matches:
        print(f"  - task={match.task}", file=sys.stderr)
        print(f"    matched_rule={match.matched_rule}", file=sys.stderr)
        print(f"    status={match.status.value}", file=sys.stderr)
        print(f"    reason={match.reason}", file=sys.stderr)


class TrafficLightDiagnosticsStage:
    """CLI-only diagnostics hook that does not modify pipeline context."""

    def run_context(self, context: AIContext) -> AIContext:
        if context.assessment_result is not None:
            print_traffic_light_diagnostics(context.assessment_result)
        return context


def add_traffic_light_diagnostics_stage(pipeline: BriefAnalysisPipeline) -> None:
    """Insert CLI-only traffic-light diagnostics right after assessment."""
    pipeline.insert_stage_after(AssessmentStage, TrafficLightDiagnosticsStage())


def build_parser() -> argparse.ArgumentParser:
    """Выполняет шаг «build parser». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    parser = argparse.ArgumentParser(
        prog="ai_assistant",
        description="Analyze one project brief and return structured JSON.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--text",
        help="Brief text passed directly on the command line.",
    )
    source_group.add_argument(
        "--file",
        type=Path,
        help="Path to a file containing the brief text.",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Only load and normalize the brief without calling the LLM pipeline.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    factory = BriefInputFactory()

    try:
        brief_input = (
            factory.from_file(args.file)
            if args.file is not None
            else factory.from_text(args.text)
        )
    except BriefInputError as exc:
        parser.error(str(exc))
        return 2

    if args.normalize_only:
        print(
            json.dumps(
                brief_input.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        settings = Config.load()
        llm_client = LLMClientFactory.create(settings)
        pipeline = BriefAnalysisPipeline.from_llm_client(
            llm_client,
            settings=settings,
        )
        add_traffic_light_diagnostics_stage(pipeline)
        context = pipeline.run_context(brief_input)
        if context.assessment_result is None:
            raise BriefAnalysisPipelineError("Pipeline did not produce assessment result")
        if context.final_response_payload is None:
            raise BriefAnalysisPipelineError("Pipeline did not produce final payload")
        result = BriefAnalysisResult.model_validate(context.final_response_payload)
    except (RuntimeError, BriefAnalysisPipelineError) as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    """Выполняет шаг «main». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
