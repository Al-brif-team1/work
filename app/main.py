"""Точка входа CLI-приложения. Она принимает бриф пользователя, запускает конвейер анализа и печатает итоговый ответ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from app.config import Config
from app.input import BriefInputError, BriefInputFactory
from app.llm.factory import LLMClientFactory
from app.llm.runner import (
    LLMRunnerProviderError,
    LLMRunnerStructuredOutputError,
    LLMRunnerTimeoutError,
)
from app.pipeline import AssessmentStage, BriefAnalysisPipeline, BriefAnalysisPipelineError
from app.pipeline.extractor import Extractor
from app.schemas import AIContext, BriefAnalysisResult


def print_model_diagnostics(header: str, model: Any) -> None:
    """Print CLI-only model diagnostics without changing public JSON."""
    print(f"\n[{header}]", file=sys.stderr)
    print(
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )


class ExtractorDiagnosticsStage:
    """CLI-only extractor diagnostics hook that does not modify pipeline context."""

    def run_context(self, context: AIContext) -> AIContext:
        if context.extraction_result is not None:
            print_model_diagnostics(
                "EXTRACTOR DIAGNOSTICS",
                context.extraction_result,
            )
        return context


class AssessmentDiagnosticsStage:
    """CLI-only assessment diagnostics hook that does not modify pipeline context."""

    def run_context(self, context: AIContext) -> AIContext:
        if context.assessment_result is not None:
            print_model_diagnostics(
                "ASSESSMENT DIAGNOSTICS",
                context.assessment_result,
            )
        return context


def add_cli_diagnostics_stages(pipeline: BriefAnalysisPipeline) -> None:
    """Insert CLI-only diagnostics after LLM stages."""
    pipeline.insert_stage_after(Extractor, ExtractorDiagnosticsStage())
    pipeline.insert_stage_after(AssessmentStage, AssessmentDiagnosticsStage())


def format_pipeline_error(exc: Exception) -> str:
    """Return a safe short CLI message for known technical LLM failures."""
    structured_error = _find_exception(exc, LLMRunnerStructuredOutputError)
    llm_error = _find_exception(exc, LLMRunnerProviderError)
    timeout_error = _find_exception(exc, LLMRunnerTimeoutError)
    if timeout_error is not None:
        return "Провайдер LLM не ответил вовремя. Попробуйте позже."

    if structured_error is not None:
        error_kind = getattr(structured_error, "error_kind", None)
        if error_kind == "empty_content":
            return "Модель вернула пустой ответ. Попробуйте повторить запрос."
        if error_kind == "length_finish":
            return "Ответ модели был обрезан из-за лимита генерации. Попробуйте сократить бриф или увеличить лимит ответа."
        if error_kind == "truncated_json":
            return "Ответ модели был обрезан и не может быть разобран. Попробуйте повторить запрос."
        return "Модель вернула некорректный структурированный ответ. Попробуйте повторить запрос."

    if llm_error is None:
        return str(exc)

    if llm_error.status_code in {401, 403}:
        return "Ошибка доступа к провайдеру LLM."
    if llm_error.status_code == 402:
        return "Недостаточно доступа или кредитов у провайдера LLM."
    if llm_error.status_code == 429:
        return "Провайдер LLM временно ограничил частоту запросов. Попробуйте позже."
    if llm_error.status_code in {500, 502, 503, 504}:
        return "Провайдер LLM временно недоступен. Попробуйте позже."
    if llm_error.retryable:
        return "Провайдер LLM временно недоступен. Попробуйте позже."

    return "Запрос к провайдеру LLM завершился ошибкой."


def _find_exception(exc: BaseException, target_type: type[BaseException]) -> BaseException | None:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, target_type):
            return current
        current = current.__cause__
    return None


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
        add_cli_diagnostics_stages(pipeline)
        context = pipeline.run_context(brief_input)
        if context.assessment_result is None:
            raise BriefAnalysisPipelineError("Pipeline did not produce assessment result")
        if context.final_response_payload is None:
            raise BriefAnalysisPipelineError("Pipeline did not produce final payload")
        result = BriefAnalysisResult.model_validate(context.final_response_payload)
    except (RuntimeError, BriefAnalysisPipelineError) as exc:
        print(f"Pipeline error: {format_pipeline_error(exc)}", file=sys.stderr)
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
