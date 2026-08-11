"""Application CLI entry module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from app.config import Config
from app.input import BriefInputError, BriefInputFactory
from app.llm.factory import LLMClientFactory
from app.pipeline import BriefAnalysisPipeline, BriefAnalysisPipelineError


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for brief input."""
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
    """Execute the CLI entry point."""
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
            model_name=settings.openrouter_model,
        )
        result = pipeline.analyze(brief_input)
    except (RuntimeError, BriefAnalysisPipelineError) as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    """CLI entry point for console scripts."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
