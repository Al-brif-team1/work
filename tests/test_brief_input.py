import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.input import BriefInputError, BriefInputFactory, BriefInputNormalizer
from app.main import run
from app.schemas import (
    AIContext,
    AssessmentRecommendation,
    AssessmentResult,
    BriefAnalysisResult,
)


class TestBriefInputNormalizer(unittest.TestCase):
    """Класс «TestBriefInputNormalizer» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_normalize_text_preserves_meaning_and_cleans_transport_junk(self) -> None:
        normalizer = BriefInputNormalizer()

        normalized = normalizer.normalize(
            "  Line  one\r\n\r\n<<< BEGIN OF TEXT >>>\n"
            "  Line   two   \n"
            "\u200bLine three\x00\r\n\r\n\r\n"
            "<<< END OF TEXT >>>\n"
        )

        self.assertEqual(
            normalized,
            "  Line one\n\n  Line two\nLine three",
        )

    def test_normalize_rejects_empty_text(self) -> None:
        normalizer = BriefInputNormalizer()

        with self.assertRaisesRegex(BriefInputError, "must not be empty"):
            normalizer.normalize("   \n\t  ")


class TestBriefInputFactory(unittest.TestCase):
    """Класс «TestBriefInputFactory» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_from_text_preserves_original_text(self) -> None:
        factory = BriefInputFactory()

        brief_input = factory.from_text("  Hello   world  ")

        self.assertEqual(brief_input.original_text, "  Hello   world  ")
        self.assertEqual(brief_input.normalized_text, "  Hello world")
        self.assertEqual(brief_input.metadata.source, "cli")
        self.assertEqual(brief_input.metadata.input_type, "text")

    def test_from_file_reads_file_and_sets_metadata(self) -> None:
        factory = BriefInputFactory()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "brief.txt"
            path.write_text("  Brief from file  ", encoding="utf-8")

            brief_input = factory.from_file(path)

        self.assertEqual(brief_input.original_text, "  Brief from file  ")
        self.assertEqual(brief_input.normalized_text, "  Brief from file")
        self.assertEqual(brief_input.metadata.source, "file")
        self.assertEqual(brief_input.metadata.input_type, "file")
        self.assertEqual(brief_input.metadata.file_path, str(path))
        self.assertEqual(brief_input.metadata.file_name, "brief.txt")

    def test_from_file_raises_for_missing_path(self) -> None:
        factory = BriefInputFactory()

        with self.assertRaisesRegex(BriefInputError, "Unable to read brief file"):
            factory.from_file(Path("missing-file.txt"))


class TestBriefCli(unittest.TestCase):
    """Класс «TestBriefCli» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def test_run_accepts_text_argument(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = run(["--text", "Project brief", "--normalize-only"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"normalized_text": "Project brief"', output.getvalue())

    def test_run_writes_final_json_to_stdout_and_diagnostics_to_stderr(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        result = BriefAnalysisResult.model_validate(
            {
                "summary": "Project summary",
                "extracted_fields": {
                    "goal": "Build a project",
                    "expected_result": "Working result",
                    "tasks": [],
                    "domain": "",
                    "direction": "development",
                    "available_materials": [],
                    "missing_information": [],
                    "complexity_factors": [],
                },
                "assessment": {
                    "recommendation": "accept",
                    "confidence": "high",
                    "reasons": ["Criterion explanation"],
                    "risks": [],
                },
                "clarifying_questions": [],
                "mvp_suggestion": "",
                "customer_response_draft": "Customer response",
            }
        )

        class PipelineStub:
            def insert_stage_after(self, stage_type, stage):
                return False

            def run_context(self, brief_input):
                print(
                    "[ARBITRATION DIAGNOSTICS] result: matched_rule=accept_ready",
                    file=stderr,
                )
                return (
                    AIContext.from_brief(brief_input)
                    .with_assessment_result(
                        AssessmentResult(
                            criterion_evaluations=[],
                            risks=[],
                            evidence=[],
                            has_risks=False,
                            recommendation=AssessmentRecommendation.ready_for_arbitration,
                        )
                    )
                    .with_final_response(
                        response_text=result.customer_response_draft,
                        response_payload=result.model_dump(mode="json"),
                    )
                )

        with (
            patch(
                "app.main.Config.load",
                return_value=SimpleNamespace(openrouter_model="test-model"),
            ),
            patch("app.main.LLMClientFactory.create", return_value=object()),
            patch(
                "app.main.BriefAnalysisPipeline.from_llm_client",
                return_value=PipelineStub(),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = run(["--text", "Project brief"])

        self.assertEqual(exit_code, 0)
        self.assertNotIn("[ARBITRATION DIAGNOSTICS]", stdout.getvalue())
        self.assertIn("[ARBITRATION DIAGNOSTICS]", stderr.getvalue())

        parsed = json.loads(stdout.getvalue())
        self.assertEqual(parsed["summary"], "Project summary")
        self.assertNotIn("[ARBITRATION DIAGNOSTICS]", json.dumps(parsed))


if __name__ == "__main__":
    unittest.main()
