import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from app.input import BriefInputError, BriefInputFactory, BriefInputNormalizer
from app.main import run


class TestBriefInputNormalizer(unittest.TestCase):
    """Tests for brief text normalization."""

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
    """Tests for building validated brief input models."""

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
    """Tests for the CLI programmatic entry point."""

    def test_run_accepts_text_argument(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = run(["--text", "Project brief", "--normalize-only"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"normalized_text": "Project brief"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
