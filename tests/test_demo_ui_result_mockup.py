"""Static regression tests for Demo UI result rendering."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_JS = REPO_ROOT / "demo_ui" / "static" / "app.js"


class TestDemoUiResultMockup(unittest.TestCase):
    def test_large_result_mockup_uses_same_payload_renderer(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('const resultMockup = document.querySelector("#result-mock");', source)
        self.assertIn("renderResultMockup(payload, resultMockup);", source)
        self.assertIn(
            'const finalStatus = textOrFallback(decision.final_status, "UNKNOWN");',
            source,
        )
        self.assertIn("if (status) status.textContent = finalStatus;", source)

    def test_large_result_mockup_does_not_add_frontend_status_classification(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        for final_status in (
            "ACCEPT",
            "ACCEPT_WITH_CLARIFICATIONS",
            "SIMPLIFY",
            "REJECT",
            "MENTOR_REVIEW",
        ):
            with self.subTest(final_status=final_status):
                self.assertTrue(final_status not in source or final_status == "SIMPLIFY")


if __name__ == "__main__":
    unittest.main()
