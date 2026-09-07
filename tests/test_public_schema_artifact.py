"""Tests for the saved public JSON Schema artifact."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.schemas import BriefAnalysisResult


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "brief_analysis_result.schema.json"
PUBLIC_PROPERTIES = {
    "summary",
    "extracted_fields",
    "assessment",
    "clarifying_questions",
    "mvp_suggestion",
    "customer_response_draft",
}


class TestPublicSchemaArtifact(unittest.TestCase):
    def test_public_schema_artifact_is_in_sync_with_model(self) -> None:
        with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
            saved_schema = json.load(schema_file)

        self.assertEqual(saved_schema, BriefAnalysisResult.model_json_schema())

    def test_public_schema_has_expected_top_level_properties(self) -> None:
        with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
            saved_schema = json.load(schema_file)

        self.assertEqual(
            set(saved_schema["properties"]),
            PUBLIC_PROPERTIES,
        )


if __name__ == "__main__":
    unittest.main()
