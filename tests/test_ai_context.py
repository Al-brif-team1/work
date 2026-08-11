"""Tests for the shared AIContext model."""

from __future__ import annotations

import unittest

from app.input import BriefInputFactory
from app.schemas import (
    AIContext,
    CompletenessItem,
    CompletenessResult,
    CompletenessStatus,
    Document,
    DocumentMetadata,
    ExtractedBrief,
    ExtractedFact,
    ExtractionResult,
    ExtractorTechnicalInfo,
    FactStatus,
    PipelineInputState,
    PipelineResults,
    PipelineTechnicalState,
    ResponseState,
    RetrievalState,
    SearchResult,
)


def make_extracted_brief() -> ExtractedBrief:
    """Create a minimal extracted brief for context tests."""
    explicit_goal = ExtractedFact(
        status=FactStatus.explicit,
        value="Build a portal",
        evidence=["Build a portal"],
        confidence=0.9,
        notes=None,
    )
    return ExtractedBrief(
        project_goal=explicit_goal,
        tasks=[],
        project_type=ExtractedFact(
            status=FactStatus.explicit,
            value="web_app",
            evidence=["portal"],
            confidence=0.9,
            notes=None,
        ),
        project_direction=ExtractedFact(
            status=FactStatus.explicit,
            value="product development",
            evidence=["product"],
            confidence=0.8,
            notes=None,
        ),
        technologies=[],
        stack=[],
        materials=[],
        expected_result=ExtractedFact(
            status=FactStatus.explicit,
            value="Working portal",
            evidence=["Working portal"],
            confidence=0.9,
            notes=None,
        ),
        constraints=[],
        deadlines=[],
        existing_resources=[],
        integrations=[],
        other_facts=[],
    )


def make_extraction_result() -> ExtractionResult:
    """Create a minimal extraction result for context tests."""
    return ExtractionResult(
        extracted_brief=make_extracted_brief(),
        technical_info=ExtractorTechnicalInfo(
            attempts=1,
            prompt_name="extractor.md",
            trace_enabled=False,
            trace_name="extractor.brief",
            model_name=None,
            raw_response=None,
            recovered_errors=[],
        ),
    )


def make_completeness_result() -> CompletenessResult:
    """Create a minimal completeness result for context tests."""
    return CompletenessResult(
        is_complete=True,
        missing_information=[],
        present_information=[
            CompletenessItem(
                field_key="project_goal",
                field_path="project_goal",
                title="Project goal",
                status=CompletenessStatus.present,
                value="Build a portal",
                reason=None,
                notes=None,
            )
        ],
        clarification_information=[],
        warnings=[],
    )


def make_search_result(document_id: str) -> SearchResult:
    """Create a search result for context tests."""
    return SearchResult(
        document=Document(
            id=document_id,
            text=f"Knowledge for {document_id}",
            metadata=DocumentMetadata(source=f"{document_id}.md"),
        ),
        score=0.9,
        rank=1,
    )


class TestAIContext(unittest.TestCase):
    """Unit tests for AIContext."""

    def test_creates_context_from_brief_and_exposes_text_properties(self) -> None:
        brief_input = BriefInputFactory().from_text("  Build   a portal  ")

        context = AIContext.from_brief(
            brief_input,
            metadata={"request_id": "req-1"},
        )

        self.assertEqual(context.original_text, "  Build   a portal  ")
        self.assertEqual(context.normalized_text, "  Build a portal")
        self.assertEqual(context.metadata["request_id"], "req-1")
        self.assertIsInstance(context.inputs, PipelineInputState)
        self.assertIsInstance(context.results, PipelineResults)
        self.assertIsInstance(context.retrieval, RetrievalState)
        self.assertIsInstance(context.response, ResponseState)
        self.assertIsInstance(context.technical, PipelineTechnicalState)
        self.assertIsNone(context.extracted_brief)

    def test_adds_stage_results_by_copy_without_mutating_original(self) -> None:
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a portal"))
        extraction_result = make_extraction_result()
        completeness_result = make_completeness_result()

        extracted_context = context.with_extraction_result(extraction_result)
        completed_context = extracted_context.with_completeness_result(
            completeness_result
        )

        self.assertIsNone(context.extracted_brief)
        self.assertIs(extracted_context.extraction_result, extraction_result)
        self.assertIs(extracted_context.extracted_brief, extraction_result.extracted_brief)
        self.assertIs(completed_context.completeness_result, completeness_result)

    def test_retrieved_context_can_be_replaced_or_appended(self) -> None:
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a portal"))
        first = make_search_result("doc-1")
        second = make_search_result("doc-2")

        replaced = context.with_retrieved_context([first])
        appended = replaced.append_retrieved_context([second])

        self.assertEqual([item.document.id for item in replaced.retrieved_context], ["doc-1"])
        self.assertEqual(
            [item.document.id for item in appended.retrieved_context],
            ["doc-1", "doc-2"],
        )

    def test_metadata_is_merged_by_copy(self) -> None:
        context = AIContext.from_brief(
            BriefInputFactory().from_text("Build a portal"),
            metadata={"request_id": "req-1"},
        )

        updated = context.with_metadata(user_id="user-1")

        self.assertEqual(context.metadata, {"request_id": "req-1"})
        self.assertEqual(
            updated.metadata,
            {"request_id": "req-1", "user_id": "user-1"},
        )

    def test_final_response_rejects_blank_text(self) -> None:
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a portal"))

        with self.assertRaises(ValueError):
            context.with_final_response("   ")

    def test_context_is_immutable_by_copy(self) -> None:
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a portal"))

        with self.assertRaises(Exception):
            context.inputs = PipelineInputState(brief_input=context.brief_input)

    def test_final_response_is_stored_in_response_state(self) -> None:
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a portal"))

        updated = context.with_final_response(
            "  Final response  ",
            {"status": "ACCEPT"},
        )

        self.assertIsNone(context.final_response_text)
        self.assertEqual(updated.response.text, "Final response")
        self.assertEqual(updated.final_response_text, "Final response")
        self.assertEqual(updated.final_response_payload, {"status": "ACCEPT"})

    def test_stage_metadata_is_merged_by_stage_name(self) -> None:
        context = AIContext.from_brief(BriefInputFactory().from_text("Build a portal"))

        first = context.with_stage_metadata("extractor", attempts=1)
        second = first.with_stage_metadata("extractor", latency_seconds=0.1)

        self.assertEqual(context.stage_metadata, {})
        self.assertEqual(
            second.stage_metadata["extractor"],
            {"attempts": 1, "latency_seconds": 0.1},
        )


if __name__ == "__main__":
    unittest.main()
