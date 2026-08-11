"""Tests for centralized prompt loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.prompts import (
    PromptManager,
    PromptNotFoundError,
    PromptRenderError,
    RenderedPrompt,
    clear_prompt_manager_cache,
    get_prompt_manager,
)


class TestPromptManager(unittest.TestCase):
    """Unit tests for PromptManager."""

    def tearDown(self) -> None:
        clear_prompt_manager_cache()

    def test_loads_prompt_by_name_and_caches_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "extractor.md"
            prompt_path.write_text("first version", encoding="utf-8")
            manager = PromptManager([tmp_dir])

            first = manager.load("extractor")
            prompt_path.write_text("changed on disk", encoding="utf-8")
            second = manager.load("extractor")

        self.assertEqual(first.content, "first version")
        self.assertIs(first, second)
        self.assertEqual(first.name, "extractor")
        self.assertEqual(first.path.name, "extractor.md")

    def test_searches_multiple_directories_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir:
            with tempfile.TemporaryDirectory() as second_dir:
                (Path(second_dir) / "risk.md").write_text(
                    "second directory",
                    encoding="utf-8",
                )
                manager = PromptManager([first_dir, second_dir])

                prompt = manager.load("risk.md")

        self.assertEqual(prompt.content, "second directory")

    def test_missing_prompt_reports_searched_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = PromptManager([tmp_dir])

            with self.assertRaises(PromptNotFoundError) as context:
                manager.load("missing")

        self.assertIn("Prompt 'missing' was not found", str(context.exception))
        self.assertIn("missing.md", str(context.exception))

    def test_loads_versioned_prompt_with_dot_version_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "versioned_prompt.v2.md").write_text(
                "version two",
                encoding="utf-8",
            )
            manager = PromptManager([tmp_dir])

            prompt = manager.load("versioned_prompt", version="2")

        self.assertEqual(prompt.content, "version two")
        self.assertEqual(prompt.version, "2")
        self.assertEqual(prompt.path.name, "versioned_prompt.v2.md")

    def test_loads_versioned_prompt_with_at_version_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "self_check@stable.md").write_text(
                "stable prompt",
                encoding="utf-8",
            )
            manager = PromptManager([tmp_dir])

            prompt = manager.load("self_check", version="stable")

        self.assertEqual(prompt.content, "stable prompt")
        self.assertEqual(prompt.version, "stable")
        self.assertEqual(prompt.path.name, "self_check@stable.md")

    def test_rejects_path_traversal_names(self) -> None:
        manager = PromptManager([])

        with self.assertRaises(ValueError):
            manager.load("../secret")

    def test_shared_prompt_manager_reuses_loaded_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "extractor.md"
            prompt_path.write_text("cached template", encoding="utf-8")
            manager = get_prompt_manager((tmp_dir,))

            first = manager.load("extractor")
            prompt_path.write_text("changed on disk", encoding="utf-8")
            second_manager = get_prompt_manager((tmp_dir,))
            second = second_manager.load("extractor")

        self.assertIs(manager, second_manager)
        self.assertIs(first, second)
        self.assertEqual(second.content, "cached template")

    def test_render_splits_front_matter_system_and_user_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "extractor.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: extractor",
                        'version: "1"',
                        "description: Extract facts.",
                        "output_model: ExtractedBrief",
                        "---",
                        "# System",
                        "Extract facts only.",
                        "# User",
                        "Brief: {{brief_text}}",
                    ]
                ),
                encoding="utf-8",
            )
            manager = PromptManager([tmp_dir])

            rendered = manager.render(
                "extractor",
                variables={"brief_text": "Build a portal"},
            )

        self.assertIsInstance(rendered, RenderedPrompt)
        self.assertEqual(rendered.name, "extractor")
        self.assertEqual(rendered.version, "1")
        self.assertEqual(rendered.system, "Extract facts only.")
        self.assertEqual(rendered.user, "Brief: Build a portal")
        self.assertEqual(rendered.metadata["output_model"], "ExtractedBrief")

    def test_render_rejects_missing_template_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "extractor.md").write_text(
                "# System\nExtract.\n# User\nBrief: {{brief_text}}",
                encoding="utf-8",
            )
            manager = PromptManager([tmp_dir])

            with self.assertRaises(PromptRenderError) as context:
                manager.render("extractor", variables={})

        self.assertIn("missing template variables", str(context.exception))
        self.assertIn("brief_text", str(context.exception))

    def test_render_supports_directory_version_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_dir = Path(tmp_dir) / "assessment"
            prompt_dir.mkdir()
            (prompt_dir / "v1.md").write_text(
                "---\nname: assessment\nversion: 1\n---\n# System\nAssess {{topic}}.",
                encoding="utf-8",
            )
            manager = PromptManager([tmp_dir])

            rendered = manager.render(
                "assessment",
                version="1",
                variables={"topic": "brief"},
            )

        self.assertEqual(rendered.name, "assessment")
        self.assertEqual(rendered.version, "1")
        self.assertEqual(rendered.system, "Assess brief.")

    def test_render_uses_cached_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "extractor.md"
            prompt_path.write_text("# System\nFirst {{value}}", encoding="utf-8")
            manager = PromptManager([tmp_dir])

            first = manager.render("extractor", variables={"value": "A"})
            prompt_path.write_text("# System\nSecond {{value}}", encoding="utf-8")
            second = manager.render("extractor", variables={"value": "B"})

        self.assertEqual(first.system, "First A")
        self.assertEqual(second.system, "First B")

    def test_render_without_sections_treats_body_as_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "mvp.md").write_text(
                "Plan MVP for {{project}}.",
                encoding="utf-8",
            )
            manager = PromptManager([tmp_dir])

            rendered = manager.render("mvp", variables={"project": "portal"})

        self.assertEqual(rendered.system, "Plan MVP for portal.")
        self.assertIsNone(rendered.user)

    def test_default_extractor_prompt_uses_front_matter_and_user_brief(self) -> None:
        manager = PromptManager()
        brief_text = "Build a support bot."

        rendered = manager.render(
            "extractor",
            variables={"brief_text": brief_text},
        )

        self.assertEqual(rendered.name, "extractor")
        self.assertEqual(rendered.version, "1")
        self.assertEqual(rendered.metadata["output_model"], "ExtractedBrief")
        self.assertIn("factual extractor", rendered.system)
        self.assertNotIn(brief_text, rendered.system)
        self.assertIn(brief_text, rendered.user or "")


if __name__ == "__main__":
    unittest.main()
