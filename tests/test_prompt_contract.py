"""Prompt architecture contract tests for active LLM prompts."""

from __future__ import annotations

import unittest
from pathlib import Path

from app.prompts import PromptManager, PromptRenderError


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


class TestPromptContract(unittest.TestCase):
    """Validate the shared prompt-file contract."""

    def test_all_prompt_files_use_front_matter_and_system_user_sections(self) -> None:
        manager = PromptManager([PROMPT_DIR])

        for prompt_path in sorted(PROMPT_DIR.glob("*.md")):
            with self.subTest(prompt=prompt_path.name):
                prompt = manager.load(prompt_path.stem)
                metadata, body = manager._extract_front_matter(prompt.content)
                system, user = manager._split_prompt_sections(body)

                self.assertTrue(metadata, f"{prompt_path.name} has no front matter")
                self.assertTrue(metadata.get("name"))
                self.assertTrue(metadata.get("version"))
                self.assertTrue(metadata.get("description"))
                self.assertTrue(metadata.get("variables"))
                self.assertTrue(metadata.get("output_model"))
                self.assertTrue(system)
                self.assertTrue(user)

    def test_declared_variables_match_template_variables(self) -> None:
        manager = PromptManager([PROMPT_DIR])

        for prompt_path in sorted(PROMPT_DIR.glob("*.md")):
            with self.subTest(prompt=prompt_path.name):
                prompt = manager.load(prompt_path.stem)
                metadata, body = manager._extract_front_matter(prompt.content)
                declared = self._declared_variables(metadata["variables"])
                actual = set(manager._template_variables(body))

                self.assertEqual(declared, actual)

                variables = {name: f"dummy {name}" for name in declared}
                rendered = manager.render(prompt_path.stem, variables=variables)
                self.assertTrue(rendered.system)
                self.assertTrue(rendered.user)

                if declared:
                    missing_name = sorted(declared)[0]
                    missing_variables = {
                        name: value
                        for name, value in variables.items()
                        if name != missing_name
                    }
                    with self.assertRaises(PromptRenderError):
                        manager.render(prompt_path.stem, variables=missing_variables)

    @staticmethod
    def _declared_variables(raw_variables: object) -> set[str]:
        """Parse the comma-separated variable metadata used by prompt files."""
        if not isinstance(raw_variables, str):
            raise AssertionError("prompt variables metadata must be a string")
        return {
            item.strip()
            for item in raw_variables.split(",")
            if item.strip()
        }


if __name__ == "__main__":
    unittest.main()
