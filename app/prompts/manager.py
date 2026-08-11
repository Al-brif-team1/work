"""Centralized prompt loading and caching."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_TEMPLATE_VARIABLE_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


class PromptManagerError(RuntimeError):
    """Base error raised by prompt loading infrastructure."""


class PromptNotFoundError(PromptManagerError):
    """Raised when a requested prompt cannot be found."""


class PromptRenderError(PromptManagerError):
    """Raised when a prompt cannot be rendered with supplied variables."""


@dataclass(frozen=True)
class Prompt:
    """Loaded prompt template with source metadata."""

    name: str
    content: str
    path: Path
    version: str | None = None
    metadata: dict[str, Any] | None = None


class RenderedPrompt(BaseModel):
    """Rendered prompt with separated system/user messages and metadata."""

    name: str
    version: str | None = None
    system: str
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class PromptManager:
    """Load prompt templates by name from one or more directories."""

    DEFAULT_EXTENSION = ".md"

    def __init__(
        self,
        prompt_directories: list[str | Path] | tuple[str | Path, ...] | None = None,
    ) -> None:
        """Create a manager with ordered prompt search directories."""
        directories = prompt_directories or (self.default_prompt_directory(),)
        self._prompt_directories = tuple(Path(directory) for directory in directories)
        self._cache: dict[tuple[str, str | None], Prompt] = {}

    @property
    def prompt_directories(self) -> tuple[Path, ...]:
        """Return the ordered prompt search directories."""
        return self._prompt_directories

    def load(self, name: str, version: str | None = None) -> Prompt:
        """Load a prompt by name and optional version."""
        normalized_name = self._normalize_name(name)
        normalized_version = self._normalize_version(version)
        cache_key = (normalized_name, normalized_version)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        path = self._resolve_prompt_path(normalized_name, normalized_version)
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PromptManagerError(f"Unable to read prompt file: {path}") from exc

        prompt = Prompt(
            name=normalized_name,
            content=content,
            path=path,
            version=normalized_version,
            metadata=self._extract_front_matter(content)[0],
        )
        self._cache[cache_key] = prompt
        return prompt

    def render(
        self,
        name: str,
        variables: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> RenderedPrompt:
        """Load and render a prompt with strict template-variable validation."""
        prompt = self.load(name, version=version)
        metadata, body = self._extract_front_matter(prompt.content)
        effective_metadata = {
            **(prompt.metadata or {}),
            **metadata,
        }
        system_template, user_template = self._split_prompt_sections(body)
        variables = variables or {}

        rendered_system = self._render_template(
            system_template,
            variables,
            prompt_name=prompt.name,
        )
        rendered_user = (
            self._render_template(
                user_template,
                variables,
                prompt_name=prompt.name,
            )
            if user_template is not None
            else None
        )

        return RenderedPrompt(
            name=str(effective_metadata.get("name") or prompt.name),
            version=str(effective_metadata.get("version") or prompt.version)
            if effective_metadata.get("version") or prompt.version
            else None,
            system=rendered_system,
            user=rendered_user,
            metadata=effective_metadata,
        )

    def clear_cache(self) -> None:
        """Clear all cached prompt templates."""
        self._cache.clear()

    def _resolve_prompt_path(
        self,
        name: str,
        version: str | None,
    ) -> Path:
        """Find the first matching prompt path in configured directories."""
        candidates = self._candidate_names(name, version)
        searched_paths: list[Path] = []

        for directory in self._prompt_directories:
            for candidate in candidates:
                path = directory / candidate
                searched_paths.append(path)
                if path.is_file():
                    return path

        searched = ", ".join(str(path) for path in searched_paths)
        version_text = f" version {version}" if version is not None else ""
        raise PromptNotFoundError(
            f"Prompt '{name}'{version_text} was not found. Searched: {searched}"
        )

    @classmethod
    def _candidate_names(cls, name: str, version: str | None) -> tuple[str, ...]:
        """Build filename candidates for plain and versioned prompts."""
        path = Path(name)
        suffix = path.suffix or cls.DEFAULT_EXTENSION
        stem = path.name[: -len(path.suffix)] if path.suffix else path.name

        if version is None:
            filename = path.name if path.suffix else f"{path.name}{suffix}"
            return (filename,)

        return (
            str(Path(stem) / f"v{version}{suffix}"),
            f"{stem}.v{version}{suffix}",
            f"{stem}@{version}{suffix}",
        )

    @classmethod
    def _extract_front_matter(cls, content: str) -> tuple[dict[str, Any], str]:
        """Extract a minimal YAML front matter block from prompt content."""
        normalized = content.lstrip()
        if not normalized.startswith("---"):
            return {}, content

        lines = normalized.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, content

        metadata_lines: list[str] = []
        closing_index: int | None = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_index = index
                break
            metadata_lines.append(line)

        if closing_index is None:
            raise PromptRenderError("Prompt front matter is missing closing delimiter")

        metadata = cls._parse_front_matter(metadata_lines)
        body = "\n".join(lines[closing_index + 1 :]).lstrip("\n")
        return metadata, body

    @staticmethod
    def _parse_front_matter(lines: list[str]) -> dict[str, Any]:
        """Parse the simple key-value YAML subset used by prompt metadata."""
        metadata: dict[str, Any] = {}
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise PromptRenderError(f"Invalid prompt metadata line: {raw_line}")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise PromptRenderError("Prompt metadata keys must not be empty")
            metadata[key] = _parse_metadata_scalar(value)
        return metadata

    @staticmethod
    def _split_prompt_sections(content: str) -> tuple[str, str | None]:
        """Split prompt body into system and optional user sections."""
        lines = content.splitlines()
        system_start: int | None = None
        user_start: int | None = None

        for index, line in enumerate(lines):
            marker = line.strip().lower()
            if marker in {"system", "# system", "## system"}:
                system_start = index + 1
                continue
            if marker in {"user", "# user", "## user"}:
                user_start = index + 1
                break

        if system_start is None:
            return content.strip(), None

        system_end = user_start - 1 if user_start is not None else len(lines)
        system = "\n".join(lines[system_start:system_end]).strip()
        user = "\n".join(lines[user_start:]).strip() if user_start is not None else None
        return system, user or None

    @classmethod
    def _render_template(
        cls,
        template: str,
        variables: dict[str, Any],
        *,
        prompt_name: str,
    ) -> str:
        """Render ``{{variable}}`` placeholders and reject missing values."""
        expected = set(cls._template_variables(template))
        missing = sorted(name for name in expected if name not in variables)
        if missing:
            raise PromptRenderError(
                f"Prompt '{prompt_name}' is missing template variables: {missing}"
            )

        def replace(match: re.Match[str]) -> str:
            variable_name = match.group(1).strip()
            return _stringify_variable(variables[variable_name])

        return _TEMPLATE_VARIABLE_RE.sub(replace, template).strip()

    @staticmethod
    def _template_variables(template: str) -> list[str]:
        """Return template variable names from ``{{variable}}`` placeholders."""
        return [match.group(1).strip() for match in _TEMPLATE_VARIABLE_RE.finditer(template)]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize and validate a prompt name."""
        normalized = name.strip()
        if not normalized:
            raise ValueError("prompt name must not be empty")
        path = Path(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("prompt name must be relative and stay within prompt directories")
        return normalized

    @staticmethod
    def _normalize_version(version: str | None) -> str | None:
        """Normalize an optional prompt version identifier."""
        if version is None:
            return None
        normalized = version.strip()
        if not normalized:
            raise ValueError("prompt version must not be empty")
        if any(char in normalized for char in ("/", "\\", ":")):
            raise ValueError("prompt version must not contain path separators")
        return normalized

    @staticmethod
    def default_prompt_directory() -> Path:
        """Return the project default prompt directory."""
        return Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=16)
def get_prompt_manager(
    prompt_directories: tuple[str | Path, ...] | None = None,
) -> PromptManager:
    """Return a shared prompt manager for reusable stage instances."""
    directories = prompt_directories or (PromptManager.default_prompt_directory(),)
    normalized_directories = tuple(Path(directory) for directory in directories)
    return PromptManager(normalized_directories)


def clear_prompt_manager_cache() -> None:
    """Clear cached prompt managers and their loaded prompt templates."""
    get_prompt_manager.cache_clear()


def _parse_metadata_scalar(value: str) -> Any:
    """Parse a small YAML-like scalar subset for prompt front matter."""
    if value == "":
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _stringify_variable(value: Any) -> str:
    """Render a template variable as prompt-safe text."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)
