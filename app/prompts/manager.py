"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_TEMPLATE_VARIABLE_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


class PromptManagerError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class PromptNotFoundError(PromptManagerError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class PromptRenderError(PromptManagerError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


@dataclass(frozen=True)
class Prompt:
    """Класс «Prompt» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    name: str
    content: str
    path: Path
    version: str | None = None
    metadata: dict[str, Any] | None = None


class RenderedPrompt(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    name: str
    version: str | None = None
    system: str
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class PromptManager:
    """Класс «PromptManager» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    DEFAULT_EXTENSION = ".md"

    def __init__(
        self,
        prompt_directories: list[str | Path] | tuple[str | Path, ...] | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        directories = prompt_directories or (self.default_prompt_directory(),)
        self._prompt_directories = tuple(Path(directory) for directory in directories)
        self._cache: dict[tuple[str, str | None], Prompt] = {}

    @property
    def prompt_directories(self) -> tuple[Path, ...]:
        """Выполняет шаг «prompt directories». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._prompt_directories

    def load(self, name: str, version: str | None = None) -> Prompt:
        """Выполняет шаг «load». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «render». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «clear cache». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        self._cache.clear()

    def _resolve_prompt_path(
        self,
        name: str,
        version: str | None,
    ) -> Path:
        """Находит нужное поле внутри вложенной структуры данных. Это похоже на движение по адресу: шаг за шагом до конкретного значения."""
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
        """Выполняет шаг «candidate names». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Выполняет шаг «extract front matter». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Разбирает текстовое значение и превращает его в программный объект. Так код дальше работает не с произвольной строкой, а с понятной структурой."""
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
        """Выполняет шаг «split prompt sections». Документация описывает назначение метода, а сама логика остается в коде ниже."""
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
        """Готовит человекочитаемый текст из внутренних данных. Это нужно для промптов, объяснений или финального ответа."""
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
        """Выполняет шаг «template variables». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return [match.group(1).strip() for match in _TEMPLATE_VARIABLE_RE.finditer(template)]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
        normalized = name.strip()
        if not normalized:
            raise ValueError("prompt name must not be empty")
        path = Path(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("prompt name must be relative and stay within prompt directories")
        return normalized

    @staticmethod
    def _normalize_version(version: str | None) -> str | None:
        """Приводит текст или данные к единому виду. Смысл не меняется: мы только убираем лишний шум, чтобы код дальше сравнивал значения надежнее."""
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
        """Выполняет шаг «default prompt directory». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=16)
def get_prompt_manager(
    prompt_directories: tuple[str | Path, ...] | None = None,
) -> PromptManager:
    """Возвращает уже подготовленный объект или настройку, чтобы остальные части проекта использовали единый источник."""
    directories = prompt_directories or (PromptManager.default_prompt_directory(),)
    normalized_directories = tuple(Path(directory) for directory in directories)
    return PromptManager(normalized_directories)


def clear_prompt_manager_cache() -> None:
    """Выполняет шаг «clear prompt manager cache». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    get_prompt_manager.cache_clear()


def _parse_metadata_scalar(value: str) -> Any:
    """Разбирает текстовое значение и превращает его в программный объект. Так код дальше работает не с произвольной строкой, а с понятной структурой."""
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
    """Выполняет шаг «stringify variable». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)
