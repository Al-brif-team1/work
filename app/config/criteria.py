"""Модуль конфигурации проекта. Он загружает настройки и criteria.yaml, чтобы детерминированные роботы работали по понятным правилам."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class CriteriaConfigError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class CriteriaYAMLSyntaxError(CriteriaConfigError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class DecisionThresholds(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    min_score: int | None = None
    max_score: int | None = None
    conditions: list[str] = Field(default_factory=list)
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class Criterion(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    key: str
    title: str
    description: str
    complexity: str | None = None
    allowed_values: list[str] = Field(default_factory=list)
    decision_thresholds: DecisionThresholds | None = None
    status_signals: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RequiredField(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    key: str
    field_path: str
    title: str
    description: str
    required: bool = True
    allowed_values: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TaskType(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    key: str
    title: str
    description: str
    criteria: list[str] = Field(default_factory=list)
    complexity: str | None = None

    model_config = ConfigDict(extra="forbid")


class ProjectType(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    key: str
    title: str
    description: str
    task_types: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RestrictedTopic(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    key: str
    title: str
    reason_kind: str
    customer_reason: str
    keywords: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RestrictedTopicsConfiguration(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    version: str
    description: str
    topics: list[RestrictedTopic] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RiskType(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    key: str
    title: str
    description: str
    severity_hint: str | None = None
    signals: list[str] = Field(default_factory=list)
    evidence_hints: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RiskAnalysisConfiguration(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    version: str
    description: str
    risk_types: list[RiskType]
    decision_thresholds: list[DecisionThresholds] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ArbitrationCondition(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    field: str
    operator: str
    value: Any | None = None
    case_sensitive: bool = False

    model_config = ConfigDict(extra="forbid")


class ArbitrationRule(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    key: str
    title: str
    description: str
    status: str
    conditions: list[ArbitrationCondition] = Field(default_factory=list)
    confidence: float | None = None

    model_config = ConfigDict(extra="forbid")


class ArbitrationConfiguration(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    version: str
    description: str
    default_status: str
    rules: list[ArbitrationRule]

    model_config = ConfigDict(extra="forbid")


class EvaluationConfiguration(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    version: str
    description: str
    project_types: list[ProjectType]
    task_types: list[TaskType]
    criteria: list[Criterion]
    required_fields: list[RequiredField]
    decision_thresholds: list[DecisionThresholds] = Field(default_factory=list)
    # Необязательная: урезанные конфиги в тестах арбитра описывают только правила,
    # и без запрещённых тем они должны грузиться по-прежнему.
    restricted_topics: RestrictedTopicsConfiguration | None = None
    risk_analysis: RiskAnalysisConfiguration | None = None
    arbitration: ArbitrationConfiguration | None = None

    model_config = ConfigDict(extra="forbid")


class CriteriaConfig(BaseModel):
    """[СТРУКТУРА ДАННЫХ] Это класс-чертеж для хранения информации. Он следит, чтобы данные не перепутались: Pydantic проверяет поля, типы и обязательные значения перед передачей между роботами конвейера."""

    evaluation: EvaluationConfiguration

    model_config = ConfigDict(extra="forbid")


class CriteriaLoader:
    """Класс «CriteriaLoader» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    @staticmethod
    def load(path: Path | None = None) -> CriteriaConfig:
        """Выполняет шаг «load». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        config_path = path or CriteriaLoader.default_path()

        try:
            raw_text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CriteriaConfigError(
                f"Unable to read criteria configuration: {config_path}"
            ) from exc

        try:
            data = _load_yaml_like(raw_text)
        except CriteriaYAMLSyntaxError as exc:
            raise CriteriaConfigError(
                f"Invalid YAML syntax in criteria configuration: {config_path}"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise CriteriaConfigError(
                f"Invalid YAML syntax in criteria configuration: {config_path}"
            ) from exc

        try:
            return CriteriaConfig.model_validate(data)
        except ValidationError as exc:
            raise CriteriaConfigError(
                f"Invalid criteria configuration schema: {exc}"
            ) from exc

    @staticmethod
    def default_path() -> Path:
        """Выполняет шаг «default path». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return Path(__file__).resolve().parents[2] / "config" / "criteria.yaml"


@lru_cache(maxsize=1)
def get_criteria_config() -> CriteriaConfig:
    """Возвращает уже подготовленный объект или настройку, чтобы остальные части проекта использовали единый источник."""
    return CriteriaLoader.load()


def _load_yaml_like(text: str) -> dict[str, Any]:
    """Выполняет шаг «load yaml like». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    lines = _preprocess_yaml_lines(text)
    if not lines:
        raise CriteriaYAMLSyntaxError("Criteria configuration is empty")

    value, index = _parse_block(lines, 0, 0)
    if index != len(lines):
        raise CriteriaYAMLSyntaxError("Unexpected trailing content in criteria YAML")
    if not isinstance(value, dict):
        raise CriteriaYAMLSyntaxError(
            "Criteria YAML must contain a mapping at top level"
        )
    return value


def _preprocess_yaml_lines(text: str) -> list[tuple[int, str]]:
    """Выполняет шаг «preprocess yaml lines». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    result: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        result.append((indent, stripped))

    return result


def _parse_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    """Разбирает текстовое значение и превращает его в программный объект. Так код дальше работает не с произвольной строкой, а с понятной структурой."""
    if index >= len(lines):
        return {}, index

    current_indent, content = lines[index]
    if current_indent < indent:
        return {}, index

    if content.startswith("- "):
        return _parse_list(lines, index, indent)

    return _parse_mapping(lines, index, indent)


def _parse_mapping(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    """Разбирает текстовое значение и превращает его в программный объект. Так код дальше работает не с произвольной строкой, а с понятной структурой."""
    mapping: dict[str, Any] = {}

    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise CriteriaYAMLSyntaxError("Invalid indentation in YAML mapping")
        if content.startswith("- "):
            break

        key, value_text = _split_key_value(content)
        index += 1

        if value_text is not None:
            mapping[key] = _parse_scalar(value_text)
            continue

        if index >= len(lines) or lines[index][0] <= indent:
            mapping[key] = {}
            continue

        child_indent = lines[index][0]
        child_value, index = _parse_block(lines, index, child_indent)
        mapping[key] = child_value

    return mapping, index


def _parse_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    """Разбирает текстовое значение и превращает его в программный объект. Так код дальше работает не с произвольной строкой, а с понятной структурой."""
    items: list[Any] = []

    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise CriteriaYAMLSyntaxError("Invalid indentation in YAML list")
        if not content.startswith("- "):
            break

        item_text = content[2:].strip()
        index += 1

        if not item_text:
            if index >= len(lines) or lines[index][0] <= indent:
                items.append(None)
                continue

            child_indent = lines[index][0]
            child_value, index = _parse_block(lines, index, child_indent)
            items.append(child_value)
            continue

        if ":" in item_text:
            key, value_text = _split_key_value(item_text)
            item: dict[str, Any] = {
                key: _parse_scalar(value_text) if value_text is not None else {}
            }

            if index < len(lines) and lines[index][0] > indent:
                child_indent = lines[index][0]
                child_value, index = _parse_block(lines, index, child_indent)
                if not isinstance(child_value, dict):
                    raise CriteriaConfigError(
                        "List items that start as mappings must continue as mappings"
                    )
                if value_text is None:
                    item[key] = child_value
                else:
                    item.update(child_value)

            items.append(item)
            continue

        items.append(_parse_scalar(item_text))

    return items, index


def _split_key_value(content: str) -> tuple[str, str | None]:
    """Выполняет шаг «split key value». Документация описывает назначение метода, а сама логика остается в коде ниже."""
    if ":" not in content:
        raise CriteriaYAMLSyntaxError(f"Invalid YAML line: {content}")

    key, value_text = content.split(":", 1)
    key = key.strip()
    value_text = value_text.strip()
    return key, value_text or None


def _parse_scalar(value_text: str) -> Any:
    """Разбирает текстовое значение и превращает его в программный объект. Так код дальше работает не с произвольной строкой, а с понятной структурой."""
    if value_text is None:
        return None

    if value_text in {"null", "Null", "NULL", "~"}:
        return None
    if value_text in {"true", "True", "TRUE"}:
        return True
    if value_text in {"false", "False", "FALSE"}:
        return False

    if value_text.startswith(("'", '"')) and value_text.endswith(("'", '"')):
        return value_text[1:-1]

    if value_text.isdigit() or (
        value_text.startswith("-") and value_text[1:].isdigit()
    ):
        return int(value_text)

    try:
        return float(value_text)
    except ValueError:
        return value_text
