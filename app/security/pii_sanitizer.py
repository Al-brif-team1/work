import re
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class PIISanitizer:
    """Класс «PIISanitizer» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    _mapping: dict[str, str] = field(default_factory=dict, init=False)
    _counters: dict[str, int] = field(default_factory=dict, init=False)

    _patterns: ClassVar[tuple[tuple[str, re.Pattern[str]], ...]] = (
        (
            "EMAIL",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ),
        (
            "URL",
            re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE),
        ),
        (
            "UUID",
            re.compile(
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
                r"[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-"
                r"[0-9a-fA-F]{12}\b"
            ),
        ),
        (
            "IP",
            re.compile(
                r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
                r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
            ),
        ),
        (
            "CARD",
            re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
        ),
        (
            "SNILS",
            re.compile(r"\b\d{3}-\d{3}-\d{3}[ -]?\d{2}\b"),
        ),
        (
            "INN",
            re.compile(r"\b(?:\d{10}|\d{12})\b"),
        ),
        (
            "PHONE",
            re.compile(
                r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)"
                r"\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\w)"
            ),
        ),
        (
            "TELEGRAM",
            re.compile(r"(?<!\w)@[A-Za-z0-9_]{5,32}\b"),
        ),
        (
            "NAME",
            re.compile(
                r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+"
                r"(?:\s+[А-ЯЁ][а-яё]+)?\b|"
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+"
                r"(?:\s+[A-Z][a-z]+)?\b"
            ),
        ),
    )

    def sanitize(self, text: str) -> str:
        """Выполняет шаг «sanitize». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        sanitized = text

        for pii_type, pattern in self._patterns:
            sanitized = pattern.sub(
                lambda match: self._replace(pii_type, match.group(0)),
                sanitized,
            )

        return sanitized

    def restore(self, text: str) -> str:
        """Выполняет шаг «restore». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        restored = text

        for placeholder, original_value in sorted(
            self._mapping.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            restored = restored.replace(placeholder, original_value)

        return restored

    @property
    def mapping(self) -> dict[str, str]:
        """Выполняет шаг «mapping». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._mapping.copy()

    def _replace(self, pii_type: str, value: str) -> str:
        existing_placeholder = self._find_existing_placeholder(value)
        if existing_placeholder is not None:
            return existing_placeholder

        placeholder = self._next_placeholder(pii_type)
        self._mapping[placeholder] = value
        return placeholder

    def _find_existing_placeholder(self, value: str) -> str | None:
        for placeholder, original_value in self._mapping.items():
            if original_value == value:
                return placeholder

        return None

    def _next_placeholder(self, pii_type: str) -> str:
        next_value = self._counters.get(pii_type, 0) + 1
        self._counters[pii_type] = next_value
        return f"<{pii_type}_{next_value}>"
