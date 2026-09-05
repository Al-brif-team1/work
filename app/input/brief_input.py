"""Пакет проекта ИИ-ассистента для анализа проектных брифов Мастерской."""

from __future__ import annotations #для анотации типов

import re #Для регулярных выражений
from pathlib import Path

from app.schemas import BriefInput, BriefInputMetadata

from app.input.document_readers import read_docx

_ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_MULTIPLE_SPACES = re.compile(r"[ \t]{2,}")
_WRAPPER_MARKERS = {
    "<<< BEGIN OF TEXT >>>",
    "<<< END OF TEXT >>>",
    "<<< BEGIN OF CONVERSATION >>>",
    "<<< END OF CONVERSATION >>>",
    "<<< BEGIN OF FILE >>>",
    "<<< END OF FILE >>>",
}


class BriefInputError(RuntimeError):
    """Собственный тип ошибки для проблем при подготовке входного брифа.
    Наследуется от встроенного исключения RuntimeError.
    """


class BriefInputNormalizer:
    """Нормализует текст входного брифа: удаляет технический шум
    и приводит форматирование к единому виду, не изменяя содержание.
    """

    def normalize(self, text: str) -> str:
        """Проверяет и очищает текст брифа, возвращая нормализованную строку."""
        if text is None or not text.strip():
            raise BriefInputError("Brief text must not be empty")

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\x00", "")
        normalized = _ZERO_WIDTH_CHARS.sub("", normalized)

        lines = []
        previous_blank = False
        for raw_line in normalized.split("\n"):
            line = raw_line.rstrip()
            if line.strip() in _WRAPPER_MARKERS:
                continue

            if not line.strip():
                if previous_blank:
                    continue
                lines.append("")
                previous_blank = True
                continue

            previous_blank = False
            leading = re.match(r"^\s*", line).group(0)
            content = line[len(leading) :]
            content = _MULTIPLE_SPACES.sub(" ", content)
            lines.append(f"{leading}{content}")

        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        result = "\n".join(lines)
        if not result.strip():
            raise BriefInputError("Brief text is empty after normalization")

        return result


class BriefInputFactory:
    """Класс «BriefInputFactory» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, normalizer: BriefInputNormalizer | None = None) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        self._normalizer = normalizer or BriefInputNormalizer()

    def from_text(
        self,
        text: str,
        metadata: BriefInputMetadata | None = None,
    ) -> BriefInput:
        """Выполняет шаг «from text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        original_text = self._ensure_text(text)
        normalized_text = self._normalizer.normalize(original_text)
        return BriefInput(
            original_text=original_text,
            normalized_text=normalized_text,
            metadata=metadata or BriefInputMetadata(),
        )

    def from_file(
        self,
        file_path: str | Path,
        metadata: BriefInputMetadata | None = None,
    ) -> BriefInput:
        """Извлекает текст из файла и преобразует его в объект BriefInput."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            if suffix == ".docx":
                original_text = read_docx(path)
            else:
                original_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise BriefInputError(f"Unable to read brief file: {path}") from exc

        input_metadata = metadata or BriefInputMetadata()
        input_metadata = input_metadata.model_copy(
            update={
                "source": "file",
                "input_type": "file",
                "file_path": str(path),
                "file_name": path.name,
            }
        )
        return self.from_text(original_text, metadata=input_metadata)

    @staticmethod
    def _ensure_text(text: str) -> str:
        """Выполняет шаг «ensure text». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        if text is None or not text.strip():
            raise BriefInputError("Brief text must not be empty")

        return text
