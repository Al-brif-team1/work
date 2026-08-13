"""Модуль базы знаний. Он готовит, индексирует и ищет справочные материалы, чтобы ИИ-этапы опирались не только на бриф, но и на контекст проекта."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Класс «Chunk» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    text: str
    start: int
    end: int


class TextChunker:
    """Класс «TextChunker» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0:
            raise ValueError("overlap must be zero or greater")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def chunk_size(self) -> int:
        """Выполняет шаг «chunk size». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._chunk_size

    @property
    def overlap(self) -> int:
        """Выполняет шаг «overlap». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._overlap

    def chunk(self, text: str) -> list[str]:
        """Выполняет шаг «chunk». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return [chunk.text for chunk in self.chunk_with_positions(text)]

    def chunk_with_positions(self, text: str) -> list[Chunk]:
        """Выполняет шаг «chunk with positions». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        normalized = text.strip()
        if not normalized:
            return []

        chunks: list[Chunk] = []
        text_length = len(normalized)
        start = 0

        while start < text_length:
            end = min(start + self._chunk_size, text_length)
            if end < text_length:
                split_point = self._preferred_split(normalized, start, end)
                if split_point > start:
                    end = split_point

            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, start=start, end=end))

            if end >= text_length:
                break

            start = max(end - self._overlap, start + 1)

        return chunks

    @staticmethod
    def _preferred_split(text: str, start: int, end: int) -> int:
        """Выполняет шаг «preferred split». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        newline = text.rfind("\n", start, end)
        if newline > start:
            return newline + 1

        space = text.rfind(" ", start, end)
        if space > start:
            return space + 1

        return end
