"""Модуль базы знаний. Он готовит, индексирует и ищет справочные материалы, чтобы ИИ-этапы опирались не только на бриф, но и на контекст проекта."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.schemas import Document, DocumentMetadata


@dataclass(frozen=True)
class DocumentLoaderConfig:
    """Класс «DocumentLoaderConfig» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    base_directory: Path
    supported_extensions: tuple[str, ...] = (".md", ".txt")


class DocumentLoaderError(RuntimeError):
    """Специальная ошибка этого участка системы. Она помогает явно показать, на каком шаге конвейера что-то пошло не так."""


class DocumentLoader:
    """Класс «DocumentLoader» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(
        self,
        base_directory: str | Path = Path("knowledge"),
        supported_extensions: tuple[str, ...] = (".md", ".txt"),
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        self._config = DocumentLoaderConfig(
            base_directory=Path(base_directory),
            supported_extensions=supported_extensions,
        )

    @property
    def base_directory(self) -> Path:
        """Выполняет шаг «base directory». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return self._config.base_directory

    def load_directory(self, directory: str | Path | None = None) -> list[Document]:
        """Выполняет шаг «load directory». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        root = Path(directory) if directory is not None else self.base_directory
        if not root.exists():
            raise DocumentLoaderError(f"Knowledge directory does not exist: {root}")
        if not root.is_dir():
            raise DocumentLoaderError(f"Knowledge path is not a directory: {root}")

        documents: list[Document] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in self._config.supported_extensions:
                documents.append(self.load_file(path))

        return documents

    def load_file(self, path: str | Path) -> Document:
        """Выполняет шаг «load file». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        file_path = Path(path)
        if not file_path.exists():
            raise DocumentLoaderError(f"Document does not exist: {file_path}")
        if file_path.suffix.lower() not in self._config.supported_extensions:
            raise DocumentLoaderError(f"Unsupported document type: {file_path.suffix}")

        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DocumentLoaderError(f"Unable to read document: {file_path}") from exc

        relative_path = self._relative_path(file_path)
        metadata = DocumentMetadata(
            source=relative_path.as_posix(),
            title=file_path.stem,
            document_type=file_path.suffix.lstrip(".").lower(),
            category=self._category_for(relative_path),
            relative_path=relative_path.as_posix(),
            extension=file_path.suffix.lower(),
            size_bytes=file_path.stat().st_size,
        )

        return Document(
            id=relative_path.as_posix(),
            text=text,
            metadata=metadata,
        )

    def _relative_path(self, file_path: Path) -> Path:
        """Выполняет шаг «relative path». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        try:
            return file_path.relative_to(self.base_directory)
        except ValueError:
            return file_path.resolve()

    @staticmethod
    def _category_for(relative_path: Path) -> str:
        """Выполняет шаг «category for». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        parent = relative_path.parent
        if str(parent) in {".", ""}:
            return "root"

        return parent.as_posix()
