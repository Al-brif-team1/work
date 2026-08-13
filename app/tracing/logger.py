import logging
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings


class LoggerFactory:
    """Класс «LoggerFactory» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    LOGGER_NAME = "ai_assistant"
    LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def create(settings: Settings) -> logging.Logger:
        """Выполняет шаг «create». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        level = LoggerFactory._resolve_level(settings.log_level)
        logger = logging.getLogger(LoggerFactory.LOGGER_NAME)
        logger.setLevel(level)
        logger.propagate = False

        if logger.handlers:
            for handler in logger.handlers:
                handler.setLevel(level)
            return logger

        formatter = logging.Formatter(
            fmt=LoggerFactory.LOG_FORMAT,
            datefmt=LoggerFactory.DATE_FORMAT,
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        file_handler = logging.FileHandler(
            filename=LoggerFactory._log_file(),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def _resolve_level(level: str) -> int:
        """Находит нужное поле внутри вложенной структуры данных. Это похоже на движение по адресу: шаг за шагом до конкретного значения."""
        resolved_level = logging.getLevelName(level.upper())
        if isinstance(resolved_level, int):
            return resolved_level

        raise ValueError(f"Unsupported log level: {level}")

    @staticmethod
    def _log_file() -> Path:
        """Выполняет шаг «log file». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "app.log"


@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    """Возвращает уже подготовленный объект или настройку, чтобы остальные части проекта использовали единый источник."""
    return LoggerFactory.create(settings=get_settings())
