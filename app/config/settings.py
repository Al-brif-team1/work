from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Класс «Settings» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    llm_api_key: str = Field(..., alias="LLM_API_KEY")
    llm_model: str = Field(..., alias="LLM_MODEL")
    llm_base_url: str = Field(..., alias="LLM_BASE_URL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int | None = Field(None, alias="LLM_MAX_TOKENS")
    llm_top_p: float | None = Field(None, alias="LLM_TOP_P")
    llm_max_attempts: int = Field(2, alias="LLM_MAX_ATTEMPTS")
    llm_timeout_seconds: float = Field(60.0, alias="LLM_TIMEOUT_SECONDS")
    llm_transport_retries: int = Field(0, alias="LLM_TRANSPORT_RETRIES")
    langfuse_public_key: str | None = Field(None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", alias="LANGFUSE_HOST")
    debug: bool = Field(False, alias="DEBUG")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO",
        alias="LOG_LEVEL",
    )
    knowledge_directory: Path = Field(Path("knowledge"), alias="KNOWLEDGE_DIR")
    knowledge_chunk_size: int = Field(1000, alias="KNOWLEDGE_CHUNK_SIZE")
    knowledge_chunk_overlap: int = Field(200, alias="KNOWLEDGE_CHUNK_OVERLAP")
    knowledge_top_k: int = Field(5, alias="KNOWLEDGE_TOP_K")

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator(
        "knowledge_chunk_size",
        "knowledge_chunk_overlap",
        "knowledge_top_k",
        "llm_max_attempts",
        "llm_timeout_seconds",
    )
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if value <= 0:
            raise ValueError("must be greater than zero")

        return value

    @field_validator("llm_max_tokens")
    @classmethod
    def _validate_optional_positive(cls, value: int | None) -> int | None:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if value is not None and value <= 0:
            raise ValueError("must be greater than zero")

        return value

    @field_validator("llm_transport_retries")
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if value < 0:
            raise ValueError("must not be negative")

        return value

    @field_validator("llm_temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if not 0.0 <= value <= 2.0:
            raise ValueError("must be between 0 and 2")

        return value

    @field_validator("llm_top_p")
    @classmethod
    def _validate_top_p(cls, value: float | None) -> float | None:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError("must be greater than 0 and not greater than 1")

        return value

    @field_validator("llm_base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        normalized = value.strip()
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")

        return normalized.rstrip("/")

    @model_validator(mode="after")
    def _validate_chunk_relationship(self) -> "Settings":
        """Проверяет данные до дальнейшей обработки. Это нужно, чтобы ошибка проявилась рано и не испортила результат следующих роботов."""
        if self.knowledge_chunk_overlap >= self.knowledge_chunk_size:
            raise ValueError("knowledge_chunk_overlap must be smaller than knowledge_chunk_size")

        return self


class Config:
    """Класс «Config» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    @staticmethod
    def load() -> Settings:
        """Выполняет шаг «load». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        env_file = Config._env_file()
        load_dotenv(dotenv_path=env_file, override=False)

        try:
            return Settings(_env_file=env_file)
        except ValidationError as exc:
            missing_fields = [
                str(error["loc"][0])
                for error in exc.errors()
                if error["type"] == "missing"
            ]

            if missing_fields:
                variables = ", ".join(missing_fields)
                raise RuntimeError(
                    f"Missing required environment variables: {variables}"
                ) from exc

            raise RuntimeError(f"Invalid application configuration: {exc}") from exc

    @staticmethod
    def _env_file() -> Path:
        """Выполняет шаг «env file». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return Path(__file__).resolve().parents[2] / ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает уже подготовленный объект или настройку, чтобы остальные части проекта использовали единый источник."""
    return Config.load()
