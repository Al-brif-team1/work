from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(..., alias="OPENROUTER_MODEL")
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
    )
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        """Ensure numeric knowledge settings are positive."""
        if value <= 0:
            raise ValueError("must be greater than zero")

        return value

    @model_validator(mode="after")
    def _validate_chunk_relationship(self) -> "Settings":
        """Ensure the chunk overlap stays smaller than the chunk size."""
        if self.knowledge_chunk_overlap >= self.knowledge_chunk_size:
            raise ValueError("knowledge_chunk_overlap must be smaller than knowledge_chunk_size")

        return self


class Config:
    """Loads validated application settings from the project .env file."""

    @staticmethod
    def load() -> Settings:
        """Load settings and convert validation failures to readable errors."""
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
        """Return the absolute path to the project .env file."""
        return Path(__file__).resolve().parents[2] / ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Config.load()
