"""Traffic-light configuration for Masterskaya student skill fit."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config.criteria import CriteriaConfigError

TrafficLightColor = Literal["green", "yellow", "red"]


class TrafficLightSpecialization(BaseModel):
    """Task fit rules for one specialization."""

    key: str
    title: str
    green: list[str] = Field(default_factory=list)
    yellow: list[str] = Field(default_factory=list)
    red: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TrafficLightDirection(BaseModel):
    """Task fit rules grouped by direction."""

    key: str
    title: str
    specializations: list[TrafficLightSpecialization] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TrafficLightConfiguration(BaseModel):
    """Top-level traffic-light business configuration."""

    version: str
    description: str
    directions: list[TrafficLightDirection]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "TrafficLightConfiguration":
        if not self.directions:
            raise ValueError("traffic light configuration has no directions")
        return self


class TrafficLightConfig(BaseModel):
    """Root model for config/traffic_light.yaml."""

    traffic_light: TrafficLightConfiguration

    model_config = ConfigDict(extra="forbid")


class TrafficLightConfigError(CriteriaConfigError):
    """Raised when traffic-light configuration cannot be loaded."""


class TrafficLightLoader:
    """Loads traffic-light rules from YAML."""

    @staticmethod
    def load(path: Path | None = None) -> TrafficLightConfig:
        config_path = path or TrafficLightLoader.default_path()

        try:
            raw_text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TrafficLightConfigError(
                f"Unable to read traffic light configuration: {config_path}"
            ) from exc

        try:
            import yaml

            data = yaml.safe_load(raw_text)
        except Exception as exc:
            raise TrafficLightConfigError(
                f"Invalid YAML syntax in traffic light configuration: {config_path}"
            ) from exc

        try:
            return TrafficLightConfig.model_validate(data)
        except ValidationError as exc:
            raise TrafficLightConfigError(
                f"Invalid traffic light configuration schema: {exc}"
            ) from exc

    @staticmethod
    def default_path() -> Path:
        return Path(__file__).resolve().parents[2] / "config" / "traffic_light.yaml"


@lru_cache(maxsize=1)
def get_traffic_light_config() -> TrafficLightConfig:
    """Return cached traffic-light configuration."""
    return TrafficLightLoader.load()
