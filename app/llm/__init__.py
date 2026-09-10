"""Модуль инфраструктуры LLM. Он отделяет работу с ИИ-моделью от бизнес-логики, чтобы роботы конвейера получали ответы единым способом."""

from app.llm.client import LLMClient, LLMProviderError, LLMStructuredOutputError
from app.llm.factory import LLMClientFactory
from app.llm.openrouter import OpenRouterLLMClient
from app.llm.runner import (
    LLMRunResult,
    LLMRunner,
    LLMRunnerError,
    LLMRunnerProviderError,
    LLMRunnerStructuredOutputError,
    LLMRunnerTimeoutError,
    LLMTokenUsage,
)

__all__ = [
    "LLMClient",
    "LLMClientFactory",
    "LLMProviderError",
    "LLMStructuredOutputError",
    "LLMRunResult",
    "LLMRunner",
    "LLMRunnerError",
    "LLMRunnerProviderError",
    "LLMRunnerStructuredOutputError",
    "LLMRunnerTimeoutError",
    "LLMTokenUsage",
    "OpenRouterLLMClient",
]
