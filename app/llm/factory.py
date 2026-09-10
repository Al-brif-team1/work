from app.config import Settings
from app.llm.client import LLMClient, UsageObserver
from app.llm.openrouter import OpenRouterLLMClient


class LLMClientFactory:
    """Класс «LLMClientFactory» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    @staticmethod
    def create(
        settings: Settings,
        *,
        usage_observer: UsageObserver | None = None,
    ) -> LLMClient:
        """Выполняет шаг «create». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        return OpenRouterLLMClient(settings=settings, usage_observer=usage_observer)
