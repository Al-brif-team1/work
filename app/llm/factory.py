from app.config import Settings
from app.llm.client import LLMClient
from app.llm.openrouter import OpenRouterLLMClient


class LLMClientFactory:
    """Factory for creating the configured LLM provider client."""

    @staticmethod
    def create(settings: Settings) -> LLMClient:
        """Create the current LLM client implementation."""
        return OpenRouterLLMClient(settings=settings)
