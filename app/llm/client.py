from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

Message = Mapping[str, str]

# Наблюдатель за ответами провайдера. Клиент зовет его после каждого успешного
# запроса и передает метаданные ответа, включая usage с числом токенов. Нужен
# замерам: сам вызов возвращает только полезную нагрузку, а расход токенов иначе
# нигде не сохраняется.
UsageObserver = Callable[[Mapping[str, Any]], None]


class LLMProviderError(RuntimeError):
    """Normalized provider failure that lets the runner make retry decisions."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable


class LLMStructuredOutputError(RuntimeError):
    """Normalized failure for model responses that are not usable structured output."""

    def __init__(
        self,
        message: str,
        *,
        error_kind: str,
        provider_metadata: dict[str, Any] | None = None,
        content_length: int | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.provider_metadata = provider_metadata or {}
        self.content_length = content_length
        self.retryable = retryable


class LLMClient(ABC):
    """Класс «LLMClient» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> str:
        """Выполняет шаг «generate». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Выполняет шаг «generate json». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Выполняет шаг «stream». Документация описывает назначение метода, а сама логика остается в коде ниже."""
        raise NotImplementedError
