from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

Message = Mapping[str, str]


class LLMClient(ABC):
    """Provider-independent interface for model calls."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> str:
        """Return a plain text model response."""
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a JSON object decoded from the model response."""
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Yield text chunks from a streaming model response."""
        raise NotImplementedError
