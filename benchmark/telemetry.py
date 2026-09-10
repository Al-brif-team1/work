"""Collect per-brief LLM usage while a benchmark run is in progress."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UsageSnapshot:
    """LLM usage accumulated for a single benchmark row."""

    llm_calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class UsageCollector:
    """Accumulate token usage reported by the LLM client for one brief.

    Плагинится в клиента как usage_observer и считает каждый запрос к провайдеру,
    включая неудачные попытки, которые потом повторяются: иначе расход токенов
    на бриф окажется занижен ровно на цену ретраев.
    """

    def __init__(self) -> None:
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._usage_reported = False

    def __call__(self, metadata: Mapping[str, Any]) -> None:
        """Record one provider response."""
        self._llm_calls += 1

        usage = metadata.get("usage")
        if not isinstance(usage, Mapping):
            return

        prompt_tokens = _optional_int(usage.get("prompt_tokens"))
        completion_tokens = _optional_int(usage.get("completion_tokens"))
        total_tokens = _optional_int(usage.get("total_tokens"))
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return

        self._usage_reported = True
        self._prompt_tokens += prompt_tokens or 0
        self._completion_tokens += completion_tokens or 0
        # Не каждый провайдер присылает total_tokens, поэтому при его отсутствии
        # складываем сами, а не теряем строку целиком.
        if total_tokens is None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        self._total_tokens += total_tokens

    def reset(self) -> None:
        """Drop accumulated usage before the next brief."""
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._usage_reported = False

    def snapshot(self) -> UsageSnapshot:
        """Return usage accumulated since the last reset."""
        if not self._usage_reported:
            # Вызовы посчитаны, но провайдер не сообщил ни одной цифры: пустые
            # токены честнее нулей, которые в метриках неотличимы от «бесплатно».
            return UsageSnapshot(
                llm_calls=self._llm_calls,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )

        return UsageSnapshot(
            llm_calls=self._llm_calls,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
        )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
