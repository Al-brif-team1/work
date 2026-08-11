import re
from dataclasses import dataclass
from typing import ClassVar, Literal

RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class InjectionResult:
    """Prompt injection detection result."""

    safe: bool
    risk_level: RiskLevel
    found_patterns: list[str]


@dataclass(frozen=True)
class InjectionPattern:
    """Regex pattern and score used by the injection detector."""

    name: str
    pattern: re.Pattern[str]
    weight: int


class InjectionDetector:
    """Detects common prompt injection attempts using local regex rules."""

    _patterns: ClassVar[tuple[InjectionPattern, ...]] = (
        InjectionPattern(
            "ignore_previous_instructions",
            re.compile(
                r"\b(ignore|disregard|bypass|override)\s+(all\s+)?"
                r"(previous|prior|above|earlier)\s+instructions\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "forget_previous_instructions",
            re.compile(
                r"\bforget\s+(all\s+)?(previous|prior|above|earlier)"
                r"\s+instructions\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "system_prompt_access",
            re.compile(r"\b(system|developer)\s+prompt\b", re.IGNORECASE),
            3,
        ),
        InjectionPattern(
            "repeat_instructions",
            re.compile(
                r"\b(repeat|reveal|print|show|display)\s+(your\s+)?"
                r"(hidden\s+)?instructions\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "show_hidden_prompt",
            re.compile(
                r"\b(show|reveal|print|display)\s+(the\s+)?hidden\s+prompt\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "output_raw_prompt",
            re.compile(
                r"\b(output|print|dump|return)\s+(the\s+)?raw\s+prompt\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "act_as_role_override",
            re.compile(
                r"\b(act\s+as|pretend\s+to\s+be|roleplay\s+as)\b",
                re.IGNORECASE,
            ),
            1,
        ),
        InjectionPattern(
            "jailbreak",
            re.compile(
                r"\b(jailbreak|dan\s+mode|developer\s+mode)\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "russian_show_system_prompt",
            re.compile(
                r"\b(выведи|покажи|раскрой|напечатай)\s+"
                r"(системный|скрытый)\s+промпт\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "russian_ignore_instructions",
            re.compile(
                r"\b(игнорируй|забудь|обойди|отмени)\s+(все\s+)?"
                r"(предыдущие\s+|прошлые\s+)?инструкции\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "russian_developer_prompt",
            re.compile(
                r"\b(системный|developer|разработческий)\s+промпт\b",
                re.IGNORECASE,
            ),
            3,
        ),
        InjectionPattern(
            "policy_bypass",
            re.compile(
                r"\b(bypass|disable|ignore)\s+"
                r"(safety|policy|guardrails|filters?)\b",
                re.IGNORECASE,
            ),
            2,
        ),
    )

    def detect(self, text: str) -> InjectionResult:
        """Return prompt injection risk information for text."""
        found_patterns: list[str] = []
        score = 0

        for injection_pattern in self._patterns:
            if injection_pattern.pattern.search(text):
                found_patterns.append(injection_pattern.name)
                score += injection_pattern.weight

        risk_level = self._risk_level(score)

        return InjectionResult(
            safe=score == 0,
            risk_level=risk_level,
            found_patterns=found_patterns,
        )

    @staticmethod
    def _risk_level(score: int) -> RiskLevel:
        if score >= 3:
            return "high"

        if score > 0:
            return "medium"

        return "low"
