from dataclasses import dataclass, field
from typing import Literal

from app.security.injection_detector import InjectionDetector, InjectionResult
from app.security.pii_sanitizer import PIISanitizer

SecurityStatus = Literal["safe", "warning", "blocked"]


@dataclass(frozen=True)
class SecurityPipelineResult:
    """Result of applying PII sanitization and injection detection."""

    status: SecurityStatus
    safe: bool
    sanitized_text: str
    restoration_map: dict[str, str]
    injection_result: InjectionResult
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class SecurityPipeline:
    """Runs configured security checks before text reaches an LLM."""

    def __init__(
        self,
        pii_sanitizer: PIISanitizer | None = None,
        injection_detector: InjectionDetector | None = None,
    ) -> None:
        """Initialize the pipeline with optional custom components."""
        self._pii_sanitizer = pii_sanitizer or PIISanitizer()
        self._injection_detector = injection_detector or InjectionDetector()

    def process(self, text: str) -> SecurityPipelineResult:
        """Sanitize text and return injection risk information."""
        sanitized_text = self._pii_sanitizer.sanitize(text)
        injection_result = self._injection_detector.detect(sanitized_text)
        restoration_map = self._pii_sanitizer.mapping

        if injection_result.risk_level == "high":
            return SecurityPipelineResult(
                status="blocked",
                safe=False,
                sanitized_text=sanitized_text,
                restoration_map=restoration_map,
                injection_result=injection_result,
                error="Critical prompt injection detected",
            )

        warnings: list[str] = []
        status: SecurityStatus = "safe"
        if injection_result.risk_level == "medium":
            warnings.append("Possible prompt injection detected")
            status = "warning"

        return SecurityPipelineResult(
            status=status,
            safe=True,
            sanitized_text=sanitized_text,
            restoration_map=restoration_map,
            injection_result=injection_result,
            warnings=warnings,
        )
