from dataclasses import dataclass, field
from typing import Literal

from app.security.injection_detector import InjectionDetector, InjectionResult
from app.security.pii_sanitizer import PIISanitizer

SecurityStatus = Literal["safe", "warning", "blocked"]


@dataclass(frozen=True)
class SecurityPipelineResult:
    """Класс «SecurityPipelineResult» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    status: SecurityStatus
    safe: bool
    sanitized_text: str
    restoration_map: dict[str, str]
    injection_result: InjectionResult
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class SecurityPipeline:
    """Класс «SecurityPipeline» хранит связанную логику проекта. Он нужен, чтобы сгруппировать данные и действия в понятный блок."""

    def __init__(
        self,
        pii_sanitizer: PIISanitizer | None = None,
        injection_detector: InjectionDetector | None = None,
    ) -> None:
        """Подготавливает объект к работе: принимает зависимости, настройки и шаблоны, чтобы при запуске этап знал, чем пользоваться."""
        self._pii_sanitizer = pii_sanitizer or PIISanitizer()
        self._injection_detector = injection_detector or InjectionDetector()

    def process(self, text: str) -> SecurityPipelineResult:
        """[ЗАПУСК РОБОТА] Главная команда этапа: она заставляет этого робота выполнить свою работу и вернуть результат в формате, который понимает следующий участок конвейера."""
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
