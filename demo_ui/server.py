"""FastAPI web layer for the Demo UI analysis endpoint."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictStr

from app.config import Config
from app.input import BriefInputError, BriefInputFactory
from app.llm.factory import LLMClientFactory
from app.llm.runner import (
    LLMRunnerProviderError,
    LLMRunnerStructuredOutputError,
    LLMRunnerTimeoutError,
)
from app.pipeline import BriefAnalysisPipeline, BriefAnalysisPipelineError
from demo_ui.dto import build_demo_response

logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    brief: StrictStr


app = FastAPI(title="AI Brief Analyst Demo API")


@lru_cache(maxsize=1)
def get_pipeline() -> BriefAnalysisPipeline:
    """Create the production pipeline once for the demo server process."""
    settings = Config.load()
    llm_client = LLMClientFactory.create(settings)
    return BriefAnalysisPipeline.from_llm_client(
        llm_client,
        settings=settings,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error_response(
        status_code=422,
        error_type="request_validation",
        message="Некорректный JSON-запрос. Передайте поле brief строкой.",
        retryable=False,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=None)
def analyze(
    request: AnalyzeRequest,
    pipeline: BriefAnalysisPipeline = Depends(get_pipeline),
):
    try:
        brief_input = BriefInputFactory().from_text(request.brief)
        context = pipeline.run_context(brief_input)
        return build_demo_response(context)
    except BriefInputError:
        logger.info("Invalid analyze request brief input")
        return _error_response(
            status_code=400,
            error_type="invalid_request",
            message="Некорректный текст брифа.",
            retryable=False,
        )
    except (BriefAnalysisPipelineError, RuntimeError) as exc:
        logger.exception("Analyze request failed")
        return _pipeline_error_response(exc)
    except Exception:
        logger.exception("Unexpected analyze request failure")
        return _error_response(
            status_code=500,
            error_type="internal_error",
            message="Внутренняя ошибка сервера.",
            retryable=False,
        )


def _pipeline_error_response(exc: Exception) -> JSONResponse:
    timeout_error = _find_exception(exc, LLMRunnerTimeoutError)
    if timeout_error is not None:
        return _error_response(
            status_code=503,
            error_type="llm_timeout",
            message="Провайдер LLM не ответил вовремя. Попробуйте позже.",
            retryable=True,
        )

    provider_error = _find_exception(exc, LLMRunnerProviderError)
    if provider_error is not None:
        retryable = bool(getattr(provider_error, "retryable", True))
        return _error_response(
            status_code=503 if retryable else 502,
            error_type="llm_provider_error",
            message=_provider_error_message(provider_error),
            retryable=retryable,
        )

    structured_error = _find_exception(exc, LLMRunnerStructuredOutputError)
    if structured_error is not None:
        return _error_response(
            status_code=502,
            error_type="llm_structured_output_error",
            message=_structured_output_error_message(structured_error),
            retryable=bool(getattr(structured_error, "retryable", True)),
        )

    return _error_response(
        status_code=500,
        error_type="internal_error",
        message="Технический сбой анализа брифа.",
        retryable=False,
    )


def _provider_error_message(exc: BaseException) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 403}:
        return "Ошибка доступа к провайдеру LLM."
    if status_code == 402:
        return "Недостаточно доступа или кредитов у провайдера LLM."
    if status_code == 429:
        return "Провайдер LLM временно ограничил частоту запросов. Попробуйте позже."
    if status_code in {500, 502, 503, 504}:
        return "Провайдер LLM временно недоступен. Попробуйте позже."
    if getattr(exc, "retryable", True):
        return "Провайдер LLM временно недоступен. Попробуйте позже."
    return "Запрос к провайдеру LLM завершился ошибкой."


def _structured_output_error_message(exc: BaseException) -> str:
    error_kind = getattr(exc, "error_kind", None)
    if error_kind == "empty_content":
        return "Модель вернула пустой ответ. Попробуйте повторить запрос."
    if error_kind == "length_finish":
        return "Ответ модели был обрезан из-за лимита генерации."
    if error_kind == "truncated_json":
        return "Ответ модели был обрезан и не может быть разобран."
    return "Модель вернула некорректный структурированный ответ."


def _error_response(
    *,
    status_code: int,
    error_type: str,
    message: str,
    retryable: bool,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": {
                "type": error_type,
                "message": message,
                "retryable": retryable,
            },
        },
    )


def _find_exception(
    exc: BaseException,
    target_type: type[BaseException],
) -> BaseException | None:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, target_type):
            return current
        current = current.__cause__ or current.__context__
    return None
