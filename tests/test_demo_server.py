"""Tests for the Demo UI FastAPI server layer."""

from __future__ import annotations

import importlib.util
import unittest
from typing import Any
from unittest.mock import patch

from app.input import BriefInputFactory
from app.llm.runner import LLMRunnerProviderError
from app.pipeline import BriefAnalysisPipeline
from app.schemas import AIContext, DecisionStatus
from tests.test_demo_dto import make_base_context

_FASTAPI_TESTING_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

if _FASTAPI_TESTING_AVAILABLE:
    from fastapi.testclient import TestClient
    from demo_ui import server


class FakePipeline:
    def __init__(
        self,
        context: AIContext | None = None,
        error: Exception | None = None,
    ) -> None:
        self.context = context or make_base_context()
        self.error = error
        self.calls: list[Any] = []

    def run_context(self, brief_input: Any) -> AIContext:
        self.calls.append(brief_input)
        if self.error is not None:
            raise self.error
        return self.context


@unittest.skipUnless(
    _FASTAPI_TESTING_AVAILABLE,
    "FastAPI/httpx are not installed in the current environment",
)
class TestDemoServer(unittest.TestCase):
    def setUp(self) -> None:
        server.app.dependency_overrides.clear()
        server.get_pipeline.cache_clear()

    def tearDown(self) -> None:
        server.app.dependency_overrides.clear()
        server.get_pipeline.cache_clear()

    def test_analyze_returns_demo_dto(self) -> None:
        pipeline = FakePipeline(make_base_context())
        server.app.dependency_overrides[server.get_pipeline] = lambda: pipeline

        response = TestClient(server.app).post(
            "/api/analyze",
            json={"brief": "Нужен портал поддержки."},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["decision"]["final_status"], "ACCEPT")
        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(pipeline.calls[0].original_text, "Нужен портал поддержки.")

    def test_business_reject_is_successful_response(self) -> None:
        pipeline = FakePipeline(
            make_base_context(
                status=DecisionStatus.reject,
                matched_rule="reject_by_business_risk",
            )
        )
        server.app.dependency_overrides[server.get_pipeline] = lambda: pipeline

        response = TestClient(server.app).post(
            "/api/analyze",
            json={"brief": "Бизнес-риск."},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["decision"]["final_status"], "REJECT")
        self.assertNotIn("error", payload)

    def test_empty_brief_is_passed_to_pipeline(self) -> None:
        pipeline = FakePipeline(
            make_base_context(
                status=DecisionStatus.reject,
                matched_rule="reject_empty_or_nonsense_brief",
            )
        )
        server.app.dependency_overrides[server.get_pipeline] = lambda: pipeline

        response = TestClient(server.app).post("/api/analyze", json={"brief": ""})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(pipeline.calls[0].original_text, "")
        self.assertEqual(response.json()["decision"]["final_status"], "REJECT")

    def test_missing_brief_is_validation_error_without_pipeline_call(self) -> None:
        pipeline = FakePipeline()
        server.app.dependency_overrides[server.get_pipeline] = lambda: pipeline

        response = TestClient(server.app).post("/api/analyze", json={})

        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "request_validation")
        self.assertFalse(payload["error"]["retryable"])
        self.assertEqual(pipeline.calls, [])

    def test_pipeline_failure_returns_safe_5xx_error(self) -> None:
        provider_error = LLMRunnerProviderError(
            "provider failed with sensitive details",
            status_code=429,
            retryable=True,
        )
        pipeline = FakePipeline(error=RuntimeError("wrapper"))
        pipeline.error.__cause__ = provider_error
        server.app.dependency_overrides[server.get_pipeline] = lambda: pipeline

        response = TestClient(server.app).post(
            "/api/analyze",
            json={"brief": "Нужен портал поддержки."},
        )

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "llm_provider_error")
        self.assertTrue(payload["error"]["retryable"])
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("sensitive details", response.text)

    def test_endpoint_calls_demo_dto_adapter(self) -> None:
        context = make_base_context()
        pipeline = FakePipeline(context)
        server.app.dependency_overrides[server.get_pipeline] = lambda: pipeline

        with patch(
            "demo_ui.server.build_demo_response",
            return_value={"ok": True, "sentinel": "dto"},
        ) as build_demo_response:
            response = TestClient(server.app).post(
                "/api/analyze",
                json={"brief": "Нужен портал поддержки."},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "sentinel": "dto"})
        build_demo_response.assert_called_once_with(context)

    def test_health_does_not_initialize_pipeline(self) -> None:
        pipeline = FakePipeline()
        server.app.dependency_overrides[server.get_pipeline] = lambda: pipeline

        response = TestClient(server.app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(pipeline.calls, [])

    def test_root_serves_static_index_html(self) -> None:
        response = TestClient(server.app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("<!doctype html>", response.text)
        self.assertIn("./styles.css", response.text)
        self.assertIn("./app.js", response.text)

    def test_static_css_and_js_are_available(self) -> None:
        client = TestClient(server.app)

        css_response = client.get("/styles.css")
        js_response = client.get("/app.js")

        self.assertEqual(css_response.status_code, 200)
        self.assertIn("text/css", css_response.headers["content-type"])
        self.assertIn(":root", css_response.text)
        self.assertEqual(js_response.status_code, 200)
        self.assertIn("javascript", js_response.headers["content-type"])
        self.assertIn('window.fetch("/api/analyze"', js_response.text)

    def test_static_routing_does_not_intercept_api_routes(self) -> None:
        pipeline = FakePipeline()
        server.app.dependency_overrides[server.get_pipeline] = lambda: pipeline
        client = TestClient(server.app)

        analyze_response = client.post("/api/analyze", json={})
        health_response = client.get("/health")

        self.assertEqual(analyze_response.status_code, 422)
        self.assertEqual(analyze_response.json()["error"]["type"], "request_validation")
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(pipeline.calls, [])

    def test_static_serving_does_not_expose_sensitive_repository_files(self) -> None:
        client = TestClient(server.app)

        for path in ("/.env", "/app/config.py", "/tests/test_demo_server.py"):
            with self.subTest(path=path):
                response = client.get(path)

                self.assertEqual(response.status_code, 404)
                self.assertNotIn("OPENAI_API_KEY", response.text)
                self.assertNotIn("FakePipeline", response.text)

    def test_get_pipeline_builds_production_pipeline_once(self) -> None:
        settings = object()
        llm_client = object()
        pipeline = object()

        with (
            patch("demo_ui.server.Config.load", return_value=settings) as load,
            patch(
                "demo_ui.server.LLMClientFactory.create",
                return_value=llm_client,
            ) as create_client,
            patch(
                "demo_ui.server.BriefAnalysisPipeline.from_llm_client",
                return_value=pipeline,
            ) as create_pipeline,
        ):
            first = server.get_pipeline()
            second = server.get_pipeline()

        self.assertIs(first, pipeline)
        self.assertIs(second, pipeline)
        load.assert_called_once_with()
        create_client.assert_called_once_with(settings)
        create_pipeline.assert_called_once_with(llm_client, settings=settings)


if __name__ == "__main__":
    unittest.main()
