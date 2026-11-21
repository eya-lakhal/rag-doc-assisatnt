"""Tests of the FastAPI endpoints (the pipeline is replaced by the fake one)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline
from app.rag import RAGPipeline


@pytest.fixture
def client(pipeline: RAGPipeline) -> Iterator[TestClient]:
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_index() -> Iterator[TestClient]:
    app.dependency_overrides[get_pipeline] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True


def test_health_degraded_without_index(client_without_index: TestClient) -> None:
    body = client_without_index.get("/health").json()
    assert body == {**body, "status": "degraded", "index_loaded": False}


def test_ask_returns_answer_sources_and_latency(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is a Pod?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A Pod is the smallest deployable unit [1]."
    assert len(body["sources"]) == 2
    assert {"file", "title", "excerpt", "score"} <= body["sources"][0].keys()
    assert body["latency_ms"] >= 0


def test_ask_top_k_override(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "What is a Pod?", "top_k": 1}).json()
    assert len(body["sources"]) == 1


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "ok?", "top_k": 0}])
def test_ask_validation_error(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_without_index_returns_503(client_without_index: TestClient) -> None:
    assert client_without_index.post("/ask", json={"question": "What is a Pod?"}).status_code == 503


def test_ask_llm_failure_returns_502(
    client: TestClient, pipeline: RAGPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(question: str, top_k: int | None = None) -> None:
        raise ConnectionError("ollama unreachable")

    monkeypatch.setattr(pipeline, "ask", boom)
    assert client.post("/ask", json={"question": "What is a Pod?"}).status_code == 502


def test_openapi_docs_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert "/ask" in client.get("/openapi.json").json()["paths"]
