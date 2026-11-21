"""FastAPI application exposing the RAG assistant.

Endpoints:
- ``GET  /health`` - liveness + whether the FAISS index is loaded;
- ``POST /ask``    - question -> grounded answer + sources + latency.

Interactive documentation (Swagger UI) is served at ``/docs``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.rag import RAGPipeline
from app.schemas import AskRequest, AskResponse, HealthResponse, Source

logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the FAISS index once at startup (not on every request)."""
    settings = get_settings()
    try:
        app.state.pipeline = RAGPipeline.from_settings(settings)
        logger.info("FAISS index loaded from %s", settings.index_dir)
    except FileNotFoundError as exc:
        # The API still starts so /health can report the problem.
        app.state.pipeline = None
        logger.warning("%s", exc)
    yield


app = FastAPI(
    title="RAG Doc Assistant",
    description="Question answering over the Kubernetes documentation with FAISS, LangChain and Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> RAGPipeline | None:
    """Dependency returning the pipeline (overridden in tests)."""
    return getattr(request.app.state, "pipeline", None)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health(
    pipeline: RAGPipeline | None = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Report whether the service is ready to answer questions."""
    return HealthResponse(
        status="ok" if pipeline is not None else "degraded",
        index_loaded=pipeline is not None,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


@app.post("/ask", response_model=AskResponse, tags=["rag"])
async def ask(
    body: AskRequest,
    pipeline: RAGPipeline | None = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> AskResponse:
    """Answer a question using only the indexed documentation."""
    if pipeline is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "FAISS index not loaded, run the ingestion first.")

    start = time.perf_counter()
    try:
        # The LangChain call is blocking: run it in a thread to keep the event loop free.
        result = await run_in_threadpool(pipeline.ask, body.question, body.top_k)
    except Exception as exc:  # noqa: BLE001 - surface LLM/Ollama failures as 502
        logger.exception("RAG pipeline failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM backend error: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info("ask latency=%.0fms chunks=%d", latency_ms, len(result.chunks))
    return AskResponse(
        question=body.question,
        answer=result.answer,
        sources=[
            Source(file=c.source, title=c.title, excerpt=c.excerpt, score=round(c.score, 4)) for c in result.chunks
        ],
        latency_ms=round(latency_ms, 1),
        model=settings.llm_model,
    )
