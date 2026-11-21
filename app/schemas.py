"""Pydantic models exchanged by the API (they also drive the Swagger documentation)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Body of ``POST /ask``."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["What is the difference between a Deployment and a StatefulSet?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of retrieved chunks (defaults to TOP_K).",
    )


class Source(BaseModel):
    """A retrieved chunk that was given to the LLM as context."""

    file: str = Field(..., description="Path of the source file, relative to the corpus root")
    title: str | None = Field(default=None, description="Title of the documentation page")
    excerpt: str = Field(..., description="Beginning of the retrieved chunk")
    score: float = Field(..., description="FAISS L2 distance (lower = more similar)")


class AskResponse(BaseModel):
    """Answer returned by ``POST /ask``."""

    question: str
    answer: str
    sources: list[Source]
    latency_ms: float = Field(..., description="End-to-end processing time in milliseconds")
    model: str = Field(..., description="LLM used to generate the answer")


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: str = Field(..., examples=["ok"])
    index_loaded: bool
    llm_model: str
    embedding_model: str
