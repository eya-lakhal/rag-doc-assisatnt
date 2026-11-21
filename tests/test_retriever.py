"""Tests of the FAISS retriever and of the RAG chain (fake LLM, no Ollama)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.rag import NO_ANSWER, PROMPT, RAGPipeline, RetrievedChunk, format_context, load_vectorstore
from tests.conftest import CORPUS


def test_retrieve_returns_top_k_sorted_by_distance(pipeline: RAGPipeline) -> None:
    chunks = pipeline.retrieve(CORPUS[1].page_content, top_k=3)

    assert len(chunks) == 3
    # Identical text => identical fake embedding => distance 0 => ranked first.
    assert chunks[0].source == "services-networking/service.md"
    assert chunks[0].score == pytest.approx(0.0, abs=1e-5)
    assert [c.score for c in chunks] == sorted(c.score for c in chunks)


def test_retrieve_uses_default_top_k(pipeline: RAGPipeline) -> None:
    assert len(pipeline.retrieve("What is a Pod?")) == 2


def test_ask_returns_answer_and_sources(pipeline: RAGPipeline) -> None:
    result = pipeline.ask("What is a Pod?")

    assert result.answer == "A Pod is the smallest deployable unit [1]."
    assert len(result.chunks) == 2
    assert all(isinstance(c, RetrievedChunk) for c in result.chunks)


def test_ask_without_context_says_i_dont_know(pipeline: RAGPipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k=None: [])
    assert pipeline.ask("Anything?").answer == NO_ANSWER


def test_prompt_enforces_grounding() -> None:
    system = PROMPT.messages[0].prompt.template
    assert "ONLY" in system
    assert NO_ANSWER in system


def test_format_context_numbers_blocks(pipeline: RAGPipeline) -> None:
    context = format_context(pipeline.retrieve("Pod", top_k=2))
    assert context.startswith("[1] (source: ")
    assert "\n\n[2] (source: " in context


def test_excerpt_is_truncated() -> None:
    from langchain_core.documents import Document

    chunk = RetrievedChunk(document=Document(page_content="a " * 1000, metadata={}), score=0.1)
    assert len(chunk.excerpt) <= 303
    assert chunk.excerpt.endswith("...")
    assert chunk.source == "unknown"


def test_index_roundtrip(tmp_path: Path, vectorstore: FAISS, embeddings: DeterministicFakeEmbedding) -> None:
    vectorstore.save_local(str(tmp_path))
    loaded = load_vectorstore(tmp_path, embeddings)
    assert loaded.index.ntotal == len(CORPUS)


def test_missing_index_raises(tmp_path: Path, embeddings: DeterministicFakeEmbedding) -> None:
    with pytest.raises(FileNotFoundError):
        load_vectorstore(tmp_path, embeddings)
