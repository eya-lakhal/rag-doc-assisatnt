"""Shared fixtures: a tiny in-memory FAISS index with deterministic fake embeddings.

No Ollama server is needed to run the test-suite (important for CI).
"""

from __future__ import annotations

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag import RAGPipeline

CORPUS = [
    Document(
        page_content="A Pod is the smallest deployable unit of computing in Kubernetes.",
        metadata={"source": "workloads/pods/_pod.md", "title": "Pods"},
    ),
    Document(
        page_content="A Service exposes a set of Pods as a network service with a stable IP.",
        metadata={"source": "services-networking/service.md", "title": "Service"},
    ),
    Document(
        page_content="A ConfigMap stores non-confidential configuration data as key-value pairs.",
        metadata={"source": "configuration/configmap.md", "title": "ConfigMaps"},
    ),
]


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def vectorstore(embeddings: DeterministicFakeEmbedding) -> FAISS:
    return FAISS.from_documents(CORPUS, embeddings)


@pytest.fixture
def pipeline(vectorstore: FAISS) -> RAGPipeline:
    llm = FakeListChatModel(responses=["A Pod is the smallest deployable unit [1]."])
    return RAGPipeline(vectorstore=vectorstore, llm=llm, top_k=2)
