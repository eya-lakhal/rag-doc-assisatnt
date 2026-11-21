"""RAG chain: FAISS retriever + grounded prompt + Ollama LLM (LangChain LCEL).

The chain is deliberately explicit rather than a black-box ``RetrievalQA``:

1. ``retrieve``  - top-k similarity search in FAISS (with scores, returned as sources);
2. ``format``    - numbered context blocks ``[1] (file) ...``;
3. ``generate``  - ``prompt | llm | StrOutputParser`` with a strict instruction to answer
   only from the context and to say "I don't know" otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import Settings

NO_ANSWER = "I don't know: the documentation I have access to does not cover this question."

SYSTEM_PROMPT = f"""You are an assistant that answers questions about the Kubernetes documentation.

Rules:
- Answer ONLY with information found in the CONTEXT below. Never use prior knowledge.
- If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise (at most 5 sentences) and technically precise.
- Cite the context blocks you used with their number, e.g. [1] or [2][3].
- Answer in the same language as the question."""

HUMAN_PROMPT = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])

EXCERPT_CHARS = 300


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the retriever, with its FAISS distance."""

    document: Document
    score: float

    @property
    def source(self) -> str:
        return str(self.document.metadata.get("source", "unknown"))

    @property
    def title(self) -> str | None:
        return self.document.metadata.get("title")

    @property
    def excerpt(self) -> str:
        text = " ".join(self.document.page_content.split())
        return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS].rstrip() + "..."


@dataclass(frozen=True)
class RAGResult:
    """Answer and the chunks that were used to produce it."""

    answer: str
    chunks: list[RetrievedChunk]


def build_embeddings(settings: Settings) -> Embeddings:
    """Embedding model shared by indexing and querying (they MUST be identical)."""
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)


def build_llm(settings: Settings) -> BaseLanguageModel:
    """Local chat model served by Ollama."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,  # 4096 tokens fit top_k=4 chunks of 1000 chars
        num_thread=settings.llm_num_thread,
        keep_alive="1h",  # avoid reloading the model (slow on CPU) between requests
    )


def load_vectorstore(index_dir: Path, embeddings: Embeddings) -> FAISS:
    """Load a FAISS index written by ``ingest/build_index.py``.

    ``allow_dangerous_deserialization`` is required because the docstore is pickled;
    it is safe here since we only ever load an index we built ourselves.
    """
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index in {index_dir}. Run `python -m ingest.build_index`.")
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered blocks the LLM can cite."""
    return "\n\n".join(
        f"[{i}] (source: {chunk.source})\n{chunk.document.page_content}" for i, chunk in enumerate(chunks, 1)
    )


class RAGPipeline:
    """Question answering over a FAISS index with an LLM."""

    def __init__(self, vectorstore: FAISS, llm: BaseLanguageModel, top_k: int = 4) -> None:
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.chain = PROMPT | llm | StrOutputParser()

    @classmethod
    def from_settings(cls, settings: Settings, index_dir: Path | None = None) -> RAGPipeline:
        """Build the production pipeline (Ollama embeddings + Ollama LLM)."""
        vectorstore = load_vectorstore(index_dir or settings.index_dir, build_embeddings(settings))
        return cls(vectorstore=vectorstore, llm=build_llm(settings), top_k=settings.top_k)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks (lowest L2 distance first)."""
        results = self.vectorstore.similarity_search_with_score(question, k=top_k or self.top_k)
        return [RetrievedChunk(document=doc, score=float(score)) for doc, score in results]

    def ask(self, question: str, top_k: int | None = None) -> RAGResult:
        """Retrieve context then generate a grounded answer."""
        chunks = self.retrieve(question, top_k)
        if not chunks:
            return RAGResult(answer=NO_ANSWER, chunks=[])
        answer = self.chain.invoke({"context": format_context(chunks), "question": question})
        return RAGResult(answer=answer.strip(), chunks=chunks)
