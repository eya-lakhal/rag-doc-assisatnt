"""Application settings, read from environment variables (or a local ``.env`` file).

Every tunable of the pipeline lives here so that the same code can be run with
different configurations (e.g. ``CHUNK_SIZE=500`` vs ``CHUNK_SIZE=1000``) without
touching the source, which is what the Ragas evaluation relies on.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration of the RAG pipeline and the API."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Models (served locally by Ollama, 100 % free) ---------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    llm_model: str = Field(default="llama3.2:3b", description="Ollama chat model used to answer")
    embedding_model: str = Field(default="nomic-embed-text", description="Ollama embedding model")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Chunking (used by ingest/build_index.py) -------------------------------------
    chunk_size: int = Field(default=1000, gt=0, description="Max characters per chunk")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters shared by two chunks")

    # --- Retrieval --------------------------------------------------------------------
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks given to the LLM")

    # --- Paths ------------------------------------------------------------------------
    raw_dir: Path = Field(default=Path("data/raw"))
    index_dir: Path = Field(default=Path("data/index/default"))


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (env is read once per process)."""
    return Settings()
