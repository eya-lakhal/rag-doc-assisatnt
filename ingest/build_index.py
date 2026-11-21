"""Load the Markdown corpus, split it into chunks, embed them and save a FAISS index.

Pipeline::

    data/raw/**/*.md --clean--> Documents --RecursiveCharacterTextSplitter--> chunks
                     --OllamaEmbeddings (nomic-embed-text)--> FAISS --> data/index/<name>/

Usage::

    # Compare chunking strategies without computing embeddings (fast):
    python -m ingest.build_index --stats-only --chunk-size 500 --chunk-overlap 75
    python -m ingest.build_index --stats-only --chunk-size 1000 --chunk-overlap 150

    # Build the index used by the API:
    python -m ingest.build_index --chunk-size 1000 --chunk-overlap 150 --out data/index/default
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag import build_embeddings

logger = logging.getLogger(__name__)

# Separators tried in order: split on Markdown headings first, then paragraphs,
# lines, sentences and words, so chunks stay as semantically coherent as possible.
MARKDOWN_SEPARATORS: list[str] = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TITLE_RE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Hugo shortcodes carrying a visible text, e.g. {{< glossary_tooltip text="Pod" term_id="pod" >}}
_SHORTCODE_TEXT_RE = re.compile(r"\{\{[<%]\s*glossary_tooltip[^}]*?text=\"([^\"]+)\"[^}]*?[>%]\}\}")
# Any other shortcode tag ({{< note >}}, {{< /note >}}, {{% code_sample ... %}}) is dropped.
_SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_markdown(text: str) -> tuple[str, str | None]:
    """Strip Hugo-specific syntax from a Kubernetes doc page.

    Returns ``(clean_text, title)``; ``title`` comes from the YAML front matter.
    """
    title: str | None = None
    match = _FRONT_MATTER_RE.match(text)
    if match:
        title_match = _TITLE_RE.search(match.group(1))
        title = title_match.group(1).strip() if title_match else None
        text = text[match.end() :]

    text = _HTML_COMMENT_RE.sub("", text)
    text = _SHORTCODE_TEXT_RE.sub(r"\1", text)
    text = _SHORTCODE_RE.sub("", text)
    text = _BLANK_LINES_RE.sub("\n\n", text).strip()
    if title:
        text = f"# {title}\n\n{text}"
    return text, title


def load_documents(raw_dir: Path) -> list[Document]:
    """Load and clean every ``*.md`` file under ``raw_dir``."""
    documents: list[Document] = []
    for path in sorted(raw_dir.rglob("*.md")):
        content, title = clean_markdown(path.read_text(encoding="utf-8"))
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": path.relative_to(raw_dir).as_posix(), "title": title},
            )
        )
    return documents


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents into overlapping chunks (metadata is copied to every chunk)."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=MARKDOWN_SEPARATORS,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def chunk_stats(chunks: list[Document]) -> dict[str, float]:
    """Simple statistics used to compare chunking configurations."""
    lengths = [len(c.page_content) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "mean_chars": round(statistics.mean(lengths), 1) if lengths else 0.0,
        "median_chars": float(statistics.median(lengths)) if lengths else 0.0,
        "max_chars": max(lengths, default=0),
    }


def build_index(chunks: list[Document], embeddings: Embeddings, out_dir: Path) -> FAISS:
    """Embed ``chunks`` and persist the FAISS index to ``out_dir``."""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_dir))
    return vectorstore


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--stats-only", action="store_true", help="Only print chunk statistics")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    documents = load_documents(args.raw_dir)
    if not documents:
        logger.error("No Markdown file in %s. Run `python -m ingest.download_docs` first.", args.raw_dir)
        return 1

    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    stats = chunk_stats(chunks)
    logger.info("%d documents -> %s", len(documents), stats)
    if args.stats_only:
        return 0

    start = time.perf_counter()
    build_index(chunks, build_embeddings(settings), args.out)
    elapsed = time.perf_counter() - start

    meta = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "n_documents": len(documents),
        **stats,
        "build_seconds": round(elapsed, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("FAISS index saved to %s in %.1fs", args.out, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
