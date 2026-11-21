"""Tests of the corpus cleaning and chunking step."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingest.build_index import chunk_stats, clean_markdown, load_documents, split_documents

SAMPLE_PAGE = """---
title: "Pods"
content_type: concept
weight: 10
---

<!-- overview -->

_Pods_ are the smallest deployable units. A Pod runs on a
{{< glossary_tooltip text="node" term_id="node" >}}.

{{< note >}}
Pods are ephemeral.
{{< /note >}}



## Using Pods

Usually you don't create Pods directly.
"""


def test_clean_markdown_extracts_title_and_removes_hugo_syntax() -> None:
    text, title = clean_markdown(SAMPLE_PAGE)

    assert title == "Pods"
    assert text.startswith("# Pods")
    assert "content_type" not in text  # front matter removed
    assert "<!--" not in text  # HTML comments removed
    assert "{{" not in text  # shortcodes removed
    assert "runs on a\nnode." in text  # glossary tooltip replaced by its text
    assert "Pods are ephemeral." in text  # content inside a shortcode is kept
    assert "\n\n\n" not in text


def test_clean_markdown_without_front_matter() -> None:
    text, title = clean_markdown("Just a paragraph.")
    assert title is None
    assert text == "Just a paragraph."


def test_load_documents_reads_markdown_recursively(tmp_path: Path) -> None:
    (tmp_path / "workloads").mkdir()
    (tmp_path / "workloads" / "pods.md").write_text(SAMPLE_PAGE, encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")  # must be ignored

    docs = load_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata == {"source": "workloads/pods.md", "title": "Pods"}


@pytest.mark.parametrize(("chunk_size", "overlap"), [(500, 75), (1000, 150)])
def test_split_respects_chunk_size_and_keeps_metadata(chunk_size: int, overlap: int) -> None:
    paragraphs = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(40))
    doc = Document(page_content=paragraphs, metadata={"source": "a.md", "title": "A"})

    chunks = split_documents([doc], chunk_size, overlap)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= chunk_size for c in chunks)
    assert all(c.metadata["source"] == "a.md" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_smaller_chunks_produce_more_chunks() -> None:
    doc = Document(page_content="\n\n".join("sentence " * 50 for _ in range(30)), metadata={"source": "a.md"})
    small = split_documents([doc], 500, 50)
    large = split_documents([doc], 1000, 50)
    assert chunk_stats(small)["n_chunks"] > chunk_stats(large)["n_chunks"]


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        split_documents([Document(page_content="x")], chunk_size=100, chunk_overlap=100)
