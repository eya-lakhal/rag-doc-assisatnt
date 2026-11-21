"""Download a subset of the Kubernetes documentation (CC BY 4.0) into ``data/raw/``.

Source: https://github.com/kubernetes/website, folder ``content/en/docs/concepts``.

Only one GitHub API call is made (the recursive git tree of the branch), then each
Markdown file is fetched from ``raw.githubusercontent.com``, which is not rate limited
like the API. A ``manifest.json`` records the commit SHA so the corpus is reproducible.

Usage::

    python -m ingest.download_docs --max-files 80
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

REPO = "kubernetes/website"
DOCS_ROOT = "content/en/docs/concepts/"

# Sub-folders kept in the corpus, in priority order. They cover the core concepts
# a developer asks about (pods, workloads, services, config, storage, ...).
DEFAULT_SECTIONS: tuple[str, ...] = (
    "overview/",
    "workloads/",
    "services-networking/",
    "configuration/",
    "storage/",
    "architecture/",
    "scheduling-eviction/",
    "security/",
)

TIMEOUT_S = 30


def list_markdown_files(ref: str, sections: tuple[str, ...], max_files: int) -> tuple[str, list[str]]:
    """Return ``(commit_sha, paths)`` of the Markdown files to download.

    Files are selected section by section (in ``sections`` order) and sorted
    alphabetically inside a section, so the selection is deterministic for a given SHA.
    ``_index.md`` files (section landing pages, mostly navigation) are skipped.
    """
    url = f"https://api.github.com/repos/{REPO}/git/trees/{ref}?recursive=1"
    response = requests.get(url, timeout=TIMEOUT_S, headers={"Accept": "application/vnd.github+json"})
    response.raise_for_status()
    payload = response.json()

    all_paths = sorted(
        item["path"]
        for item in payload["tree"]
        if item["type"] == "blob"
        and item["path"].startswith(DOCS_ROOT)
        and item["path"].endswith(".md")
        and not item["path"].endswith("_index.md")
    )

    selected: list[str] = []
    for section in sections:
        prefix = DOCS_ROOT + section
        selected.extend(p for p in all_paths if p.startswith(prefix))
    return payload["sha"], selected[:max_files]


def download_file(sha: str, path: str, raw_dir: Path) -> Path:
    """Download one file at commit ``sha`` and store it under ``raw_dir``."""
    url = f"https://raw.githubusercontent.com/{REPO}/{sha}/{path}"
    response = requests.get(url, timeout=TIMEOUT_S)
    response.raise_for_status()

    target = raw_dir / path.removeprefix(DOCS_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(response.text, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="Branch, tag or commit of kubernetes/website")
    parser.add_argument("--max-files", type=int, default=80, help="Number of Markdown files (50-100)")
    parser.add_argument("--out", type=Path, default=settings.raw_dir, help="Output directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sha, paths = list_markdown_files(args.ref, DEFAULT_SECTIONS, args.max_files)
    logger.info("Commit %s: %d files selected", sha[:10], len(paths))

    args.out.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths, start=1):
        download_file(sha, path, args.out)
        logger.info("[%d/%d] %s", i, len(paths), path.removeprefix(DOCS_ROOT))

    manifest = {
        "source": f"https://github.com/{REPO}",
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "commit": sha,
        "files": [p.removeprefix(DOCS_ROOT) for p in paths],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Done: corpus written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
