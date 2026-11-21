"""Evaluate several RAG configurations with Ragas, using a local Ollama model as judge.

For each configuration (index + top_k):
1. run the RAG pipeline on every question of ``eval/testset.json``;
2. score the answers with 4 Ragas metrics:
   - faithfulness       : is the answer supported by the retrieved context? (hallucination)
   - answer_relevancy   : does the answer address the question?
   - context_precision  : are the relevant chunks ranked first?
   - context_recall     : does the context contain what the reference answer needs?
3. write per-question scores to ``eval/scores_<config>.json`` and a summary table
   to ``eval/results.md``.

Usage (indexes must be built first, see README)::

    python -m eval.run_ragas
    python -m eval.run_ragas --limit 5          # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from app.config import get_settings
from app.rag import RAGPipeline

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
METRICS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "llm_context_precision_with_reference": "Context precision",
    "context_recall": "Context recall",
}


@dataclass(frozen=True)
class EvalConfig:
    """One RAG configuration to compare."""

    name: str
    index_dir: Path
    top_k: int


CONFIGS: list[EvalConfig] = [
    # Same top_k, only the chunking changes: isolates the effect of chunk_size.
    EvalConfig("chunk500_k3", Path("data/index/chunk500"), top_k=3),
    EvalConfig("chunk1000_k3", Path("data/index/chunk1000"), top_k=3),
]


def load_testset(path: Path) -> list[dict[str, str]]:
    """Load the hand-written ``[{question, reference}, ...]`` test set."""
    items = json.loads(path.read_text(encoding="utf-8"))
    return [{"question": it["question"], "reference": it["reference"]} for it in items]


def generate_samples(pipeline: RAGPipeline, testset: list[dict[str, str]], top_k: int) -> tuple[list[dict], float]:
    """Run the pipeline on every question; return Ragas samples and mean latency (s)."""
    samples: list[dict] = []
    latencies: list[float] = []
    for i, item in enumerate(testset, 1):
        start = time.perf_counter()
        result = pipeline.ask(item["question"], top_k=top_k)
        latencies.append(time.perf_counter() - start)
        samples.append(
            {
                "user_input": item["question"],
                "response": result.answer,
                "retrieved_contexts": [c.document.page_content for c in result.chunks],
                "reference": item["reference"],
            }
        )
        logger.info("[%d/%d] %.1fs  %s", i, len(testset), latencies[-1], item["question"])
    return samples, sum(latencies) / len(latencies)


def valid_scores(values: list[float | None]) -> list[float]:
    """Keep valid scores only (the small judge sometimes times out or returns bad JSON -> NaN)."""
    return [v for v in values if isinstance(v, int | float) and not math.isnan(v)]


def summarize(rows: list[dict], latency_s: float) -> dict[str, float]:
    """Mean of each metric over valid scores, plus the number of valid scores (``<metric>_n``)."""
    summary: dict[str, float] = {"n_questions": len(rows), "latency_s": latency_s}
    for metric in METRICS:
        valid = valid_scores([r.get(metric) for r in rows])
        summary[metric] = sum(valid) / len(valid) if valid else float("nan")
        summary[f"{metric}_n"] = len(valid)
    return summary


def format_cell(mean: float, n_valid: int, n_total: int) -> str:
    """``0.783 (9/10)``: the mean is meaningless without the number of valid scores."""
    value = "n/a" if math.isnan(mean) else f"{mean:.3f}"
    return f"{value} ({n_valid}/{n_total})"


def score(
    samples: list[dict], judge_model: str, base_url: str, embedding_model: str, num_thread: int | None = None
) -> list[dict]:
    """Score samples with Ragas; returns one dict of metric -> value per sample."""
    judge = LangchainLLMWrapper(
        ChatOllama(
            model=judge_model, base_url=base_url, temperature=0, num_ctx=4096, num_thread=num_thread, keep_alive="1h"
        )
    )
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=embedding_model, base_url=base_url))
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),  # 1 generated question instead of 3: 3x cheaper on CPU
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        # Local CPU inference is slow: few parallel workers, generous timeout.
        run_config=RunConfig(timeout=900, max_workers=1, max_retries=2),
        show_progress=True,
    )
    return result.to_pandas().to_dict(orient="records")


def write_report(summary: dict[str, dict[str, float]], n_questions: int, judge_model: str, path: Path) -> None:
    """Write the Markdown comparison table (comments are added by hand below the table)."""
    header = "| Configuration | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " | Mean latency (s) |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    rows = [
        f"| `{name}` | "
        + " | ".join(format_cell(vals[m], int(vals[f"{m}_n"]), int(vals["n_questions"])) for m in METRICS)
        + f" | {vals['latency_s']:.1f} |"
        for name, vals in summary.items()
    ]
    marker = "<!-- RESULTS:END -->"
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    analysis = previous.split(marker, 1)[1] if marker in previous else "\n\n## Analysis\n\n_To be written._\n"
    content = "\n".join(
        [
            "# Ragas evaluation results",
            "",
            f"- Date: {date.today().isoformat()}",
            f"- Test set: {n_questions} hand-written questions (`eval/testset.json`)",
            f"- Judge LLM: `{judge_model}` (Ollama, local), embeddings: `nomic-embed-text`",
            "- Scores are in [0, 1], higher is better. Each cell is the mean over the questions where the judge",
            "  returned a valid score, followed by `(valid / total)`; `n/a` = no valid score.",
            "",
            header,
            sep,
            *rows,
            "",
            marker,
        ]
    )
    path.write_text(content + analysis, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, default=EVAL_DIR / "testset.json")
    parser.add_argument("--judge-model", default=settings.llm_model, help="Ollama model used as Ragas judge")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions")
    parser.add_argument("--only", nargs="*", help="Names of the configurations to run")
    parser.add_argument(
        "--report-only", action="store_true", help="Rewrite results.md from eval/summary.json (no LLM call)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    summary_path = EVAL_DIR / "summary.json"
    if args.report_only:
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        write_report(saved["configs"], saved["n_questions"], saved["judge_model"], EVAL_DIR / "results.md")
        return 0

    testset = load_testset(args.testset)[: args.limit]
    configs = [c for c in CONFIGS if not args.only or c.name in args.only]

    summary: dict[str, dict[str, float]] = {}
    for config in configs:
        logger.info("=== %s (index=%s, top_k=%d) ===", config.name, config.index_dir, config.top_k)
        pipeline = RAGPipeline.from_settings(settings, index_dir=config.index_dir)
        samples, latency = generate_samples(pipeline, testset, config.top_k)
        rows = score(
            samples, args.judge_model, settings.ollama_base_url, settings.embedding_model, settings.llm_num_thread
        )

        (EVAL_DIR / f"scores_{config.name}.json").write_text(json.dumps(rows, indent=2, default=str), "utf-8")
        summary[config.name] = summarize(rows, latency)
        logger.info("%s -> %s", config.name, summary[config.name])

    payload = {"n_questions": len(testset), "judge_model": args.judge_model, "configs": summary}
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(summary, len(testset), args.judge_model, EVAL_DIR / "results.md")
    logger.info("Report written to %s", EVAL_DIR / "results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
