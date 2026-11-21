# Ragas evaluation results

- Date: 2026-11-21
- Test set: 10 hand-written questions (`eval/testset.json`)
- Judge LLM: `llama3.2:3b` (Ollama, local), embeddings: `nomic-embed-text`
- Scores are in [0, 1], higher is better. Each cell is the mean over the questions where the judge
  returned a valid score, followed by `(valid / total)`; `n/a` = no valid score.

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall | Mean latency (s) |
|---|---|---|---|---|---|
| `chunk500_k3` | 0.375 (2/10) | 0.877 (2/10) | n/a (0/10) | 0.783 (9/10) | 58.8 |
| `chunk1000_k3` | 1.000 (1/10) | 0.676 (4/10) | 1.000 (4/10) | 0.773 (9/10) | 75.0 |

<!-- RESULTS:END -->

## Setup

- Two configurations that differ **only** by chunking: `chunk_size=500, overlap=75` (3 268 chunks) vs
  `chunk_size=1000, overlap=150` (1 613 chunks); same embeddings, same LLM, `top_k=3`, temperature 0.
- The first 10 questions of the 26-question test set were evaluated (CPU time, see below).
- Raw per-question scores: `eval/scores_chunk500_k3.json`, `eval/scores_chunk1000_k3.json`; summary: `eval/summary.json`.

## Reliability of the Ragas scores

The judge is a 3B model running on CPU. Many judge calls timed out (900 s) or returned JSON that Ragas could not
parse, which gives `NaN`. Only **context recall** has a valid score for 9/10 questions in both configurations;
the other means rest on 0 to 4 questions and **cannot be used to compare the configurations** (for example the
faithfulness of 1.000 for `chunk1000_k3` is a single question). This is the main lesson of this run: an
LLM-as-a-judge evaluation is only as good as the judge.

## Manual check of the answers

Because of this, every generated answer was also compared by hand with the reference answer
(correct = all key facts right; partial = right but incomplete or off-focus; wrong = a false fact).

| Configuration | Correct | Partial | Wrong |
|---|---|---|---|
| `chunk500_k3` | 5 | 3 | **2** |
| `chunk1000_k3` | **8** | 2 | 0 |

The two errors of `chunk500_k3` are hallucinations caused by fragmented context:

- **Q4 (CronJob `concurrencyPolicy`)**: answers `AllowConcurrent` / `ForbidConcurrent`, values that do not exist,
  and misses `Replace`. With 500-character chunks the list of values is split across chunks, so the model only
  sees part of it and fills the gap. With 1000-character chunks the whole list is in one chunk and the answer is exact.
- **Q8 (QoS eviction order)**: gives the **reverse** order (Guaranteed evicted first). The sentence with the order
  and the definition of the classes ended up in different chunks.

## Conclusions

- **chunk_size 1000 is the better configuration** for this corpus: 8/10 correct answers and no hallucination,
  against 5/10 and 2 hallucinations with 500. Larger chunks keep lists and explanations together.
- **Context recall is equivalent** (0.78 vs 0.77): with `top_k=3` both configurations usually retrieve the right
  page; the difference comes from how complete the retrieved passage is, not from finding the page.
- **Latency is higher with 1000** (75 s vs 59 s per answer on CPU) because the prompt is about twice as long.
  This is the cost of the quality gain; with a GPU both are a few seconds.
- The default configuration of the API is therefore `CHUNK_SIZE=1000, CHUNK_OVERLAP=150`.

## Next steps for a more reliable evaluation

- Use a stronger judge (e.g. `qwen2.5:14b` or larger on a GPU) and run the full 26 questions.
- Run each configuration several times to measure variance.
- Also compare `top_k` 3 vs 5 and add a re-ranker once the judge is reliable.

Reproduce:

```powershell
python -m eval.run_ragas --limit 10          # ~8 h on a laptop CPU
python -m eval.run_ragas --report-only       # rebuild the table from eval/summary.json
```
