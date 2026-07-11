# Embedding Comparison

This project can compare retrieval quality across embedding models using a fixed McAuley
corpus size and the same retriever-only metrics used elsewhere.

Run a quick smoke comparison:

```bash
python -m scripts.compare_embeddings --max-items 1000 --holdout-size 100 --size 50
```

Run the fuller comparison:

```bash
python -m scripts.compare_embeddings --max-items 10000 --size 200
```

The script builds one Chroma store per model under `data/embedding_compare/` and writes
per-model metrics plus `docs/eval/embeddings/summary.json`. It uses a comparison-specific
holdout file at `docs/eval/embeddings/holdout.json` so it does not overwrite the default
`data/eval_holdout.json`.

Default models:

| Model | Why compare it |
|-------|----------------|
| `sentence-transformers/all-mpnet-base-v2` | Current default; stronger general-purpose semantic embedding |
| `sentence-transformers/all-MiniLM-L6-v2` | Smaller/faster baseline for speed vs. accuracy trade-off |
| `BAAI/bge-base-en-v1.5` | Retrieval-tuned embedding family |

Full comparison (`--max-items 10000 --size 200`, APE metrics exclude near-zero-price rows via
`APE_MIN_TRUTH_PRICE`; `n_ape=197` of 200 for all three models):

| Model | category_precision@5 | hit_rate@5 | price_medianAPE@5 | meanAPE | Notes |
|-------|----------------------|------------|-------------------|---------|-------|
| `all-mpnet-base-v2` | 52.7% | 82.0% | 39.9% | 65.9% | Current default |
| `all-MiniLM-L6-v2` | 52.2% | 81.5% | 37.5% | 69.0% | ~5× smaller model, faster build |
| `bge-base-en-v1.5` | 52.0% | 82.0% | 35.0% | 57.9% | Retrieval-tuned; evaluated *without* its recommended query prefix |

How to read this honestly: at n=200 the standard error on a ~52% proportion is about ±3.5
points, so **category precision and hit rate are statistically indistinguishable across all
three models**. The price-error metrics lean consistently toward `bge-base-en-v1.5`
(medianAPE 35.0% vs 39.9%, meanAPE 57.9% vs 65.9%), but the gaps are modest and come from a
single run on one holdout — treat them as a direction worth confirming, not an established
ranking.

An earlier version of this comparison reported meanAPE of 159–183%. Those numbers were
produced before `APE_MIN_TRUTH_PRICE` existed: three near-zero-price holdout rows contributed
over 100 points of meanAPE by themselves (the same failure mode that once flipped a re-ranker
A/B comparison — see README → Evaluation). Category precision and hit rate reproduced exactly
across the rebuild, confirming the build is deterministic; only the APE columns changed.

Conclusion: stay on `all-mpnet-base-v2` — not because it "wins" (on these numbers nothing
does), but because it is the validated production path and no candidate shows a gain large
enough to clear the noise floor. `bge-base-en-v1.5` remains the most promising challenger:
it trends better on price error *despite* being evaluated without its recommended
"Represent this sentence…" query prefix. A switch should be gated on (1) a prefix-enabled
rerun, (2) an end-to-end `eval_pricers.py` comparison, and (3) the gain exceeding sampling
noise.

For a fast pipeline check after changing the script, a smoke run
(`--max-items 1000 --holdout-size 100 --size 50`) exercises all the plumbing in a few
minutes, but its n=50 metrics are far too noisy to compare models with — don't read
conclusions out of it.

Do not treat embedding choice as a config comment only. Keep the selected default tied to
the measured trade-off between retrieval quality, build time, local disk use, and runtime
latency.
