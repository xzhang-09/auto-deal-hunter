# Local Evaluation Results

Snapshots of local evaluation runs, with the raw metrics saved as JSON files in this
directory. **Figures depend on the model, the vector store size, and `--size`,** so they
drift over time — treat these as dated observations, not stable benchmarks. The commands
that produce them are documented in the top-level [README](../../README.md#evaluation).

## Pricer baseline

Run against the current `data/eval_holdout.json` and vector store
([`baseline_pricer.json`](baseline_pricer.json)):

```text
python -m auto_deal_hunter.scripts.eval_pricers --size 200 --output-json docs/eval/baseline_pricer.json
MAE: $25.76   RMSE: $66.45   Bias: -$5.71   Over-prediction: 39%   n=200
LLM usage: 200 calls, 261,470 in + 2,402 out tokens, ~$0.0407
```

## Retriever baseline

Same holdout and store ([`baseline_retrieval.json`](baseline_retrieval.json)):

```text
python -m auto_deal_hunter.scripts.eval_retrieval --size 200 --k 5
category_precision@5: 57%   hit_rate@5: 86%   price_medianAPE@5: 29%   (meanAPE: 48%)   n=200
```

The high hit rate means most held-out items retrieve at least one same-category neighbor,
while the lower category precision and high mean absolute percentage error show why the app
treats retrieval as a screening/ranking signal rather than a pricing oracle.

Items whose true per-unit price is below $1 are excluded from the APE metrics (`n_ape` in
the JSON output): with a near-zero denominator, a few dollars of neighbor movement swings a
single query's APE by thousands of percentage points, which once flipped the sign of an A/B
comparison on its own before the exclusion was added.

## Re-ranker comparison

On the same local holdout, the `cross-encoder` re-ranker
([`rerank_cross_encoder_retrieval.json`](rerank_cross_encoder_retrieval.json)) moved
category precision from 56.8% to 57.5%, hit rate from 85.5% to 88.0%, and median APE from
28.6% to 28.4%, with mean APE flat (48.3% vs 48.4%). All of these differences are within
sampling noise at n=200 (the standard error on a ~57% proportion is about ±3.5 points), so
the honest reading is "no measurable gain," and the re-ranker stays opt-in rather than
becoming the default: it adds a cross-encoder inference pass per deal without a demonstrated
retrieval improvement.

The same conclusion holds one level up and one level down. The `llm` re-ranker scored
category precision 56.3%, hit rate 86.5%, and median APE 25.2%
([`rerank_llm_retrieval.json`](rerank_llm_retrieval.json)) — again within noise of the
baseline, while adding an LLM call per retrieval, so it is the most expensive way to not
improve the metrics. End-to-end (`RERANK_MODE=cross-encoder
python -m auto_deal_hunter.scripts.eval_pricers --size 200`,
[`rerank_cross_encoder_pricer.json`](rerank_cross_encoder_pricer.json)), the pricer scored
MAE $23.91 / RMSE $56.77 / bias −$12.51 versus the baseline's MAE $25.76 / RMSE $66.45 /
bias −$5.71 — MAE within noise, with a slightly stronger low tilt. One holdout item also
became unpriceable under re-ranking (the reshuffled comparables made the model echo the
prompt placeholder), which the eval now reports as `n_failed` instead of crashing — worth
watching, since a config change that shifts retrieval can push individual items over the
pricer's fail-loudly edge.

Per-query error analysis of an earlier run also showed a real failure mode worth knowing
about: `ms-marco` cross-encoders score topical relevance, not price comparability, so they
can promote a lexically better match from the wrong price tier — e.g. ranking a $39.99
square hood first for a $9.99 lens-hood query because both say "77mm", or promoting an
$89.99 triple-pack fan for a $14 single-fan query on a brand match. Across the holdout these
promotions roughly balance out (68 queries got worse by >1pt, 70 got better), but they are
the thing to fix — likely with a price-aware re-ranking objective — before the re-ranker can
earn the default slot.

## Scanner model comparison

One local batch ([`scanner_models.json`](scanner_models.json)), produced with
`scripts/compare_scanner_models.py`: both models selected the **same five deals**;
`gpt-4.1-nano` scored 5/5 faithful at ~35% lower scan cost, while `gpt-4o-mini` had one
summary flagged for adding specs not present in the listing (injected from prior product
knowledge — exactly the failure the judge exists to catch). A single RSS batch with five
selections per model is directional, not conclusive: rerun on a few different batches before
setting `SCANNER_MODEL=gpt-4.1-nano`, but the mechanism and the measurement are in place.

## Message judge

Run over 8 saved opportunities with `--negative-control`
([`message_judge.json`](message_judge.json)):

```text
faithfulness_rate=100% mean_score=5.00 n=8
negative control: judge_recall=100% on 24 corrupted messages
LLM usage: 40 calls, 12,954 in + 2,086 out tokens, ~$0.0032
```

All 24 corrupted messages were caught (scores dropped to 1–2 with the specific misstatement
named in `issues`), and all 8 clean messages passed — so the 100% faithfulness rate on real
messages reflects a judge that demonstrably catches violations, not one that rubber-stamps.

## Embedding comparison

See [`embeddings/`](embeddings/) for the raw stores/metrics and
[`../embeddings.md`](../embeddings.md) for the write-up.
