# Auto Deal Hunter

Auto Deal Hunter is a Python app that scans retail deal feeds, estimates a product's market value via **RAG** over an independent Amazon reference set, and surfaces deals whose current price appears meaningfully below that estimate. Its scan/estimate/notify capabilities are also exposed as MCP tools for reuse by any MCP client.

## Demo

![Gradio demo](docs/assets/gradio-demo.png)

The Gradio app shows saved opportunities, live agent logs, and a 3D projection of the embedded product reference library.

## Features

- **Scan** — Fetches new retail deals from DealNews RSS feeds for Electronics, Computers, and Smart Home.
- **Estimate** — Retrieves similar Amazon catalog items from ChromaDB and asks an LLM (default `gpt-4o-mini`, set via `LLM_MODEL`) for a fair-value estimate.
- **Cost-aware** — Logs LLM token usage and an estimated dollar cost per run.
- **Guardrail** — Caps reported savings at the seller's list price when a list price is available, reducing false bargains from high model estimates.
- **Identity-aware** — Detects multi-packs, bundles, and subscriptions so a listing isn't mis-valued against single-unit comparables: multi-packs are rebased to a per-unit price (on both the deal and its comparables), while bundles and subscriptions are skipped.
- **Notify** — Sends optional Pushover notifications for compelling deals. Estimates built on a
  weak RAG match (no close comparable in the vector store) fall below `RAG_MIN_CONFIDENCE` and are
  still saved but held back from notification, cutting false bargains at the source.
- **Gradio UI** — Displays the opportunity table, guardrail summary, logs, and a 3D t-SNE map of the vector store.

## Prerequisites

- Python 3.10+
- OpenAI API key for the agent loop and price estimator (or an OpenAI-compatible endpoint — see
  [`OPENAI_BASE_URL`](#environment-variables))
- [Hugging Face](https://huggingface.co/) access for downloading the McAuley-Lab dataset
- Optional: Pushover user/app tokens for push notifications

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

Installing the project (`pip install -e .`) puts the package tree (`app`, `agents`, `domain`,
`ingest`, `core`, `infra`, `evaluation`, `scripts`) on the import path. The MCP client also
passes the project root to the spawned MCP server process so its package imports resolve the
same way when scans run from the Gradio UI.

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Required: `OPENAI_API_KEY`. Set `HF_TOKEN` if Hugging Face requires authentication for the dataset download. Optional: `PUSHOVER_USER`/`PUSHOVER_TOKEN` for push notifications. See [Environment Variables](#environment-variables) for the full list.

## Quick Start

### 1. Build the vector store

Build this once before running the agent. The script downloads the Electronics category of
[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
for RAG-based price estimation.

For a faster first run, build a smaller local store:

```bash
MCAULEY_MAX_ITEMS=1000 EVAL_HOLDOUT_SIZE=50 python scripts/build_vector_store.py
```

For the default larger store:

```bash
python scripts/build_vector_store.py
```

By default, the vector store is written to `data/products_vectorstore/`. Set
`PRODUCTS_VECTORSTORE_PATH` to use a different location. A holdout sample is saved to
`data/eval_holdout.json` and is excluded from the vector store, so `scripts/eval_pricers.py`
can measure generalization without retrieving the exact test item.

### 2. Run the agent

```bash
python -m app.ui
```

This opens a Gradio UI in your browser. The app scans deals, estimates values, saves
opportunities to `data/deals.sqlite`, and auto-refreshes every 5 minutes. Saved opportunities
not re-confirmed within `DEALS_TTL_HOURS` (default 72h) are pruned so the table stays focused
on currently-live deals.

## Docker

Docker is the easiest way to run the app with a clean Python environment.

1. Create `.env` from the example and fill in your keys:

```bash
cp .env.example .env
```

2. Build the vector store once. The compose service mounts `./data` into the container, so the generated vector store and SQLite runtime data persist across container rebuilds:

```bash
docker compose run --rm auto-deal-hunter python scripts/build_vector_store.py
```

For a faster first run:

```bash
docker compose run --rm -e MCAULEY_MAX_ITEMS=1000 -e EVAL_HOLDOUT_SIZE=50 auto-deal-hunter python scripts/build_vector_store.py
```

3. Start the Gradio app:

```bash
docker compose up --build
```

Open `http://localhost:7860` after the container starts.

## How It Works

```text
   DealNews RSS ──▶ ScannerAgent    PricerAgent            MessagingAgent ──▶ Pushover
                    (filter + LLM   (RAG: ChromaDB +        (LLM-crafted
                     selection)      LLM estimate)           message)
                          │               │                     ▲
                          ▼               ▼                     │
   scan pipeline ──▶ candidates ──▶ estimates ──▶ deterministic best deal (max total capped savings)
                                                          │
                                                          ▼
                                              SQLite store + Gradio UI
```

1. `ScannerAgent` fetches DealNews RSS entries and extracts product details, deal price, and list price when available.
2. Used, refurbished, renewed, open-box, and pre-owned items are filtered out before selection. Bundles and subscriptions are skipped, and multi-packs are rebased to a per-unit price so they are valued against single-unit comparables.
3. `PricerAgent` embeds deal descriptions and retrieves similar Amazon products from ChromaDB.
4. An LLM (default `gpt-4o-mini`) estimates market value from the retrieved product context and returns a structured price.
5. The pipeline *gathers* candidates and their estimates, but the single best deal is chosen **deterministically**, not by model judgment ([`core/scoring.py`](core/scoring.py); see [Design notes](#design-notes-the-scan-pipeline-and-a-deterministic-selector)).
6. A push is sent for the best deal — unless its estimate rests on a weak RAG match (below `RAG_MIN_CONFIDENCE`), in which case the deal is still saved but not notified.
7. Gradio displays opportunities, live logs, guardrail summary, and a 3D t-SNE view of the vector store.

Each run also logs LLM token usage and an estimated dollar cost ([`infra/usage.py`](infra/usage.py)), and scraped DealNews pages are cached on disk ([`infra/http_cache.py`](infra/http_cache.py)) so repeated scans are fast and gentle on the source.

### Design notes: the scan pipeline and a deterministic selector

The deal-hunting flow is mostly deterministic (scan → estimate → score candidates), so the LLM
is **not** trusted to choose the winner. By default the pipeline runs **in-process**
([`app/pipeline.py`](app/pipeline.py)): it calls the scanner, estimates each candidate, and
selects the best by a plain `max` over the list-price-capped total savings (per-unit discount ×
pack size, [`core/scoring.py`](core/scoring.py)) — keeping the LLM for the parts it is good at
(summarizing listings and estimating value from context).

The same three capabilities are also exposed as MCP tools ([`app/mcp_server.py`](app/mcp_server.py))
so any MCP client can reuse them, and an LLM tool-calling loop that drives those tools
([`app/mcp_client.py`](app/mcp_client.py)) is kept as a demonstration of MCP orchestration —
opt in with `SCAN_MODE=agent`. The direct pipeline is the default because it avoids the agent
path's incidental complexity (spawning a subprocess, injecting `PYTHONPATH`, merging token
usage across the process boundary, and re-pairing tool-call arguments to scanned deals) for a
flow whose outcome is deterministic either way.

## Estimate quality guardrail

The opportunity table reports savings as `min(estimate, list_price) - deal_price` when a list
price is available, and as `estimate - deal_price` when no list price is known. This keeps a high
model estimate from manufacturing savings above the seller's own list price.

- **The estimate stays independent.** The pricer never sees the seller's list/MSRP price, so
  the estimate can't simply echo it.
- **`list_price` is a downstream sanity bound.** A new-retail item's fair value should not
  exceed its original price. The dashboard shows the share of checkable deals whose estimate
  exceeds list price. Deals with no detected list price are left unchecked rather than penalized.

## Evaluation

Two scripts score the system on the held-out McAuley sample in `data/eval_holdout.json` (items
excluded from the vector store, so they measure generalization rather than exact-match lookup).

> **Figures depend on the model, the vector store size, and `--size`,** so they drift over time.
> The sample outputs below show the metric format only — run the scripts yourself for current
> numbers.

### Pricer (end-to-end)

```bash
python scripts/eval_pricers.py --size 200
```

Runs `PricerAgent` (RAG + LLM) against the holdout, scoring each estimate against the item's
known price. To save aggregate metrics and fail the command when quality drifts beyond limits
(useful in CI):

```bash
python scripts/eval_pricers.py --size 200 --output-json data/eval_metrics.json --max-mae 150 --max-abs-bias 50 --max-over-rate 0.65
```

The command prints aggregate accuracy metrics plus the run's LLM token cost:

```text
MAE: $<dollars>   RMSE: $<dollars>   Bias: +$<dollars>   Over-prediction: <pct>%   n=<count>
LLM usage: <calls> calls, <in> in + <out> out tokens, ~$<cost>
```

| Metric | Meaning | How to read it |
|--------|---------|----------------|
| **MAE** | Mean absolute error, in dollars | Average miss regardless of direction — lower is better |
| **RMSE** | Root mean squared error, in dollars | Like MAE but penalizes large misses more — lower is better |
| **Bias** | Mean *signed* error (estimate − truth) | Direction of the error: `+` means the pricer runs high on average, `−` means low; near `$0` is well-centered |
| **Over-prediction** | Share of items estimated above the true price | How often it guesses high; well above 50% signals a systematic upward tilt |

**Bias** and **over-prediction** are the two to watch: they quantify the upward tilt that makes
estimates exceed a deal's list price — a positive bias with an over-prediction rate well above
50% is exactly the tilt the list-price guardrail exists to absorb.

### Retriever (no LLM)

End-to-end price error blends two failure modes: a bad retriever (wrong neighbors) and a bad LLM
(wrong reasoning over good neighbors). To isolate the retriever — no LLM calls, near-free — run:

```bash
python scripts/eval_retrieval.py --size 200 --k 5
```

The command prints per-k retrieval metrics:

```text
category_precision@5: <pct>%   hit_rate@5: <pct>%   price_MAPE@5: <pct>%   n=<count>
```

It scores `category_precision@k` (are the neighbors the right category?), `hit_rate@k` (at least
one same-category neighbor?), and `price_MAPE@k` (how close the median neighbor price is to the
true price). Use this to tell whether a high pricer MAE is a retrieval problem or a reasoning
problem before tuning either side.

### Reproducibility

All LLM calls run at `temperature=0` (`LLM_TEMPERATURE`) with a fixed `seed` (`LLM_SEED`). Note
that OpenAI's `seed` is **best-effort**, not a hard guarantee — outputs can still vary slightly
between runs and across model updates. Treat the eval as low-variance, not bit-exact.

## Development

Run the test suite (unit tests for the scrapers, agents, MCP hand-off, deterministic selector,
cost tracker, and eval metrics):

```bash
python -m unittest discover -s tests
```

Lint with `ruff check .`. Tests run offline — network, OpenAI, and Sentence-Transformers calls
are stubbed — so no API key or vector store is required.

### Reproducible installs

`pyproject.toml` pins only version *ranges* (with upper bounds capping the next major of the
fast-moving libraries), which is right for development. For a byte-for-byte reproducible deploy,
generate a lockfile **in the target environment** (matching OS and Python version — a macOS lock
won't match a Linux container):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip freeze --exclude-editable > requirements.lock
```

Commit `requirements.lock` and install from it in the container (`pip install -r requirements.lock`).
Regenerate it whenever `pyproject.toml` dependencies change. Do **not** run `pip freeze` from a broad
Anaconda/base environment — unrelated packages and local file URLs leak into the snapshot.

## Project Structure

```text
auto-deal-hunter/
├── app/                   # Application entrypoint + orchestration
│   ├── ui.py              # Gradio UI (entrypoint)
│   ├── orchestrator.py    # Run loop: scan → estimate → select → persist
│   ├── mcp_client.py      # Agentic loop driving the MCP tools
│   └── mcp_server.py      # MCP server exposing scan/estimate/notify tools
├── agents/                # Agent implementations
│   ├── agent.py
│   ├── scanner_agent.py
│   ├── pricer_agent.py    # RAG + LLM fair-value estimator
│   └── messaging_agent.py
├── domain/                # Pure domain models (no I/O)
│   ├── deal.py            # Deal, DealSelection, Opportunity, deal_id
│   ├── identity.py        # Product identity kinds and metadata
│   └── item.py            # McAuley catalog item
├── ingest/                # DealNews scraping + price extraction
│   ├── identity.py        # Deterministic product-identity extraction
│   ├── scraper.py         # ScrapedDeal, RSS fetch, new-retail filtering
│   └── list_price.py      # List/deal-price regex + widget parsing
├── core/                  # Business logic
│   ├── identity_policy.py # Priceability and per-unit rebasing rules
│   ├── scoring.py         # Deterministic best-deal selection
│   └── opportunity_store.py  # SQLite persistence
├── infra/                 # Cross-cutting infrastructure
│   ├── config.py          # Single source of truth for model/embedding settings
│   ├── usage.py           # LLM token + cost accounting
│   ├── http_cache.py      # On-disk read-through cache for scraped pages
│   ├── log_utils.py
│   └── paths.py           # Shared runtime/data paths
├── evaluation/            # Importable, tested eval metrics
│   ├── pricer.py          # MAE / bias / over-prediction
│   └── retrieval.py       # category_precision@k / hit_rate@k / price_MAPE@k
├── scripts/               # Thin CLI wrappers
│   ├── audit_identity.py   # Identity-rule audit over live scraped deals
│   ├── build_vector_store.py
│   ├── eval_pricers.py    # End-to-end price accuracy (RAG + LLM)
│   └── eval_retrieval.py  # Retriever-only quality (no LLM)
├── data/                  # Runtime state and local vector store
└── pyproject.toml
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| **Credentials** | |
| `OPENAI_API_KEY` | API key for the MCP agent loop and RAG estimator (OpenAI, or the configured OpenAI-compatible endpoint) |
| `OPENAI_BASE_URL` | Optional. Point the OpenAI client at an OpenAI-compatible Chat Completions endpoint. The endpoint must support the features this app uses, especially tool calling; structured-output and `seed` behavior vary by backend |
| `HF_TOKEN` | Optional Hugging Face API token, used when dataset access or rate limits require authentication |
| `PUSHOVER_USER` | Pushover user key (for push notifications) |
| `PUSHOVER_TOKEN` | Pushover app token |
| **Model & runtime config** | |
| `SCAN_MODE` | `direct` (default) runs the scan pipeline in-process; `agent` drives the MCP tool server through an LLM tool-calling loop (demo of MCP orchestration). Both select the same deterministic best deal |
| `LLM_MODEL` | Chat model used by every agent, served by the configured OpenAI-compatible endpoint (default: `gpt-4o-mini`) |
| `LLM_TEMPERATURE` | Sampling temperature for all LLM calls (default: `0`) |
| `LLM_SEED` | Best-effort sampling seed for reproducibility (default: `42`) |
| `LLM_MAX_RETRIES` | Automatic retries with exponential backoff for transient OpenAI errors, per client (default: `3`) |
| `EMBEDDING_MODEL` | Sentence-Transformers model for the vector store; must match between build and query (default: `sentence-transformers/all-mpnet-base-v2`) |
| `RAG_MIN_CONFIDENCE` | Minimum retrieval confidence (`0`–`1`) for a deal to be push-notified; low-confidence deals are still saved but not pushed. `0` disables the gate (default: `0.15`) |
| `DEALHUNTER_HTTP_CACHE` | Set to `0`/`off`/`false` to disable the scraped-page cache (default: on) |
| **Paths & data** | |
| `PRODUCTS_VECTORSTORE_PATH` | Path to ChromaDB store (default: `data/products_vectorstore`) |
| `DEALS_DB_PATH` | SQLite runtime database path (default: `data/deals.sqlite`) |
| `DEALS_TTL_HOURS` | Prune opportunities not re-confirmed within this many hours; `0` disables expiry (default: `72`) |
| `MEMORY_FILENAME` | Legacy JSON memory path imported on startup when present (default: `data/memory.json`) |
| `MCAULEY_CATEGORY` | McAuley-Lab category to pull (default: `Electronics`) |
| `MCAULEY_MAX_ITEMS` | Cap on items embedded into the vector store (default: `50000`) |
| `EVAL_HOLDOUT_SIZE` | Items held out for `eval_pricers.py` (default: `500`) |
| `EVAL_HOLDOUT_PATH` | Holdout sample path used by `eval_pricers.py` (default: `data/eval_holdout.json`) |

## Limitations

- **Reference-based estimates, not live quotes.** `Est. Value` is computed from a static local
  vector store plus an LLM. It is useful for ranking deals, but it can lag current market prices,
  especially for fast-moving categories, discontinued products, new releases, and long-tail brands
  with weak comparables.
- **Source-dependent scans.** Deal quality depends on RSS feed availability and DealNews page
  structure.
- **Runtime dependencies can fail.** Scans depend on DealNews pages, the local vector store, and an
  OpenAI-compatible endpoint. Network errors, model/API failures, or missing vector-store data can
  interrupt a run.
- **Screening signal only.** Treat estimates as a shortlist signal, not as the sole reason to buy.
- **Local build cost.** The first vector-store build can take time and disk space, especially with
  the default 50,000-item cap.

## Future Improvements

- **Estimate confidence metadata.** Track comparable count, similarity distance, price spread,
  vector-store build timestamp, and identity/quantity/variant match strength. Low-confidence deals
  could be displayed but skipped for push notifications.
- **Optional live-price layer.** Add exact live-match pricing when high-confidence same-spec
  listings are available, and optionally refresh RAG comparables with current prices. Compute a
  median or weighted median from valid new, in-stock listings while excluding DealNews and
  same-source mirrors.
- **Live data isolation.** Cache live lookups with timestamps and keep them out of the reproducible
  offline eval path. Most stable live-price sources are paid, rate-limited, or account-gated.
- **Accuracy tracking over time.** Record surfaced deals, later observed prices, and manual
  accepted/rejected labels to measure whether the system is reducing false positives.

## License

MIT License. See [LICENSE](LICENSE) for details.
