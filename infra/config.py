"""Single source of truth for model and inference settings.

Previously the model id ``gpt-4o-mini`` and the embedding model name were hardcoded
in four+ files. Centralizing them here means one place to change for a model upgrade,
and one place that the build script and the query path both read so the embedder can
never silently drift apart from the indexed vectors.
"""
import os

from dotenv import load_dotenv

# Load .env as early as possible: the constants below are read at import time, so .env must
# be applied before any consumer imports this module. Doing it here removes the ordering
# dependency that otherwise required every entrypoint to call load_dotenv before importing
# config. override=True matches the rest of the app's load_dotenv usage.
load_dotenv(override=True)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Chat model used by every agent (scanner, estimator, messenger, MCP loop).
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# temperature=0 is the strongest reproducibility lever we have; `seed` is only
# best-effort on OpenAI's side, so we set both. See README "Reproducibility".
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.0)
LLM_SEED = int(os.getenv("LLM_SEED", "42"))

# Automatic retries (with exponential backoff, handled by the OpenAI SDK) for transient
# failures -- connection errors, timeouts, 429s, 5xx. Without this a single network blip
# aborts an entire scan. Applied uniformly to every OpenAI client the app constructs.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Embedding model for the RAG vector store. The build script writes this name into
# the Chroma collection metadata; PricerAgent reads it back and refuses to query a
# store built with a different embedder (mismatched vector spaces => garbage neighbors).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# Minimum retrieval confidence, in [0, 1], for a deal to earn a push notification. Confidence
# is derived from the nearest RAG comparable's cosine distance (1 - distance): a deal whose
# closest comparable is essentially unrelated has no trustworthy basis for its estimate, so it
# is still saved to the store but NOT pushed. The default is deliberately low -- it only
# suppresses near-orthogonal (no-good-comparable) matches; raise it to trade fewer false pushes
# for more missed deals. Set to 0 to disable the gate (push everything, as before).
RAG_MIN_CONFIDENCE = _get_float("RAG_MIN_CONFIDENCE", 0.15)

# Distance metric for the Chroma collection. all-mpnet-base-v2 is tuned for cosine
# similarity, and a bounded cosine distance ([0, 2]) is what an interpretable RAG-confidence
# threshold needs -- raw L2 on un-normalized vectors has no fixed scale. The build script
# stamps this into the collection's hnsw:space and normalizes embeddings; PricerAgent reads it
# back and refuses a store built with a different metric. Changing it requires a rebuild
# (Chroma cannot alter a collection's space in place).
VECTOR_SPACE = "cosine"

# Deals are ephemeral: DealNews edits/expires listings, so a stored opportunity is a
# snapshot that goes stale. Opportunities not re-confirmed within this window are pruned
# so the store reflects currently-live deals instead of growing without bound. Set to 0
# (or negative) to disable expiry and keep every opportunity forever.
DEALS_TTL_HOURS = _get_float("DEALS_TTL_HOURS", 72.0)
