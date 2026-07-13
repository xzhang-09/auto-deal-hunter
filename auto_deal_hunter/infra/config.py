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


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Chat model used by every agent (scanner, estimator, messenger, MCP loop).
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def _model_override(name: str) -> str:
    return os.getenv(name) or LLM_MODEL


SCANNER_MODEL = _model_override("SCANNER_MODEL")
PRICER_MODEL = _model_override("PRICER_MODEL")
MESSAGING_MODEL = _model_override("MESSAGING_MODEL")
JUDGE_MODEL = _model_override("JUDGE_MODEL")
MCP_MODEL = _model_override("MCP_MODEL")
OPENAI_API_STYLE = (os.getenv("OPENAI_API_STYLE") or "responses").lower()

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
# for more missed deals. Set to 0 to disable the gate.
RAG_MIN_CONFIDENCE = _get_float("RAG_MIN_CONFIDENCE", 0.15)

# Estimates above this multiple of the seller's list price are treated as retrieval
# mismatches (the RAG neighbors were the wrong kind of product) rather than ordinary
# estimator noise: the deal keeps its list-price-capped savings and stays in the store,
# but its push confidence is zeroed and the dashboard shows its estimate as unreliable.
# Ordinary overestimates run a few percent above list; mismatches run at multiples of it,
# so 2x cleanly separates the two. Raise it if legitimate deals get flagged.
ESTIMATE_MISMATCH_RATIO = _get_float("ESTIMATE_MISMATCH_RATIO", 2.0)

# Distance metric for the Chroma collection. all-mpnet-base-v2 is tuned for cosine
# similarity, and a bounded cosine distance ([0, 2]) is what an interpretable RAG-confidence
# threshold needs -- raw L2 on un-normalized vectors has no fixed scale. The build script
# stamps this into the collection's hnsw:space and normalizes embeddings; PricerAgent reads it
# back and refuses a store built with a different metric. Changing it requires a rebuild
# (Chroma cannot alter a collection's space in place).
VECTOR_SPACE = "cosine"

# Optional second-stage retrieval re-ranking. Default is off so local runs and CI do not
# download a cross-encoder model unless explicitly requested.
RERANK_MODE = os.getenv("RERANK_MODE", "off").lower()
RERANK_CANDIDATES = _get_int("RERANK_CANDIDATES", 20)

# Telegram feedback uses a background long-polling thread. Keep it opt-in so importing or
# testing the app never starts network activity merely because notification credentials exist.
TELEGRAM_FEEDBACK_ENABLED = os.getenv("TELEGRAM_FEEDBACK_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TELEGRAM_POLL_TIMEOUT_SECONDS = _get_int("TELEGRAM_POLL_TIMEOUT_SECONDS", 25)

# Deals are ephemeral: DealNews edits/expires listings, so a stored opportunity is a
# snapshot that goes stale. Opportunities not re-confirmed within this window are pruned
# so the store reflects currently-live deals instead of growing without bound. Set to 0
# (or negative) to disable expiry and keep every opportunity forever.
DEALS_TTL_HOURS = _get_float("DEALS_TTL_HOURS", 72.0)
