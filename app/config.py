"""Single source of truth for model and inference settings.

Previously the model id ``gpt-4o-mini`` and the embedding model name were hardcoded
in four+ files. Centralizing them here means one place to change for a model upgrade,
and one place that the build script and the query path both read so the embedder can
never silently drift apart from the indexed vectors.
"""
import os


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

# Embedding model for the RAG vector store. The build script writes this name into
# the Chroma collection metadata; FrontierAgent reads it back and refuses to query a
# store built with a different embedder (mismatched vector spaces => garbage neighbors).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
