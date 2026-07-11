from typing import Protocol

from pydantic import BaseModel, Field

from infra.config import LLM_MAX_RETRIES, LLM_MODEL, RERANK_MODE
from infra.openai_compat import parse_structured


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        """Return candidate indices in preferred order."""


class RerankResult(BaseModel):
    ranked_indices: list[int] = Field(description="Zero-based candidate indices, best first")


class NoopReranker:
    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        return list(range(len(candidates)))


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        if not candidates:
            return []
        pairs = [(query, candidate) for candidate in candidates]
        scores = self.model.predict(pairs)
        return sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)


class LLMReranker:
    def __init__(self, client=None, model: str = LLM_MODEL):
        if client is None:
            from openai import OpenAI

            client = OpenAI(max_retries=LLM_MAX_RETRIES)
        self.client = client
        self.model = model

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        if not candidates:
            return []
        prompt = (
            "Rank these product comparables for pricing the query item. "
            "Return zero-based candidate indices from best to worst.\n\n"
            f"Query:\n{query}\n\n"
            "Candidates:\n"
            + "\n".join(f"{i}: {candidate}" for i, candidate in enumerate(candidates))
        )
        result = parse_structured(
            self.client,
            model=self.model,
            user_prompt=prompt,
            text_format=RerankResult,
        )
        return _validated_order(result.ranked_indices, len(candidates))


def _validated_order(indices: list[int], n: int) -> list[int]:
    if len(indices) != n or sorted(indices) != list(range(n)):
        return list(range(n))
    return indices


def build_reranker(mode: str = RERANK_MODE) -> Reranker:
    if mode == "off":
        return NoopReranker()
    if mode == "cross-encoder":
        return CrossEncoderReranker()
    if mode == "llm":
        return LLMReranker()
    raise ValueError(f"Unknown RERANK_MODE '{mode}'. Use off, cross-encoder, or llm.")
