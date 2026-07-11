import json
import re
from typing import List
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from agents.agent import Agent
from infra.config import (
    EMBEDDING_MODEL,
    LLM_MAX_RETRIES,
    LLM_SEED,
    LLM_TEMPERATURE,
    PRICER_MODEL,
    RERANK_CANDIDATES,
    RERANK_MODE,
    VECTOR_SPACE,
)
from infra import usage
from infra.openai_compat import parse_structured
from core.reranker import build_reranker


# Placeholder number shown in the prompt's JSON format example. It is intentionally invalid as a
# fair-value estimate: if the model echoes the example when RAG context is uninformative, the
# guard in `price()` catches it instead of surfacing a fabricated positive value.
PLACEHOLDER_PRICE = 0.0


class PricerAgent(Agent):
    name = "Pricer Agent"
    color = Agent.BLUE
    MODEL = PRICER_MODEL

    def __init__(self, collection):
        self.log("Initializing")
        self.client = OpenAI(max_retries=LLM_MAX_RETRIES)
        self.collection = collection
        self._check_embedding_model(collection)
        self._check_distance_space(collection)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = build_reranker(RERANK_MODE)
        self.log("Ready")

    def _check_embedding_model(self, collection) -> None:
        """Refuse to query a store built with a different embedder.

        The build script stamps the embedding model name into the collection metadata.
        Querying with a different model produces vectors in an incompatible space and
        silently returns nonsense neighbors, so fail loudly instead. Stores without metadata
        are allowed with a warning."""
        metadata = getattr(collection, "metadata", None) or {}
        built_with = metadata.get("embedding_model")
        if built_with is None:
            self.log("Vector store has no embedding_model metadata; assuming it matches config")
        elif built_with != EMBEDDING_MODEL:
            raise ValueError(
                f"Vector store was built with embedding model '{built_with}' but config "
                f"EMBEDDING_MODEL is '{EMBEDDING_MODEL}'. Rebuild the store or fix the config."
            )

    def _check_distance_space(self, collection) -> None:
        """Refuse to query a store built with the wrong distance metric.

        The build script stamps hnsw:space into the collection metadata. The query path is
        built around cosine distance (mpnet is tuned for cosine; a bounded distance is needed
        for an interpretable confidence threshold), so a store built with L2 ranks neighbors
        differently and yields un-thresholdable distances -- fail loudly. Legacy stores predate
        this stamp and have no key; warn and recommend a rebuild rather than raising, since
        their neighbors are usually close enough to keep working in the meantime."""
        metadata = getattr(collection, "metadata", None) or {}
        space = metadata.get("hnsw:space")
        if space is None:
            self.log(
                "Vector store has no hnsw:space metadata (likely built with the default L2 "
                f"metric); rebuild with build_vector_store.py for {VECTOR_SPACE} distances"
            )
        elif space != VECTOR_SPACE:
            raise ValueError(
                f"Vector store was built with distance metric '{space}' but the query path "
                f"expects '{VECTOR_SPACE}'. Rebuild the store or fix the config."
            )

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        msg = "Context - similar products:\n\n"
        for similar, price in zip(similars, prices):
            msg += f"Product: {similar}\nPrice: ${price:.2f}\n\n"
        return msg

    @staticmethod
    def _to_comparables(documents, metadatas):
        """Normalize retrieved neighbors to a single-unit basis.

        A multipack comparable (e.g. a 36-pack stored at its pack price) would otherwise drag
        the LLM's per-unit value estimate upward. When the build step recorded a quantity > 1,
        divide to a per-unit price and annotate the listing so the model reads it correctly.
        Comparables without quantity metadata are treated as single-unit listings."""
        documents_out, prices_out = [], []
        for doc, meta in zip(documents, metadatas):
            quantity = meta.get("quantity") or 1
            price = meta["price"]
            if quantity > 1:
                price = price / quantity
                doc = f"{doc} [per-unit price; originally sold as a pack of {quantity}]"
            documents_out.append(doc)
            prices_out.append(price)
        return documents_out, prices_out

    def find_similars(self, description: str):
        self.log("RAG search for similar products")
        # normalize_embeddings to match the build path: the store holds unit vectors under
        # cosine distance, so the query must be a unit vector too for consistent neighbors.
        vector = self.model.encode([description], normalize_embeddings=True)
        n_results = RERANK_CANDIDATES if RERANK_MODE != "off" else 5
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(), n_results=n_results
        )
        raw_documents = results["documents"][0][:]
        raw_metadatas = results["metadatas"][0][:]
        order = self.reranker.rerank(description, raw_documents)[:5]
        ordered_documents = [raw_documents[i] for i in order]
        ordered_metadatas = [raw_metadatas[i] for i in order]
        documents, prices = self._to_comparables(ordered_documents, ordered_metadatas)
        # Cosine distances to the neighbors (parallel to documents); used to score confidence.
        # Confidence intentionally remains based on the pre-rerank nearest neighbor so its
        # cosine-distance semantics stay stable across reranker modes.
        distances = results.get("distances")
        distances = list(distances[0][:]) if distances else []
        return documents, prices, distances

    @staticmethod
    def _retrieval_confidence(distances: List[float]) -> float:
        """Confidence in [0, 1] from the nearest comparable's cosine distance (``1 - distance``).

        A close nearest neighbor (small distance) means a trustworthy comparable exists; a
        distant one means the RAG context is weak, so the estimate rests on nothing solid. No
        neighbors at all -> 0. The notify path uses this to withhold a push for weak matches."""
        if not distances:
            return 0.0
        nearest = min(distances)
        return max(0.0, min(1.0, 1.0 - nearest))

    @staticmethod
    def get_price(s: str) -> float:
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "price" in data:
            try:
                return float(data["price"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid price response: {s}") from exc

        normalized = s.replace("$", "").replace(",", "")
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", normalized)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one price, got: {s}")
        return float(matches[0])

    def _estimate_price(self, prompt: str) -> float:
        try:
            result = parse_structured(
                self.client,
                model=self.MODEL,
                user_prompt=prompt,
                text_format=PriceEstimate,
            )
            return result.price
        except (AttributeError, ValueError):
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                seed=LLM_SEED,
            )
            usage.TRACKER.record(self.MODEL, getattr(response, "usage", None))
            return self.get_price(response.choices[0].message.content)

    def price(self, description: str) -> float:
        """Fair-value estimate as a plain float. Kept for callers (eval, tests) that only need
        the number; ``estimate_with_confidence`` returns the retrieval-confidence alongside it."""
        estimate, _ = self.estimate_with_confidence(description)
        return estimate

    def estimate_with_confidence(self, description: str) -> tuple[float, float]:
        """Estimate plus a retrieval confidence in [0, 1] (see ``_retrieval_confidence``)."""
        documents, prices, distances = self.find_similars(description)
        self.log(f"Calling {self.MODEL} with RAG context")
        prompt = (
            f'Estimate the price. Respond as JSON only: {{"price": {PLACEHOLDER_PRICE:.2f}}}\n\n'
            f"{description}\n\n{self.make_context(documents, prices)}"
        )
        result = self._estimate_price(prompt)
        if result <= 0:
            # A fair-value estimate is never <= 0. In practice this means the model echoed the
            # prompt's placeholder example instead of estimating -- its fallback when the RAG
            # context is uninformative. Fail loudly so the deal is skipped rather than surfacing
            # a fabricated value. The caller (estimate_value tool / eval) handles the error.
            raise ValueError(
                f"Pricer produced no usable estimate (got {result}); the RAG context was likely "
                f"uninformative for this product. Description: {description[:80]!r}"
            )
        confidence = self._retrieval_confidence(distances)
        self.log(f"Predicted ${result:.2f} (retrieval confidence {confidence:.2f})")
        return result, confidence


class PriceEstimate(BaseModel):
    price: float
