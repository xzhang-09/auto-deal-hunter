import json
import re
from typing import List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.agent import Agent
from app.config import EMBEDDING_MODEL, LLM_MODEL, LLM_SEED, LLM_TEMPERATURE
from app import usage


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE
    MODEL = LLM_MODEL

    def __init__(self, collection):
        self.log("Initializing")
        self.client = OpenAI()
        self.collection = collection
        self._check_embedding_model(collection)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.log("Ready")

    def _check_embedding_model(self, collection) -> None:
        """Refuse to query a store built with a different embedder.

        The build script stamps the embedding model name into the collection metadata.
        Querying with a different model produces vectors in an incompatible space and
        silently returns nonsense neighbors, so fail loudly instead. Stores built before
        this stamp existed have no metadata and are allowed (with a warning)."""
        metadata = getattr(collection, "metadata", None) or {}
        built_with = metadata.get("embedding_model")
        if built_with is None:
            self.log("Vector store has no embedding_model metadata; assuming it matches config")
        elif built_with != EMBEDDING_MODEL:
            raise ValueError(
                f"Vector store was built with embedding model '{built_with}' but config "
                f"EMBEDDING_MODEL is '{EMBEDDING_MODEL}'. Rebuild the store or fix the config."
            )

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        msg = "Context - similar products:\n\n"
        for similar, price in zip(similars, prices):
            msg += f"Product: {similar}\nPrice: ${price:.2f}\n\n"
        return msg

    def find_similars(self, description: str):
        self.log("RAG search for similar products")
        vector = self.model.encode([description])
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(), n_results=5
        )
        documents = results["documents"][0][:]
        prices = [m["price"] for m in results["metadatas"][0][:]]
        return documents, prices

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

    def price(self, description: str) -> float:
        documents, prices = self.find_similars(description)
        self.log(f"Calling {self.MODEL} with RAG context")
        messages = [
            {
                "role": "user",
                "content": f'Estimate the price. Respond as JSON only: {{"price": 123.45}}\n\n{description}\n\n{self.make_context(documents, prices)}',
            }
        ]
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            seed=LLM_SEED,
        )
        usage.TRACKER.record(self.MODEL, getattr(response, "usage", None))
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Predicted ${result:.2f}")
        return result
