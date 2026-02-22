import re
from typing import List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE
    MODEL = "gpt-4o-mini"

    def __init__(self, collection):
        self.log("Initializing")
        self.client = OpenAI()
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.log("Ready")

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

    def get_price(self, s: str) -> float:
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        documents, prices = self.find_similars(description)
        self.log(f"Calling {self.MODEL} with RAG context")
        messages = [
            {
                "role": "user",
                "content": f"Estimate the price. Respond with price only, no explanation.\n\n{description}\n\n{self.make_context(documents, prices)}",
            }
        ]
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            seed=42,
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Predicted ${result:.2f}")
        return result
