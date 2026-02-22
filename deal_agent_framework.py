import os
import sys
import json
import logging
from typing import List
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from sklearn.manifold import TSNE
import numpy as np

from agents.deals import Opportunity
from agent_mcp import run_sync

load_dotenv(override=True)

BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    DB = os.getenv("PRODUCTS_VECTORSTORE_PATH", "products_vectorstore")
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("products")

    def read_memory(self) -> List[Opportunity]:
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as f:
                data = json.load(f)
            return [Opportunity(**item) for item in data]
        return []

    def write_memory(self) -> None:
        data = [o.model_dump() for o in self.memory]
        with open(self.MEMORY_FILENAME, "w") as f:
            json.dump(data, f, indent=2)

    def log(self, message: str):
        logging.info(BG_BLUE + WHITE + "[Framework] " + message + RESET)

    def run(self):
        self.log("Starting MCP-based agent")
        _, opportunity = run_sync(self.memory)
        if opportunity:
            self.memory.append(opportunity)
            self.write_memory()
        self.log("Run complete")
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("products")
        result = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=max_datapoints
        )
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [m["category"] for m in result["metadatas"]]
        colors = [COLORS[CATEGORIES.index(c)] if c in CATEGORIES else "gray" for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced = tsne.fit_transform(vectors)
        return documents, reduced, colors
