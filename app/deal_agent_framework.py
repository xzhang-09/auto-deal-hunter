import os
import sys
import logging
from typing import List

from dotenv import load_dotenv
import chromadb
from sklearn.manifold import TSNE
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent_mcp import run_sync
from app import usage
from app.opportunity_store import OpportunityStore
from app.paths import DEFAULT_DEALS_DB_PATH, DEFAULT_MEMORY_PATH, DEFAULT_VECTORSTORE_PATH, ensure_data_dirs
from models.deals import Opportunity

load_dotenv(override=True)

BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

CATEGORIES = [
    "All Electronics",
    "Computers",
    "Camera & Photo",
    "Cell Phones & Accessories",
    "Home Audio & Theater",
    "Industrial & Scientific",
    "Tools & Home Improvement",
    "Car Electronics",
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
    DB = os.getenv("PRODUCTS_VECTORSTORE_PATH", str(DEFAULT_VECTORSTORE_PATH))
    MEMORY_FILENAME = os.getenv("MEMORY_FILENAME", str(DEFAULT_MEMORY_PATH))
    DEALS_DB_PATH = os.getenv("DEALS_DB_PATH", str(DEFAULT_DEALS_DB_PATH))

    def __init__(self):
        ensure_data_dirs()
        init_logging()
        client = chromadb.PersistentClient(path=self.DB)
        self.opportunity_store = OpportunityStore(self.DEALS_DB_PATH)
        self.opportunity_store.migrate_from_json(self.MEMORY_FILENAME)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("products")

    def read_memory(self) -> List[Opportunity]:
        store = getattr(self, "opportunity_store", OpportunityStore(self.DEALS_DB_PATH))
        return store.list_opportunities()

    def write_memory(self) -> None:
        store = getattr(self, "opportunity_store", OpportunityStore(self.DEALS_DB_PATH))
        store.replace_all(self.memory)

    def log(self, message: str):
        logging.info(BG_BLUE + WHITE + "[Framework] " + message + RESET)

    def run(self):
        self.log("Starting MCP-based agent")
        usage.TRACKER.reset()
        _, opportunity = run_sync(self.memory)
        if opportunity:
            self.memory.append(opportunity)
            self.opportunity_store.append(opportunity)
        self.log(usage.TRACKER.report())
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
        labels = [c if c in CATEGORIES else "Other" for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced = tsne.fit_transform(vectors)
        return documents, reduced, colors, labels
