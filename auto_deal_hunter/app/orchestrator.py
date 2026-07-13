import os
import sys
import logging
import threading
from collections import Counter
from typing import List

from dotenv import load_dotenv
import chromadb
from sklearn.manifold import TSNE
import numpy as np

from auto_deal_hunter.app.pipeline import DealPipeline
from auto_deal_hunter.infra import usage
from auto_deal_hunter.infra.config import (
    DEALS_TTL_HOURS,
    TELEGRAM_FEEDBACK_ENABLED,
    TELEGRAM_POLL_TIMEOUT_SECONDS,
)
from auto_deal_hunter.infra.log_utils import BG_BLUE, RESET, WHITE
from auto_deal_hunter.core.opportunity_store import OpportunityStore
from auto_deal_hunter.infra.paths import DEFAULT_DEALS_DB_PATH, DEFAULT_MEMORY_PATH, DEFAULT_VECTORSTORE_PATH, ensure_data_dirs
from auto_deal_hunter.domain.deal import Opportunity

load_dotenv(override=True)

# Scan path. "direct" (default) runs the pipeline in-process (app.pipeline); "agent" drives the
# MCP tool server through an LLM tool-calling loop (app.mcp_client) -- kept as a demo of MCP
# orchestration. Both produce the same deterministically-selected best deal.
SCAN_MODE = os.getenv("SCAN_MODE", "direct").lower()

# Palette for the reference-map legend. Categories are derived from the data at plot time
# (see get_plot_data), so this only needs enough distinct colors; its length caps how many
# categories get their own color before the rest fall into "Other".
PLOT_PALETTE = ["red", "blue", "brown", "orange", "green", "purple", "cyan", "magenta"]


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


class Orchestrator:
    DB = os.getenv("PRODUCTS_VECTORSTORE_PATH", str(DEFAULT_VECTORSTORE_PATH))
    MEMORY_FILENAME = os.getenv("MEMORY_FILENAME", str(DEFAULT_MEMORY_PATH))
    DEALS_DB_PATH = os.getenv("DEALS_DB_PATH", str(DEFAULT_DEALS_DB_PATH))

    def __init__(self):
        # Guards run(): the Gradio timer, the "Scan now" button, and every open browser session
        # all trigger run() against this shared instance. Concurrent runs would double the LLM
        # spend, reset each other's usage tracker mid-run, and re-notify the same deals. A single
        # scan at a time is the correct semantics, so overlapping calls skip rather than queue.
        self._run_lock = threading.Lock()
        ensure_data_dirs()
        init_logging()
        client = chromadb.PersistentClient(path=self.DB)
        self.opportunity_store = OpportunityStore(self.DEALS_DB_PATH)
        self.opportunity_store.migrate_from_json(self.MEMORY_FILENAME)
        self._prune_stale()
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("products")
        self._pipeline = DealPipeline(self.collection)
        self._telegram_feedback = None
        if TELEGRAM_FEEDBACK_ENABLED:
            token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            if token and chat_id:
                from auto_deal_hunter.infra.telegram_feedback import TelegramFeedbackPoller

                self._telegram_feedback = TelegramFeedbackPoller(
                    token,
                    chat_id,
                    self.opportunity_store,
                    TELEGRAM_POLL_TIMEOUT_SECONDS,
                )
                self._telegram_feedback.start()

    def read_memory(self) -> List[Opportunity]:
        store = getattr(self, "opportunity_store", OpportunityStore(self.DEALS_DB_PATH))
        return store.list_opportunities()

    def log(self, message: str):
        logging.info(BG_BLUE + WHITE + "[Framework] " + message + RESET)

    def _prune_stale(self) -> None:
        removed = self.opportunity_store.prune_stale(DEALS_TTL_HOURS)
        if removed:
            self.log(f"Pruned {removed} stale opportunit{'y' if removed == 1 else 'ies'} (TTL {DEALS_TTL_HOURS}h)")

    def run(self):
        # Non-blocking: if a scan is already running, skip this trigger instead of queueing a
        # duplicate. Returning the current memory keeps the UI responsive and the table correct.
        if not self._run_lock.acquire(blocking=False):
            self.log("Scan already in progress; skipping this trigger")
            return self.memory
        try:
            usage.TRACKER.reset()
            if SCAN_MODE == "agent":
                self.log("Starting MCP agent loop (SCAN_MODE=agent)")
                from auto_deal_hunter.app.mcp_client import run_sync

                _, opportunity = run_sync(self.memory)
            else:
                self.log("Starting in-process scan pipeline")
                _, opportunity = self._pipeline.run(self.memory)
            if opportunity:
                self.memory.append(opportunity)
                self.opportunity_store.append(opportunity)
            self._prune_stale()
            self.memory = self.read_memory()
            self.log(usage.TRACKER.report())
            self.log("Run complete")
            return self.memory
        finally:
            self._run_lock.release()

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
        # Data-driven legend: the most common categories in the store each get a palette color;
        # the rest fall into "Other". Adapts to whatever MCAULEY_CATEGORY the store was built
        # on, so the map needs no manual edits when the product category changes.
        top = [c for c, _ in Counter(categories).most_common(len(PLOT_PALETTE))]
        color_of = {c: PLOT_PALETTE[i] for i, c in enumerate(top)}
        colors = [color_of.get(c, "gray") for c in categories]
        labels = [c if c in color_of else "Other" for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced = tsne.fit_transform(vectors)
        return documents, reduced, colors, labels
