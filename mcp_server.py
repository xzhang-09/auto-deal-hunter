"""
MCP Server exposing deal-hunting tools for the autonomous agent.
Tools: scan_deals, estimate_value, notify_deal
"""
import os
import sys
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

sys.path.insert(0, str(Path(__file__).parent))
from agents.scanner_agent import ScannerAgent
from agents.frontier_agent import FrontierAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.deals import Deal, Opportunity

mcp = FastMCP("deal-hunter", log_level="WARNING")

DB_PATH = os.getenv("PRODUCTS_VECTORSTORE_PATH") or str(Path(__file__).parent / "products_vectorstore")
USE_SPECIALIST = os.getenv("USE_SPECIALIST", "false").lower() == "true"


def _get_agents():
    import chromadb

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("products")
    if USE_SPECIALIST:
        estimator = EnsembleAgent(collection)
    else:
        estimator = FrontierAgent(collection)
    return {
        "scanner": ScannerAgent(),
        "estimator": estimator,
        "messenger": MessagingAgent(),
    }


@mcp.tool()
def scan_deals(memory_json: str = "[]") -> str:
    """Scan RSS feeds for bargain deals. Returns JSON list of deals with product_description, price, url.
    memory_json: JSON array of previously surfaced opportunities (use deal URLs to avoid duplicates)."""
    agents = _get_agents()
    try:
        memory_data = json.loads(memory_json) if memory_json else []
        memory = [
            Opportunity(deal=Deal(**o["deal"]), estimate=o["estimate"], discount=o["discount"])
            for o in memory_data
        ]
    except Exception:
        memory = []
    selection = agents["scanner"].scan(memory=memory)
    if selection:
        return json.dumps([d.model_dump() for d in selection.deals])
    return "[]"


@mcp.tool()
def estimate_value(description: str) -> str:
    """Estimate the true market value of a product from its description (RAG + optional fine-tuned Specialist)."""
    agents = _get_agents()
    estimate = agents["estimator"].price(description)
    return f"The estimated true value of this product is ${estimate:.2f}"


@mcp.tool()
def notify_deal(
    description: str,
    deal_price: float,
    estimated_true_value: float,
    url: str,
) -> str:
    """Send a push notification about a compelling deal. Call once per run for the best deal."""
    agents = _get_agents()
    agents["messenger"].notify(description, deal_price, estimated_true_value, url)
    return "Notification sent successfully"


if __name__ == "__main__":
    mcp.run(transport="stdio")
