"""
MCP Server exposing deal-hunting tools for the autonomous agent.
Tools: scan_deals, estimate_value, notify_deal
"""
import os
import sys
import json
import logging

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.scanner_agent import ScannerAgent
from agents.frontier_agent import FrontierAgent
from agents.messaging_agent import MessagingAgent
from app.paths import DEFAULT_VECTORSTORE_PATH, ensure_data_dirs
from models.deals import Deal, Opportunity

mcp = FastMCP("deal-hunter", log_level="WARNING")

DB_PATH = os.getenv("PRODUCTS_VECTORSTORE_PATH") or str(DEFAULT_VECTORSTORE_PATH)
_AGENTS_CACHE = None


def _get_agents():
    global _AGENTS_CACHE
    if _AGENTS_CACHE is not None:
        return _AGENTS_CACHE

    import chromadb

    ensure_data_dirs()
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("products")
    _AGENTS_CACHE = {
        "scanner": ScannerAgent(),
        "estimator": FrontierAgent(collection),
        "messenger": MessagingAgent(),
    }
    return _AGENTS_CACHE


@mcp.tool()
def scan_deals(memory_json: str = "[]") -> str:
    """Scan RSS feeds for bargain deals. Returns JSON list of deals with product_description, price, url.
    memory_json: JSON array of previously surfaced opportunities (use deal URLs to avoid duplicates)."""
    agents = _get_agents()
    try:
        memory_data = json.loads(memory_json) if memory_json else []
        memory = [
            Opportunity(deal=Deal(**o["deal"]), estimate=o["estimate"])
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
    """Estimate the true market value of a product from its description (RAG + GPT-4o-mini)."""
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
    if estimated_true_value <= deal_price:
        raise ValueError(
            f"Estimated value (${estimated_true_value:.2f}) is not above "
            f"the deal price (${deal_price:.2f}) - this is not a compelling deal."
        )
    agents = _get_agents()
    agents["messenger"].notify(description, deal_price, estimated_true_value, url)
    return "Notification sent successfully"


if __name__ == "__main__":
    mcp.run(transport="stdio")
