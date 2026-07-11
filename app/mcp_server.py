"""
MCP Server exposing deal-hunting tools for the autonomous agent.
Tools: scan_deals, estimate_value, notify_deal
"""
import os
import sys
import json
import logging

from mcp.server.fastmcp import FastMCP

from agents.scanner_agent import ScannerAgent
from agents.pricer_agent import PricerAgent
from agents.messaging_agent import MessagingAgent
from core.source_ids import deal_id
from infra.config import RAG_MIN_CONFIDENCE
from infra.paths import DEFAULT_VECTORSTORE_PATH, ensure_data_dirs
from infra import usage
from domain.deal import Opportunity

# .env is loaded transitively via infra.config (imported by the agents above).
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("deal-hunter", log_level="WARNING")

DB_PATH = os.getenv("PRODUCTS_VECTORSTORE_PATH") or str(DEFAULT_VECTORSTORE_PATH)
_AGENTS_CACHE = None

# Retrieval confidence recorded by estimate_value, keyed by deal_id, read by notify_deal to
# withhold a push for weakly-supported estimates. Naturally per-run: the server runs in a fresh
# subprocess each scan, so this starts empty every run.
_CONFIDENCE_BY_ID: dict[str, float] = {}


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
        "estimator": PricerAgent(collection),
        "messenger": MessagingAgent(),
    }
    return _AGENTS_CACHE


@mcp.tool()
def scan_deals(memory_json: str = "[]") -> str:
    """Scan RSS feeds for bargain deals. Returns JSON list of deals with product_description, price, url.
    memory_json: JSON array of already surfaced opportunities (use deal URLs to avoid duplicates)."""
    agents = _get_agents()
    try:
        memory_data = json.loads(memory_json) if memory_json else []
        memory = [Opportunity(**o) for o in memory_data]
    except Exception as exc:
        # Falling back to an empty memory silently disables dedup: every already-surfaced deal
        # would be re-estimated (cost) and re-notified (spam) with no trace. Log loudly so a
        # payload/schema drift that breaks deserialization is diagnosable instead of invisible.
        logging.warning("scan_deals could not parse memory_json (%s); dedup disabled this run", exc)
        memory = []
    selection = agents["scanner"].scan(memory=memory)
    if selection:
        return json.dumps([d.model_dump() for d in selection.deals])
    return "[]"


@mcp.tool()
def estimate_value(description: str, url: str = "") -> str:
    """Estimate the true market value of a product from its description (RAG + GPT-4o-mini).
    Pass the deal's `url` (from scan_deals) so the estimate can be paired back to its exact
    deal for deterministic ranking; the url does not affect the estimate itself."""
    agents = _get_agents()
    estimate, confidence = agents["estimator"].estimate_with_confidence(description)
    if url:
        _CONFIDENCE_BY_ID[deal_id(url)] = confidence
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
    # Withhold the push when the estimate rests on a weak RAG match: a low-confidence estimate
    # is the main source of false bargains. The deal is still saved by the orchestrator; only
    # the notification is suppressed. Unknown confidence (deal never estimated) is not gated.
    confidence = _CONFIDENCE_BY_ID.get(deal_id(url))
    if confidence is not None and confidence < RAG_MIN_CONFIDENCE:
        logging.info(
            "Withholding push: estimate confidence %.2f < threshold %.2f for %s",
            confidence, RAG_MIN_CONFIDENCE, url,
        )
        return (
            f"Push withheld: estimate confidence {confidence:.2f} is below the "
            f"{RAG_MIN_CONFIDENCE:.2f} threshold. The deal is still saved."
        )
    agents = _get_agents()
    agents["messenger"].notify(description, deal_price, estimated_true_value, url)
    return "Notification sent successfully"


@mcp.tool()
def get_run_usage() -> str:
    """Operator telemetry (NOT model-facing): this server process's accumulated LLM token
    usage, as JSON. The deal-hunting agents run here, inside a subprocess separate from the
    orchestrator, and record to this process's own usage tracker. The client pulls these
    totals back to aggregate a complete per-run cost; it hides this tool from the LLM and
    calls it directly after the agent loop, so the counts never enter the model's context."""
    return json.dumps(
        {
            "prompt_tokens": usage.TRACKER.prompt_tokens,
            "completion_tokens": usage.TRACKER.completion_tokens,
            "calls": usage.TRACKER.calls,
            "unpriced_models": usage.TRACKER.unpriced_models,
        }
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
