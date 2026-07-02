"""
Autonomous agent that uses MCP tools to hunt for deals.
Connects to MCP server via stdio, gets tools, runs OpenAI agentic loop.
"""
import asyncio
import json
import logging
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from infra.config import LLM_MAX_RETRIES, LLM_MODEL, LLM_SEED, LLM_TEMPERATURE
from core.scoring import best_opportunity
from core.source_ids import deal_id
from infra.paths import DEFAULT_VECTORSTORE_PATH, PROJECT_ROOT
from infra import usage
from domain.deal import Deal, Opportunity

load_dotenv(override=True)

MODEL = LLM_MODEL
MAX_AGENT_STEPS = 8
_ESTIMATE_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
SYSTEM_MSG = "You find great deals using your tools and notify the user of the best bargain."
USER_MSG = """First, scan for bargain deals. Then for each deal, estimate its true value,
passing that deal's url to the estimate tool along with its description.
Pick the single most compelling deal (price much lower than estimated value) and notify the user.
Then reply OK to indicate success."""


def _mcp_server_params() -> StdioServerParameters:
    env = dict(os.environ)
    env.setdefault("PRODUCTS_VECTORSTORE_PATH", str(DEFAULT_VECTORSTORE_PATH))
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(PROJECT_ROOT)
    )
    return StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "app" / "mcp_server.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def mcp_tool_to_openai(mcp_tool) -> dict:
    schema = getattr(mcp_tool, "inputSchema", None) or {}
    if "properties" not in schema:
        schema = {"type": "object", "properties": schema.get("properties", {}), "required": schema.get("required", [])}
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": getattr(mcp_tool, "description", "") or "",
            "parameters": schema,
        },
    }


def _parse_estimate(result_text: str) -> float | None:
    match = _ESTIMATE_RE.search(result_text or "")
    return float(match.group(1).replace(",", "")) if match else None


def candidate_from_estimate(
    description: str, estimate: float, scanned_deals_by_url: dict, url: str | None = None
) -> Opportunity | None:
    """Pair an estimate_value result back to the scanned deal it was run on.

    The agent estimates each scanned deal; we re-key those estimates to the deal's price
    and list_price so a deterministic ranking (not the model) can pick the winner. Pairing
    prefers the deal's ``url`` (matched by stable product id, robust to the model echoing a
    different slug/query), which the estimate_value tool now asks the model to pass. Exact
    product_description matching is kept only as a fallback for the pre-url calling convention;
    if neither pairs, the candidate is skipped (falling back to the model's own notify choice).
    """
    deal = None
    if url:
        deal = scanned_deals_by_url.get(url)
        if deal is None:
            target_id = deal_id(url)
            deal = next(
                (d for d in scanned_deals_by_url.values() if deal_id(d.get("url", "")) == target_id),
                None,
            )
    if deal is None:
        deal = next(
            (d for d in scanned_deals_by_url.values() if d.get("product_description") == description),
            None,
        )
    if deal is None:
        return None
    return Opportunity(
        deal=Deal(
            product_description=deal["product_description"],
            price=deal["price"],
            list_price=deal.get("list_price"),
            url=deal["url"],
            quantity=deal.get("quantity", 1),
        ),
        estimate=estimate,
    )


def opportunity_from_notify_args(args: dict, scanned_deals_by_url: dict) -> Opportunity:
    scanned_deal = scanned_deals_by_url.get(args["url"])
    if scanned_deal is None:
        # The agent may notify with a URL whose slug/query differs from the scanned one; match
        # on the stable product id so the scraped list_price isn't lost in the hand-off.
        target_id = deal_id(args["url"])
        scanned_deal = next(
            (d for d in scanned_deals_by_url.values() if deal_id(d.get("url", "")) == target_id),
            {},
        )
    return Opportunity(
        deal=Deal(
            product_description=args["description"],
            price=args["deal_price"],
            list_price=scanned_deal.get("list_price"),
            url=args["url"],
            quantity=scanned_deal.get("quantity", 1),
        ),
        estimate=args["estimated_true_value"],
    )


async def run_agent(memory: list) -> tuple[list, Opportunity | None]:
    server_params = _mcp_server_params()

    memory_data = [o.model_dump() for o in memory]
    opportunity = None
    scanned_deals_by_url = {}
    candidates: dict[str, Opportunity] = {}

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            # get_run_usage is operator telemetry, not a deal-hunting capability: keep it out
            # of the tool list shown to the LLM so it never appears in model context, then call
            # it directly (below) once the agent loop is done.
            openai_tools = [
                mcp_tool_to_openai(t) for t in tools_result.tools if t.name != "get_run_usage"
            ]

            client = OpenAI(max_retries=LLM_MAX_RETRIES)
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_MSG},
            ]

            for _ in range(MAX_AGENT_STEPS):
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=openai_tools,
                    temperature=LLM_TEMPERATURE,
                    seed=LLM_SEED,
                )
                usage.TRACKER.record(MODEL, getattr(response, "usage", None))
                msg = response.choices[0].message

                if response.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                                }
                                for tc in msg.tool_calls
                            ],
                        }
                    )
                    for tc in msg.tool_calls:
                        name = tc.function.name
                        args = json.loads(tc.function.arguments or "{}")
                        if name == "scan_deals":
                            args["memory_json"] = json.dumps(memory_data)
                        result = await session.call_tool(name, args)
                        content = ""
                        if result.content:
                            for block in result.content:
                                if hasattr(block, "text"):
                                    content += block.text
                        if name == "scan_deals" and not result.isError:
                            try:
                                scanned_deals_by_url = {
                                    deal["url"]: deal for deal in json.loads(content or "[]")
                                }
                            except (json.JSONDecodeError, TypeError):
                                scanned_deals_by_url = {}
                        if name == "estimate_value" and not result.isError:
                            estimate = _parse_estimate(content)
                            if estimate is not None:
                                candidate = candidate_from_estimate(
                                    args.get("description", ""),
                                    estimate,
                                    scanned_deals_by_url,
                                    args.get("url"),
                                )
                                if candidate is not None:
                                    candidates[deal_id(candidate.deal.url)] = candidate
                        if name == "notify_deal" and not result.isError:
                            opportunity = opportunity_from_notify_args(args, scanned_deals_by_url)
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": content}
                        )
                else:
                    break
            else:
                raise RuntimeError(f"Agent exceeded {MAX_AGENT_STEPS} tool-call steps")

            # Pull the server subprocess's accumulated token usage into this process's
            # tracker. The work agents (scanner/pricer/messenger) execute inside the MCP
            # server process and record to a *separate* usage.TRACKER there; without this
            # merge the orchestrator's cost report would miss the bulk of the spend (only the
            # orchestration loop above records into this process). Best-effort: a failure here
            # must not sink an otherwise successful run.
            try:
                usage_result = await session.call_tool("get_run_usage", {})
                usage_text = "".join(
                    block.text
                    for block in (usage_result.content or [])
                    if hasattr(block, "text")
                )
                server_usage = json.loads(usage_text or "{}")
                usage.TRACKER.merge(
                    prompt_tokens=server_usage.get("prompt_tokens", 0),
                    completion_tokens=server_usage.get("completion_tokens", 0),
                    calls=server_usage.get("calls", 0),
                    unpriced_models=server_usage.get("unpriced_models", ()),
                )
            except Exception as exc:
                logging.warning("Could not collect MCP server usage: %s", exc)

    # Deterministic selection: the model gathers candidates, but the single best deal is
    # chosen by a reproducible max over the list-price-capped discount, not model judgment.
    # Falls back to the model's notify choice only when no estimate could be paired to a deal.
    deterministic = best_opportunity(candidates.values())
    if deterministic is not None:
        if opportunity is not None and deal_id(opportunity.deal.url) != deal_id(deterministic.deal.url):
            logging.info(
                "Overriding agent's notify pick (discount=$%.2f) with deterministic best "
                "(discount=$%.2f)",
                opportunity.discount,
                deterministic.discount,
            )
        opportunity = deterministic

    return memory, opportunity


def run_sync(memory: list) -> tuple[list, Opportunity | None]:
    return asyncio.run(run_agent(memory))
