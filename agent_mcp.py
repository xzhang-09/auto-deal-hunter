"""
Autonomous agent that uses MCP tools to hunt for deals.
Connects to MCP server via stdio, gets tools, runs OpenAI agentic loop.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.deals import Deal, Opportunity

load_dotenv(override=True)

MODEL = "gpt-4o-mini"
SYSTEM_MSG = "You find great deals using your tools and notify the user of the best bargain."
USER_MSG = """First, scan for bargain deals. Then for each deal, estimate its true value.
Pick the single most compelling deal (price much lower than estimated value) and notify the user.
Then reply OK to indicate success."""


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


async def run_agent(memory: list) -> tuple[list, Opportunity | None]:
    project_dir = Path(__file__).parent
    env = dict(os.environ)
    env.setdefault("PRODUCTS_VECTORSTORE_PATH", str(project_dir / "products_vectorstore"))
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(project_dir / "mcp_server.py")],
        cwd=str(project_dir),
        env=env,
    )

    memory_data = [o.model_dump() for o in memory]
    opportunity = None

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            openai_tools = [mcp_tool_to_openai(t) for t in tools_result.tools]

            client = OpenAI()
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_MSG},
            ]

            done = False
            while not done:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=openai_tools,
                )
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
                        if name == "notify_deal" and not result.isError:
                            opportunity = Opportunity(
                                deal=Deal(
                                    product_description=args["description"],
                                    price=args["deal_price"],
                                    url=args["url"],
                                ),
                                estimate=args["estimated_true_value"],
                                discount=args["estimated_true_value"] - args["deal_price"],
                            )
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": content}
                        )
                else:
                    done = True

    return memory, opportunity


def run_sync(memory: list) -> tuple[list, Opportunity | None]:
    return asyncio.run(run_agent(memory))
