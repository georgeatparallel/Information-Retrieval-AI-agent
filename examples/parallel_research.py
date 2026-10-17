# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp==1.26.0", "httpx==0.28.1", "pydantic>=2.11,<3"]
# ///
"""Search the web anonymously and export source excerpts for IR-Agent ingestion."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation, TextContent
from pydantic import BaseModel

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "IR-Agent/1.0 (parallel-research-example)"


class SearchHit(BaseModel):
    url: str
    title: str | None = None
    excerpts: list[str]


class SearchResponse(BaseModel):
    results: list[SearchHit]


async def research(objective: str, queries: list[str], *, timeout: float = 60) -> str:
    """Execute one MCP search and render its results as an uploadable document.

    The deadline covers initialization, the tool call and transport cleanup. No
    credentials are loaded and no retries are made, including after rate limits.
    """
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                async with ClientSession(
                    read,
                    write,
                    client_info=Implementation(name="ir-agent-parallel-research", version="1.0"),
                ) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "web_search",
                        {
                            "objective": objective,
                            "search_queries": queries,
                            "session_id": str(uuid.uuid4()),
                        },
                    )
                    if result.isError:
                        message = "\n".join(
                            c.text for c in result.content if isinstance(c, TextContent)
                        )
                        raise RuntimeError(message or "Parallel returned a tool error.")
                    payload = result.structuredContent
                    if payload is None:
                        text = "\n".join(
                            c.text for c in result.content if isinstance(c, TextContent)
                        )
                        payload = json.loads(text)
                    response = SearchResponse.model_validate(payload)

    if not response.results:
        raise RuntimeError("No sources found. Refine the objective or search queries.")
    lines = [
        "# Web research",
        "",
        objective,
        "",
        f"Retrieved: {datetime.now(UTC).isoformat()}",
        "",
        "Source: Parallel Search MCP (search excerpts, not complete pages).",
        "",
    ]
    for index, hit in enumerate(response.results, 1):
        lines.extend([f"## {index}. {hit.title or hit.url}", "", f"URL: {hit.url}", ""])
        for excerpt in hit.excerpts:
            lines.extend([excerpt, ""])
    return "\n".join(lines)


def nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return value.strip()


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return seconds


def error_message(exc: BaseException) -> str:
    # The MCP transport uses task groups, which can wrap connection/tool errors.
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(error_message(child) for child in exc.exceptions)
    return str(exc) or type(exc).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objective", type=nonempty, help="The information to find.")
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        type=nonempty,
        help="Related keyword query; repeat for multiple search angles.",
    )
    parser.add_argument(
        "--output", type=Path, help="New Markdown file; existing files are preserved."
    )
    parser.add_argument(
        "--timeout",
        type=positive_seconds,
        default=60,
        help="Total request deadline in seconds (default: 60).",
    )
    args = parser.parse_args()
    # Check before making a network request, then use exclusive creation below.
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        document = asyncio.run(research(args.objective, args.query, timeout=args.timeout))
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(document)
        else:
            print(document)
    except Exception as exc:
        print(f"Search failed: {error_message(exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
