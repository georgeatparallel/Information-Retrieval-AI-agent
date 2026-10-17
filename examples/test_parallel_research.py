"""Exercise the example through the real MCP SDK and HTTP transport, offline."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import parallel_research as example


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.requests: list[httpx.Request] = []
        self.clients: list[httpx.AsyncClient] = []
        self.tool_result = {
            "content": [],
            "structuredContent": {
                "results": [
                    {
                        "title": "Fusion paper",
                        "url": "https://example.org/rrf",
                        "excerpts": ["Combine rankings using reciprocal ranks."],
                    }
                ],
            },
        }
        self.tool_started = asyncio.Event()
        self.wait_forever = False
        self.http_error = False
        original_client = httpx.AsyncClient

        async def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.method != "POST":
                return httpx.Response(405)
            body = json.loads(request.content)
            method = body["method"]
            if method.startswith("notifications/"):
                return httpx.Response(202)
            if method == "initialize":
                result = {
                    "protocolVersion": body["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
            elif method == "tools/call":
                self.tool_started.set()
                if self.wait_forever:
                    await asyncio.Event().wait()
                if self.http_error:
                    return httpx.Response(503)
                result = self.tool_result
            else:
                raise AssertionError(f"Unexpected method: {method}")
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

        def client_factory(**kwargs: object) -> httpx.AsyncClient:
            client = original_client(transport=httpx.MockTransport(handle), **kwargs)
            self.clients.append(client)
            return client

        self.client_patch = patch.object(example.httpx, "AsyncClient", side_effect=client_factory)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    async def test_search_request_and_uploadable_document(self) -> None:
        # Saved/environment credentials are intentionally irrelevant to this caller.
        with patch.dict("os.environ", {"PARALLEL_API_KEY": "unused", "OPENAI_API_KEY": "unused"}):
            document = await example.research("Explain RRF", ["reciprocal rank fusion"])
        self.assertIn("Fusion paper", document)
        self.assertIn("URL: https://example.org/rrf", document)
        self.assertIn("Combine rankings using reciprocal ranks.", document)
        self.assertIn("Retrieved:", document)
        calls = [json.loads(r.content) for r in self.requests if r.method == "POST"]
        call = next(c for c in calls if c["method"] == "tools/call")
        self.assertEqual(call["params"]["name"], "web_search")
        arguments = call["params"]["arguments"]
        self.assertEqual(arguments["objective"], "Explain RRF")
        self.assertEqual(arguments["search_queries"], ["reciprocal rank fusion"])
        self.assertTrue(arguments["session_id"])
        self.assertIn("notifications/initialized", [c["method"] for c in calls])
        for request in self.requests:
            self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
            self.assertEqual(
                request.headers["User-Agent"], "IR-Agent/1.0 (parallel-research-example)"
            )
            self.assertNotIn("Authorization", request.headers)
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_text_json_response(self) -> None:
        payload = self.tool_result.pop("structuredContent")
        self.tool_result["content"] = [{"type": "text", "text": json.dumps(payload)}]
        self.assertIn("Fusion paper", await example.research("RRF", ["rank fusion"]))

    async def test_tool_error_is_not_a_document(self) -> None:
        self.tool_result = {
            "isError": True,
            "content": [{"type": "text", "text": "Quota exceeded"}],
        }
        with self.assertRaises(ExceptionGroup) as raised:
            await example.research("RRF", ["rank fusion"])
        self.assertIn("Quota exceeded", example.error_message(raised.exception))
        self.assertEqual(sum(b'"tools/call"' in r.content for r in self.requests), 1)

    async def test_invalid_response_and_empty_results(self) -> None:
        for payload in ({"results": "invalid"}, {"results": []}):
            with self.subTest(payload=payload):
                self.tool_result["structuredContent"] = payload
                with self.assertRaises((ExceptionGroup, RuntimeError)):
                    await example.research("RRF", ["rank fusion"])

    async def test_http_error(self) -> None:
        self.http_error = True
        with self.assertRaises(ExceptionGroup) as raised:
            await example.research("RRF", ["rank fusion"])
        self.assertIn("503", example.error_message(raised.exception))
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_deadline_closes_transport(self) -> None:
        self.wait_forever = True
        with self.assertRaises(TimeoutError):
            await example.research("RRF", ["rank fusion"], timeout=0.1)
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_cancellation_closes_transport(self) -> None:
        self.wait_forever = True
        task = asyncio.create_task(example.research("RRF", ["rank fusion"]))
        await asyncio.wait_for(self.tool_started.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(all(client.is_closed for client in self.clients))


class CommandTests(unittest.TestCase):
    def test_file_output_and_existing_file_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "research.md"
            argv = [
                "parallel_research.py",
                "RRF",
                "--query",
                "rank fusion",
                "--output",
                str(output),
            ]
            with (
                patch("sys.argv", argv),
                patch.object(example, "research", new_callable=AsyncMock) as search,
            ):
                search.return_value = "# Sources\nURL: https://example.org\n"
                self.assertEqual(example.main(), 0)
                self.assertEqual(output.read_text(), search.return_value)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    example.main()
                self.assertEqual(search.await_count, 1)
                self.assertEqual(output.read_text(), search.return_value)

    def test_failed_search_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "research.md"
            argv = [
                "parallel_research.py",
                "RRF",
                "--query",
                "rank fusion",
                "--output",
                str(output),
            ]
            stderr = io.StringIO()
            with (
                patch("sys.argv", argv),
                patch.object(example, "research", new_callable=AsyncMock) as search,
            ):
                search.side_effect = RuntimeError("Quota exceeded")
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(example.main(), 1)
            self.assertFalse(output.exists())
            self.assertIn("Quota exceeded", stderr.getvalue())

    def test_invalid_arguments(self) -> None:
        for args in (
            [" ", "--query", "rrf"],
            ["rrf", "--query", " "],
            ["rrf", "--query", "rrf", "--timeout", "nan"],
            ["rrf", "--query", "rrf", "--timeout", "0"],
        ):
            with (
                self.subTest(args=args),
                patch("sys.argv", ["example", *args]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                example.main()


if __name__ == "__main__":
    unittest.main()
