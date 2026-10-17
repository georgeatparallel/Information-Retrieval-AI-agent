# Web research for the document corpus

Use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
to find current web sources and save their titles, URLs and search excerpts as
Markdown. Review the file, then upload it into IR-Agent to make the material
available to document search and chat.

This is a standalone command-line example. It does not register a tool in the
running agent or change its DuckDuckGo web search. Anonymous search uses Fast
mode and needs no account or API key. The free endpoint is intended for light
use and has rate limits.

## Run a search

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use
Python 3.11 or later. From the repository root:

```bash
uv run --script examples/parallel_research.py \
  "Find sources explaining reciprocal rank fusion for hybrid retrieval" \
  --query "reciprocal rank fusion hybrid retrieval" \
  --query "reciprocal rank fusion original paper" \
  --output web-research.md
```

uv installs the dependencies declared in the script into a separate environment.
You do not need to install the backend, download embedding models, configure an
LLM or add credentials. The script does not read `.env` or API keys from the
environment. The objective and queries are sent to the public search service;
use public topics rather than private corpus contents.

The example calls `web_search` over Streamable HTTP at
`https://search.parallel.ai/mcp`. Use a focused objective and related keyword
queries, ideally 3–6 words each. Repeat `--query` for different angles on the
same objective. Omit `--output` to print the Markdown to standard output. An
existing output file is never overwritten.

The document includes the retrieval time and numbered sources, each with its
title, a plain-text `URL:` line and excerpts. These are search excerpts, not full
pages or a generated answer. Review source relevance and accuracy before upload.

## Add the results to IR-Agent

After a successful search, start IR-Agent using the root README's quick start.
Upload `web-research.md` through the sidebar, or use its existing upload endpoint:

```bash
curl --fail-with-body -X POST http://localhost:8000/api/v1/documents/upload \
  -F 'file=@web-research.md;type=text/markdown' \
  -F 'title=Web research: reciprocal rank fusion'
```

If the backend has `API_KEY` enabled, add your `X-API-Key` header. Ask chat about
the uploaded research or use `/api/v1/search` to retrieve it. The URLs remain in
the document text; IR-Agent's citations identify the uploaded document, not
separate indexed web pages. Rerun the example to a new file when you need fresher
sources, and remove obsolete uploads through the usual document-delete flow.

## Failures and development

Network, protocol and tool errors exit with status 1 and a message on stderr.
No output document is created when search fails or returns no sources. The
default overall deadline is 60 seconds; set `--timeout 120` for a slower request.
Ctrl-C cancels the operation. Rate-limit errors are reported without automatic
retries; wait for the indicated reset before trying again.

Run the isolated example tests without the backend dependencies:

```bash
uv run --with mcp==1.26.0 --with httpx==0.28.1 \
  python -m unittest discover -s examples -p 'test_*.py' -v
```
