"""Knowledge MCP server - the help center as an MCP service.

Why expose RAG over MCP when the agents call the retriever in-process? Because the
same knowledge base becomes instantly usable by *any* MCP client - desktop assistants,
Cursor, VS Code, an internal Slack bot or another team's agent - with zero
integration code. This server shows all three MCP primitives:

* **tools**     - ``search_knowledge_base`` (hybrid search + rerank), ``get_article``
* **resources** - ``kb://index`` and the ``kb://articles/{article_id}`` template
* **prompts**   - ``grounded_answer``: a reusable, citation-enforcing prompt

It is public and read-only (``readOnlyHint``), so it needs no per-user auth.
Run standalone (``samadhan serve knowledge``) or mounted inside the API at ``/mcp/knowledge``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from samadhan.config import Settings, get_settings
from samadhan.rag.documents import KBDocument, load_knowledge_base
from samadhan.rag.retriever import HybridRetriever

RetrieverProvider = Callable[[], Awaitable[HybridRetriever]]


class KBHit(BaseModel):
    article_id: str
    title: str
    section: str
    url: str
    snippet: str
    relevance: float | None = Field(default=None, description="Cross-encoder relevance probability (0-1)")


def create_knowledge_server(get_retriever: RetrieverProvider, settings: Settings | None = None) -> FastMCP:
    settings = settings or get_settings()
    docs_cache: dict[str, KBDocument] = {}

    def _docs() -> dict[str, KBDocument]:
        if not docs_cache:
            docs_cache.update({d.doc_id: d for d in load_knowledge_base(settings.retrieval.kb_dir)})
        return docs_cache

    mcp = FastMCP(
        name="voltwise-knowledge",
        instructions="Search the Voltwise help center. Cite article titles and URLs in answers.",
        mask_error_details=True,
    )
    read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

    @mcp.tool(annotations=read_only)
    async def search_knowledge_base(query: str, category: str | None = None, top_k: int = 5) -> list[KBHit]:
        """Hybrid (semantic + keyword) search over the Voltwise help center, reranked by a cross-encoder.

        category (optional): shipping, returns, warranty, billing, orders, account, support, product, troubleshooting."""
        if not query.strip():
            raise ToolError("query must not be empty")
        if len(query) > 1000:  # public tool: bound the embedding/rerank work one call can trigger
            raise ToolError("query must be at most 1000 characters")
        retriever = await get_retriever()
        cfg = retriever.default_config()
        hits = await retriever.retrieve(query, category=category, config=replace(cfg, top_k=max(1, min(top_k, 10))))
        return [
            KBHit(
                article_id=h.doc_id,
                title=h.title,
                section=h.section,
                url=h.url,
                snippet=h.text[:700],
                relevance=h.rerank_score,
            )
            for h in hits
        ]

    @mcp.tool(annotations=read_only)
    async def get_article(article_id: str) -> str:
        """Get the full Markdown text of a help-center article by id (e.g. KB-003)."""
        doc = _docs().get(article_id.strip().upper())
        if doc is None:
            raise ToolError(f"Unknown article {article_id}. Use search_knowledge_base first.")
        return f"# {doc.title}\nSource: {doc.url}\n\n{doc.body}"

    @mcp.resource("kb://index", mime_type="application/json")
    def kb_index() -> list[dict[str, Any]]:
        """All help-center articles: id, title, category and URL."""
        return [{"id": d.doc_id, "title": d.title, "category": d.category, "url": d.url} for d in _docs().values()]

    @mcp.resource("kb://articles/{article_id}", mime_type="text/markdown")
    def article_resource(article_id: str) -> str:
        """A single help-center article as Markdown."""
        doc = _docs().get(article_id.upper())
        if doc is None:
            raise ValueError(f"Unknown article {article_id}")
        return doc.body

    @mcp.prompt
    def grounded_answer(question: str) -> list[Message]:
        """Answer a customer question strictly from the help center, with citations."""
        return [
            Message(
                "You are a Voltwise support expert. First call search_knowledge_base with a focused query. "
                "Answer ONLY from the returned snippets, cite article titles, and say clearly when the "
                f"help center doesn't cover the question.\n\nQuestion: {question}"
            )
        ]

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "knowledge"})

    return mcp
