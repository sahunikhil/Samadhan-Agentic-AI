"""Graph factories for LangGraph Studio / Agent Server (``langgraph dev``).

``langgraph.json`` points here. The Agent Server brings its own checkpointer and
store, so these factories compile the graphs *without* persistence. Heavy,
long-lived services (ONNX models, vector store, MCP toolkit) are created once per
process and reused; only the (cheap) graph compilation happens per call.

    uv run samadhan serve commerce & uv run samadhan serve helpdesk &
    uv run --with "langgraph-cli[inmem]" langgraph dev --allow-blocking

``--with`` layers the CLI on top of the project environment instead of adding it to the
lock file: ``langgraph-api`` pins older ``sse-starlette``/``structlog`` than the API needs.
``--allow-blocking`` is only needed with *embedded* Qdrant (its local client does sync disk
I/O); point ``SAMADHAN_RETRIEVAL__QDRANT_URL`` at a Qdrant server to drop it.
``langgraph.json`` references ``samadhan.studio:make_graph`` by *module* path so both graphs
share this module (and one embedded-Qdrant client) instead of loading the file twice.
"""

from __future__ import annotations

import asyncio
from typing import Any

from samadhan.agents.graph import build_support_graph
from samadhan.agents.specialists import GraphDeps
from samadhan.agents.toolkit import MCPToolkit
from samadhan.config import get_settings
from samadhan.llm import ModelRegistry
from samadhan.rag.embeddings import EmbeddingModels
from samadhan.rag.graph import build_knowledge_graph
from samadhan.rag.ingest import ingest_knowledge_base
from samadhan.rag.retriever import HybridRetriever
from samadhan.rag.stores import create_vector_store

_deps: GraphDeps | None = None
_lock = asyncio.Lock()


async def _get_deps() -> GraphDeps:
    global _deps
    async with _lock:
        if _deps is None:
            settings = get_settings()
            embeddings = EmbeddingModels(settings.retrieval)
            await asyncio.to_thread(embeddings.warmup)
            # Constructors touch the filesystem (embedded Qdrant); keep that off the event loop -
            # `langgraph dev` flags blocking calls in async code.
            store = await asyncio.to_thread(create_vector_store, settings)
            await store.ensure_schema(embeddings.dense_dim)
            if await store.count() == 0:
                await ingest_knowledge_base(settings, store, embeddings)
            models = ModelRegistry(settings.llm)
            retriever = HybridRetriever(store, embeddings, settings.retrieval)
            _deps = GraphDeps(
                settings=settings,
                models=models,
                retriever=retriever,
                toolkit=MCPToolkit(settings),
                knowledge_graph=build_knowledge_graph(settings, models, retriever),
            )
        return _deps


async def make_graph() -> Any:
    """The full multi-agent support graph (Studio injects checkpointer + store)."""
    return build_support_graph(await _get_deps())


async def make_knowledge_graph() -> Any:
    """The Corrective-RAG subgraph on its own - handy for prompt/retrieval iteration."""
    return (await _get_deps()).knowledge_graph
