"""Graph factories for LangGraph Studio / Agent Server (``langgraph dev``).

``langgraph.json`` points here. The Agent Server brings its own checkpointer and
store, so these factories compile the graphs *without* persistence. Heavy,
long-lived services (ONNX models, vector store, MCP toolkit) are created once per
process and reused; only the (cheap) graph compilation happens per call.

    uv run caseflow serve commerce & uv run caseflow serve helpdesk &
    uv run --with "langgraph-cli[inmem]" langgraph dev --allow-blocking

``--with`` layers the CLI on top of the project environment instead of adding it to the
lock file: ``langgraph-api`` pins older ``sse-starlette``/``structlog`` than the API needs.
``--allow-blocking`` is only needed with *embedded* Qdrant (its local client does sync disk
I/O); point ``CASEFLOW_RETRIEVAL__QDRANT_URL`` at a Qdrant server to drop it.
``langgraph.json`` references ``caseflow.studio:make_graph`` by *module* path so both graphs
share this module (and one embedded-Qdrant client) instead of loading the file twice.
"""

from __future__ import annotations

import asyncio
from typing import Any

from caseflow.agents.graph import build_support_graph
from caseflow.agents.specialists import GraphDeps
from caseflow.agents.toolkit import MCPToolkit
from caseflow.config import get_settings
from caseflow.llm import ModelRegistry
from caseflow.rag.embeddings import EmbeddingModels
from caseflow.rag.graph import build_knowledge_graph
from caseflow.rag.ingest import ingest_knowledge_base
from caseflow.rag.retriever import HybridRetriever
from caseflow.rag.stores import create_vector_store

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
