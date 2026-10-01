"""Composition root: builds and tears down every long-lived dependency.

One function knows how the pieces fit together; everything else receives its
dependencies explicitly (easy to test, easy to swap). Two persistence profiles:

* **Local / zero-infra** (default): SQLite checkpointer, in-memory Store with a
  semantic index, embedded Qdrant on disk. ``uv run caseflow ...`` just works.
* **Production**: one Postgres (``CASEFLOW_PERSISTENCE__DATABASE_URL``) shared by
  the checkpointer, the Store (pgvector-backed semantic memory with TTL) and -
  optionally - the knowledge-base vectors (``CASEFLOW_RETRIEVAL__BACKEND=pgvector``).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from caseflow.agents.graph import build_support_graph
from caseflow.agents.guardrails import PromptGuardClassifier
from caseflow.agents.specialists import GraphDeps
from caseflow.agents.toolkit import MCPToolkit
from caseflow.config import Settings, get_settings
from caseflow.llm import ModelRegistry
from caseflow.observability import configure_logging, get_logger
from caseflow.rag.embeddings import EmbeddingModels, FastEmbedEmbeddings
from caseflow.rag.graph import build_knowledge_graph
from caseflow.rag.ingest import ingest_knowledge_base
from caseflow.rag.retriever import HybridRetriever
from caseflow.rag.semantic_cache import (
    CacheVerdict,
    SemanticCache,
    kb_version_from_hashes,
    store_has_vector_index,
)
from caseflow.rag.stores import VectorStore, create_vector_store
from caseflow.resilience import configure_breakers

log = get_logger(__name__)

MEMORY_TTL_MINUTES = 60 * 24 * 365  # long-term memories expire after a year (data minimization)


@dataclass
class Container:
    settings: Settings
    models: ModelRegistry
    embeddings: EmbeddingModels
    vector_store: VectorStore
    retriever: HybridRetriever
    toolkit: MCPToolkit
    knowledge_graph: CompiledStateGraph[Any, Any, Any, Any]
    graph: CompiledStateGraph[Any, Any, Any, Any]
    checkpointer: Any
    store: Any


def _semantic_cache(
    settings: Settings, models: ModelRegistry, store: Any, vector_store: VectorStore
) -> SemanticCache | None:
    if not settings.retrieval.semantic_cache:
        return None
    if not store_has_vector_index(store):
        log.info("semantic_cache_disabled", reason="store has no vector index")
        return None

    async def kb_version() -> str:
        return kb_version_from_hashes(await vector_store.document_hashes())

    return SemanticCache(
        store,
        kb_version=kb_version,
        verifier=models.structured("fast", CacheVerdict),
        company=settings.agent.company_name,
        candidate_threshold=settings.retrieval.cache_candidate_threshold,
        ttl_minutes=settings.retrieval.cache_ttl_minutes,
    )


@asynccontextmanager
async def build_container(
    settings: Settings | None = None,
    *,
    models: ModelRegistry | None = None,
    checkpointer: Any = None,
    store: Any = None,
    auto_ingest: bool = True,
) -> AsyncIterator[Container]:
    settings = settings or get_settings()
    configure_logging(settings)
    configure_breakers(settings.agent.breaker_failure_threshold, settings.agent.breaker_reset_timeout_s)
    models = models or ModelRegistry(settings.llm)

    async with AsyncExitStack() as stack:
        embeddings = EmbeddingModels(settings.retrieval)
        await asyncio.to_thread(embeddings.warmup)  # load ONNX models at startup, not on first request

        vector_store = create_vector_store(settings)
        stack.push_async_callback(vector_store.close)
        await vector_store.ensure_schema(embeddings.dense_dim)
        if auto_ingest and await vector_store.count() == 0:
            log.info("empty_index_auto_ingest")
            await ingest_knowledge_base(settings, vector_store, embeddings)

        index = {"dims": embeddings.dense_dim, "embed": FastEmbedEmbeddings(embeddings), "fields": ["fact"]}
        if checkpointer is None or store is None:
            db_url = settings.persistence.database_url
            if db_url is not None:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from langgraph.store.postgres.aio import AsyncPostgresStore
                from psycopg.rows import dict_row
                from psycopg_pool import AsyncConnectionPool

                pool = AsyncConnectionPool(
                    db_url.get_secret_value(),
                    max_size=settings.persistence.pool_max_size,
                    open=False,
                    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                )
                await pool.open()
                stack.push_async_callback(pool.close)
                if checkpointer is None:
                    checkpointer = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
                    await checkpointer.setup()
                if store is None:
                    store = AsyncPostgresStore(
                        pool,  # type: ignore[arg-type]
                        index=index,  # type: ignore[arg-type]
                        ttl={"default_ttl": MEMORY_TTL_MINUTES, "refresh_on_read": True},
                    )
                    await store.setup()
            else:
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                if checkpointer is None:
                    settings.persistence.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpointer = await stack.enter_async_context(
                        AsyncSqliteSaver.from_conn_string(str(settings.persistence.sqlite_path))
                    )
                if store is None:
                    store = InMemoryStore(index=index)  # type: ignore[arg-type]

        retriever = HybridRetriever(vector_store, embeddings, settings.retrieval)
        toolkit = MCPToolkit(settings)
        knowledge_graph = build_knowledge_graph(
            settings, models, retriever, cache=_semantic_cache(settings, models, store, vector_store)
        )
        prompt_guard = (
            PromptGuardClassifier(settings.agent.prompt_guard_model, os.environ["GROQ_API_KEY"])
            if settings.agent.prompt_guard_model and os.environ.get("GROQ_API_KEY") and settings.environment != "test"
            else None
        )
        deps = GraphDeps(
            settings=settings, models=models, retriever=retriever, toolkit=toolkit,
            knowledge_graph=knowledge_graph, prompt_guard=prompt_guard,
        )  # fmt: skip
        graph = build_support_graph(deps, checkpointer=checkpointer, store=store)
        log.info(
            "container_ready",
            smart=models.model_name("smart"),
            fast=models.model_name("fast"),
            vector_backend=vector_store.name,
            persistence="postgres" if settings.persistence.database_url else "sqlite",
        )
        yield Container(
            settings=settings,
            models=models,
            embeddings=embeddings,
            vector_store=vector_store,
            retriever=retriever,
            toolkit=toolkit,
            knowledge_graph=knowledge_graph,
            graph=graph,
            checkpointer=checkpointer,
            store=store,
        )
