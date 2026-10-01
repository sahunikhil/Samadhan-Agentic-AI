"""pgvector backend: same contract, same quality bar as Qdrant (runs in CI against a Postgres service)."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from caseflow.rag.ingest import ingest_knowledge_base
from caseflow.rag.retriever import HybridRetriever, RetrievalConfig

DSN = os.environ.get("CASEFLOW_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="set CASEFLOW_TEST_DATABASE_URL to a pgvector-enabled Postgres"),
]


@pytest.fixture
async def pg_retriever(settings: Any, embeddings: Any) -> Any:
    from caseflow.rag.stores.pgvector import PgVectorStore

    store = PgVectorStore(DSN or "", table=f"kb_test_{uuid.uuid4().hex[:8]}")
    report = await ingest_knowledge_base(settings, store, embeddings, force=True)
    assert report.chunks_written > 50
    yield HybridRetriever(store, embeddings, settings.retrieval)
    async with store._pool.connection() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {store.table}")  # test-only cleanup
    await store.close()


async def test_incremental_ingestion_skips_unchanged_docs(pg_retriever: Any, settings: Any, embeddings: Any) -> None:
    report = await ingest_knowledge_base(settings, pg_retriever.store, embeddings)
    assert report.chunks_written == 0 and report.documents_unchanged == report.documents_total


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
async def test_every_mode_finds_the_right_article(pg_retriever: Any, mode: str) -> None:
    hits = await pg_retriever.retrieve(
        "Is the PowerCell 20K allowed on a plane?", config=RetrievalConfig(mode, rerank=False)
    )
    assert "KB-023" in [h.doc_id for h in hits[:3]]


async def test_hybrid_rerank_and_category_filter(pg_retriever: Any) -> None:
    hits = await pg_retriever.retrieve("restocking fee for opened laptops", category="returns")
    assert hits[0].doc_id == "KB-003"
    assert all(h.category == "returns" for h in hits)
    assert hits[0].rerank_score is not None and hits[0].rerank_score > 0.5
