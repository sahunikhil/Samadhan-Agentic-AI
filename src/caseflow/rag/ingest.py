"""Incremental, idempotent knowledge-base ingestion.

Run it as often as you like (CI, cron, on every deploy): documents whose content
hash hasn't changed are skipped, changed documents are re-chunked and re-embedded,
and documents deleted from the source are removed from the index. This keeps
embedding cost proportional to *changes*, not to corpus size.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from caseflow.config import Settings
from caseflow.observability import get_logger
from caseflow.rag.documents import chunk_documents, load_knowledge_base
from caseflow.rag.embeddings import EmbeddingModels
from caseflow.rag.stores.base import VectorStore

log = get_logger(__name__)


@dataclass
class IngestReport:
    documents_total: int = 0
    documents_indexed: list[str] = field(default_factory=list)
    documents_unchanged: int = 0
    documents_deleted: list[str] = field(default_factory=list)
    chunks_written: int = 0
    seconds: float = 0.0


async def ingest_knowledge_base(
    settings: Settings, store: VectorStore, models: EmbeddingModels, *, force: bool = False
) -> IngestReport:
    started = time.perf_counter()
    r = settings.retrieval
    docs = load_knowledge_base(r.kb_dir)
    await store.ensure_schema(models.dense_dim)

    indexed = {} if force else await store.document_hashes()
    changed = [d for d in docs if indexed.get(d.doc_id) != d.content_hash]
    removed = sorted(set(indexed) - {d.doc_id for d in docs})

    report = IngestReport(documents_total=len(docs), documents_unchanged=len(docs) - len(changed))
    if removed:
        await store.delete_documents(removed)
        report.documents_deleted = removed
    if changed:
        # Delete old chunks first: a shorter new version must not leave orphan chunks behind.
        await store.delete_documents([d.doc_id for d in changed])
        chunks = chunk_documents(changed, chunk_size=r.chunk_size, chunk_overlap=r.chunk_overlap)
        dense, sparse = await models.embed_passages([c.embed_text for c in chunks])
        report.chunks_written = await store.upsert(chunks, dense, sparse)
        report.documents_indexed = [d.doc_id for d in changed]

    report.seconds = round(time.perf_counter() - started, 2)
    log.info(
        "kb_ingested",
        backend=store.name,
        indexed=len(report.documents_indexed),
        unchanged=report.documents_unchanged,
        deleted=len(removed),
        chunks=report.chunks_written,
        seconds=report.seconds,
    )
    return report
