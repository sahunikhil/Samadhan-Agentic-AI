"""Chunking ablation: how should the knowledge base be split? (no LLM - runs in seconds/minutes)

Each configuration gets its own in-memory Qdrant collection, the full KB is chunked + embedded
with it, and the 45 labeled retrieval queries run through the production retriever (hybrid +
rerank). Reported per configuration:

* quality  - hit@1, hit@5, MRR, nDCG@5 (document-level, like the retrieval suite)
* cost     - number of chunks, average chunk length, and **context chars @5**: how much text the
             generator receives per question. Document-level hits are *lenient to big chunks*
             (a 2,000-char chunk "hits" if the answer is anywhere inside), so quality must be read
             together with the context it costs - more context = more tokens, more latency, and
             more distraction for the generator ("lost in the middle").

Configurations isolate one variable at a time against the default (markdown-aware, 1000 chars,
150 overlap, contextual header): chunk size, structure awareness, and the contextual header.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from samadhan.config import Settings
from samadhan.evaluation.datasets import retrieval_cases
from samadhan.evaluation.report import SuiteResult, mean
from samadhan.evaluation.retrieval_metrics import score_ranking
from samadhan.rag.documents import chunk_documents, load_knowledge_base
from samadhan.rag.embeddings import EmbeddingModels
from samadhan.rag.retriever import HybridRetriever
from samadhan.rag.stores.qdrant import QdrantHybridStore

# (label, chunk_size, overlap, strategy, contextual_header)
CONFIGS: list[tuple[str, int, int, str, bool]] = [
    ("md-500", 500, 75, "markdown", True),
    ("md-1000 (default)", 1000, 150, "markdown", True),
    ("md-2000", 2000, 300, "markdown", True),
    ("md-1000-no-header", 1000, 150, "markdown", False),
    ("fixed-1000", 1000, 150, "fixed", True),
    ("fixed-1000-no-header", 1000, 150, "fixed", False),
]


async def evaluate_chunking(settings: Settings, *, k: int = 5) -> SuiteResult:
    import asyncio

    models = EmbeddingModels(settings.retrieval)
    await asyncio.to_thread(models.warmup)
    docs = load_knowledge_base(settings.retrieval.kb_dir)
    cases = retrieval_cases()
    summary: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for label, size, overlap, strategy, header in CONFIGS:
        chunks = chunk_documents(
            docs, chunk_size=size, chunk_overlap=overlap, strategy=strategy, contextual_header=header
        )
        store = QdrantHybridStore(collection=f"ablation_{len(rows)}", path=Path(":memory:"))
        await store.ensure_schema(models.dense_dim)
        started = time.perf_counter()
        dense, sparse = await models.embed_passages([c.embed_text for c in chunks])
        await store.upsert(chunks, dense, sparse)
        index_s = time.perf_counter() - started
        retriever = HybridRetriever(store, models, settings.retrieval)
        per_metric: dict[str, list[float]] = {}
        context_chars: list[int] = []
        latencies: list[float] = []
        for case in cases:
            t0 = time.perf_counter()
            hits = await retriever.retrieve(case.question)
            latencies.append((time.perf_counter() - t0) * 1000)
            context_chars.append(sum(len(h.text) for h in hits[:k]))
            for name, value in score_ranking([h.doc_id for h in hits], case.relevant, k=k).items():
                per_metric.setdefault(name, []).append(value)
        await store.close()
        row: dict[str, Any] = {
            "config": label,
            "chunk_size": size,
            "strategy": strategy,
            "contextual_header": header,
            "chunks": len(chunks),
            "avg_chunk_chars": round(statistics.mean(len(c.text) for c in chunks)),
            f"context_chars@{k}": round(statistics.mean(context_chars)),
            "index_seconds": round(index_s, 1),
            "p50_ms": round(statistics.median(latencies), 1),
            **{name: mean(values) for name, values in per_metric.items()},
        }
        rows.append(row)
        for key in ("hit@1", f"hit@{k}", "mrr", f"ndcg@{k}", f"context_chars@{k}", "chunks"):
            if key in row:
                summary[f"{label}.{key}"] = float(row[key])
    return SuiteResult("chunking", summary, rows, meta={"cases": len(cases), "k": k, "retriever": "hybrid+rerank"})
