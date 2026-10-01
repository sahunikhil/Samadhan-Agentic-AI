"""Two-stage retrieval: hybrid recall, then cross-encoder precision.

Stage 1 - *recall*: dense + BM25 retrievers each fetch ``candidate_k`` chunks,
fused with RRF. Cheap, and casts a wide net so the right chunk is *somewhere* in
the candidate set.

Stage 2 - *precision*: a cross-encoder scores each (query, chunk) pair jointly and
re-orders the candidates; the top ``top_k`` go to the LLM. Fewer, better chunks
mean lower token cost *and* fewer hallucinations (less distracting context).

Every knob is a parameter, which is what makes the retrieval ablation study in
``caseflow.evaluation.retrieval`` possible (dense vs sparse vs hybrid, +/- rerank).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from opentelemetry.trace import SpanKind

from caseflow.config import RetrievalSettings
from caseflow.observability import RETRIEVAL_LATENCY
from caseflow.rag.embeddings import EmbeddingModels
from caseflow.rag.stores.base import SearchHit, SearchMode, VectorStore
from caseflow.telemetry import current_context, tracer


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    mode: SearchMode = "hybrid"
    rerank: bool = True
    top_k: int = 5
    candidate_k: int = 20

    @property
    def label(self) -> str:
        return f"{self.mode}{'+rerank' if self.rerank else ''}"


class HybridRetriever:
    def __init__(self, store: VectorStore, models: EmbeddingModels, settings: RetrievalSettings) -> None:
        self.store = store
        self.models = models
        self.settings = settings

    def default_config(self) -> RetrievalConfig:
        s = self.settings
        return RetrievalConfig(
            mode="hybrid" if s.hybrid else "dense", rerank=s.rerank, top_k=s.top_k, candidate_k=s.candidate_k
        )

    async def retrieve(
        self,
        query: str,
        *,
        category: str | None = None,
        config: RetrievalConfig | None = None,
        extra_queries: list[str] | None = None,
        rerank_query: str | None = None,
    ) -> list[SearchHit]:
        """Hybrid recall for ``query`` (+ optional ``extra_queries``, unioned = multi-query),
        then cross-encoder precision against ``rerank_query`` (defaults to ``query``).

        Why separate them: a rewritten query is good for *recall* (keywords, resolved pronouns),
        but relevance should be judged against what the customer actually asked. Live evals showed
        rewrites can drop the reranker's score for the right chunk from 0.99 to 0.19.
        """
        cfg = config or self.default_config()
        with tracer().start_as_current_span(
            f"retrieval {self.settings.collection}",
            context=current_context(),  # parent = the graph node calling us
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "retrieval",
                "gen_ai.data_source.id": self.settings.collection,
                "gen_ai.retrieval.top_k": cfg.top_k,
                "db.system.name": self.store.name,
                "caseflow.retrieval.mode": cfg.mode,
                "caseflow.retrieval.rerank": cfg.rerank,
            },
        ) as span:
            hits = await self._retrieve(query, category, cfg, extra_queries, rerank_query)
            span.set_attribute("caseflow.retrieval.returned", len(hits))
            if hits and hits[0].rerank_score is not None:
                span.set_attribute("caseflow.retrieval.top_rerank_score", hits[0].rerank_score)
            return hits

    async def _retrieve(
        self,
        query: str,
        category: str | None,
        cfg: RetrievalConfig,
        extra_queries: list[str] | None,
        rerank_query: str | None,
    ) -> list[SearchHit]:
        started = time.perf_counter()
        # When reranking, over-fetch so the cross-encoder has candidates to promote.
        fetch = cfg.candidate_k if cfg.rerank else cfg.top_k
        hits: dict[str, SearchHit] = {}
        for q in dict.fromkeys([query, *(extra_queries or [])]):
            dense, sparse = await self.models.embed_query(q)
            for hit in await self.store.search(
                query_text=q,
                dense=dense,
                sparse=sparse,
                mode=cfg.mode,
                limit=fetch,
                candidate_k=cfg.candidate_k,
                category=category,
                dense_weight=self.settings.dense_weight,
                sparse_weight=self.settings.sparse_weight,
                rrf_k=self.settings.rrf_k,
            ):
                hits.setdefault(hit.chunk_id, hit)  # first query's fusion score wins for ties
        ranked = list(hits.values())
        if cfg.rerank and ranked:
            # The first ~700 chars carry the relevance signal; longer inputs only add latency.
            scores = await self.models.rerank(
                rerank_query or query, [f"{h.title} > {h.section}\n{h.text[:700]}" for h in ranked]
            )
            for hit, score in zip(ranked, scores, strict=True):
                hit.rerank_score = round(score, 4)
            ranked.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        RETRIEVAL_LATENCY.observe(time.perf_counter() - started)
        return ranked[: cfg.top_k]
