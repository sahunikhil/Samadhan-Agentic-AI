"""Local embedding + reranking models (FastEmbed / ONNX Runtime).

Why local models?
  * **$0 and no API key** - runs on any CPU, including a free-tier VM.
  * **Privacy** - customer questions never leave the box to be embedded.
  * **Fast** - bge-small (33M params) embeds a query in a few milliseconds.

Three models, three jobs:
  * dense  ``BAAI/bge-small-en-v1.5`` (384-d) - semantic similarity ("send back" ~ "return").
  * sparse ``Qdrant/bm25`` - exact keyword match (SKUs, "VC-65W", "IPX5", numbers).
  * cross-encoder ``ms-marco-MiniLM-L-6-v2`` - reads (query, chunk) *together* and
    scores relevance far more accurately than vector similarity; too slow for the
    whole corpus, perfect for re-ordering the top ~20 candidates.

ONNX inference is CPU-bound and synchronous, so every call is pushed to a worker
thread (``asyncio.to_thread``) - otherwise one embedding call would freeze the
event loop and every other concurrent request with it.
"""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings

from samadhan.config import RetrievalSettings


@dataclass(slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class EmbeddingModels:
    """Lazily-loaded, thread-safe holder for the three FastEmbed models."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._dense: Any = None
        self._sparse: Any = None
        self._reranker: Any = None

    @property
    def dense_dim(self) -> int:
        return 384 if "small" in self.settings.dense_model else 768

    def _cache_kwargs(self) -> dict[str, Any]:
        d = self.settings.model_cache_dir
        return {"cache_dir": str(d)} if d else {}

    def _get_dense(self) -> Any:
        with self._lock:
            if self._dense is None:
                from fastembed import TextEmbedding

                self._dense = TextEmbedding(self.settings.dense_model, **self._cache_kwargs())
            return self._dense

    def _get_sparse(self) -> Any:
        with self._lock:
            if self._sparse is None:
                from fastembed import SparseTextEmbedding

                self._sparse = SparseTextEmbedding(self.settings.sparse_model, **self._cache_kwargs())
            return self._sparse

    def _get_reranker(self) -> Any:
        with self._lock:
            if self._reranker is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._reranker = TextCrossEncoder(self.settings.reranker_model, **self._cache_kwargs())
            return self._reranker

    def warmup(self) -> None:
        """Load all models up-front (at startup, not on the first customer request)."""
        self._get_dense()
        self._get_sparse()
        if self.settings.rerank:
            self._get_reranker()

    # ---- sync primitives (run in worker threads) ------------------------------------
    def _embed_passages(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVector]]:
        dense = [v.tolist() for v in self._get_dense().passage_embed(texts, batch_size=32)]
        sparse = [
            SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
            for s in self._get_sparse().passage_embed(texts, batch_size=32)
        ]
        return dense, sparse

    def _embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        dense = next(iter(self._get_dense().query_embed(text))).tolist()
        s = next(iter(self._get_sparse().query_embed(text)))
        return dense, SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

    def _rerank(self, query: str, docs: list[str]) -> list[float]:
        # Small batches: a cross-encoder pads every pair in a batch to the longest one, so
        # batch_size=8 is ~35% faster than 32 on mixed-length chunks (measured, CPU).
        return [float(x) for x in self._get_reranker().rerank(query, docs, batch_size=8)]

    # ---- async API --------------------------------------------------------------------
    async def embed_passages(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVector]]:
        return await asyncio.to_thread(self._embed_passages, texts)

    async def embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        return await asyncio.to_thread(self._embed_query, text)

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Return relevance probabilities in [0, 1] (sigmoid of the cross-encoder logit)."""
        if not docs:
            return []
        logits = await asyncio.to_thread(self._rerank, query, docs)
        return [sigmoid(x) for x in logits]


class FastEmbedEmbeddings(Embeddings):
    """LangChain ``Embeddings`` adapter so the same local model powers the LangGraph
    Store's semantic memory search (and RAGAS answer-relevancy)."""

    def __init__(self, models: EmbeddingModels) -> None:
        self._models = models

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._models._get_dense().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._models._get_dense().query_embed(text))).tolist()  # type: ignore[no-any-return]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
