"""Qdrant hybrid store: named dense vector + BM25 sparse vector + weighted RRF.

Hybrid search in one round trip using Qdrant's Query API::

    query_points(
        prefetch=[dense top-k, sparse top-k],    # two independent retrievers
        query=RrfQuery(rrf=Rrf(k=60, weights=[1.0, 0.8])),  # fuse by rank, not score
    )

Why Reciprocal Rank Fusion? Dense cosine scores (~0.6-0.9) and BM25 scores (0-30)
live on different scales, so adding them is meaningless. RRF only uses *ranks*:
score(d) = sum_i w_i / (k + rank_i(d)). A chunk ranked well by *both* retrievers
wins; a chunk found by only one still survives. No score calibration needed.

The BM25 sparse vector is stored with ``Modifier.IDF`` so Qdrant computes inverse
document frequency over the live collection (the query only carries term counts).
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import AsyncQdrantClient, models

from caseflow.rag.documents import Chunk
from caseflow.rag.embeddings import SparseVector
from caseflow.rag.stores.base import SearchHit, SearchMode

DENSE = "dense"
SPARSE = "bm25"


class QdrantHybridStore:
    name = "qdrant"

    def __init__(
        self, *, collection: str, url: str | None = None, path: Path | None = None, api_key: str | None = None
    ):
        self.collection = collection
        self.embedded = url is None
        if url:
            self.client = AsyncQdrantClient(url=url, api_key=api_key, timeout=30)
        elif path is not None and str(path) == ":memory:":
            self.client = AsyncQdrantClient(location=":memory:")
        else:
            assert path is not None
            path.mkdir(parents=True, exist_ok=True)
            self.client = AsyncQdrantClient(path=str(path))

    async def ensure_schema(self, dense_dim: int) -> None:
        if await self.client.collection_exists(self.collection):
            return
        await self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            # HNSW defaults are fine at this scale; tune m/ef_construct for millions of chunks.
        )
        if not self.embedded:  # payload indexes are a server feature (no-op in embedded mode)
            for field in ("doc_id", "category"):
                await self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    async def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVector]) -> int:
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={DENSE: d, SPARSE: models.SparseVector(indices=s.indices, values=s.values)},
                payload=c.payload(),
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for i in range(0, len(points), 128):
            await self.client.upsert(self.collection, points=points[i : i + 128], wait=True)
        return len(points)

    async def delete_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        await self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])
            ),
            wait=True,
        )

    async def document_hashes(self) -> dict[str, str]:
        if not await self.client.collection_exists(self.collection):
            return {}
        hashes: dict[str, str] = {}
        offset = None
        while True:
            points, offset = await self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=["doc_id", "content_hash"], with_vectors=False
            )
            for p in points:
                if p.payload:
                    hashes[str(p.payload["doc_id"])] = str(p.payload["content_hash"])
            if offset is None:
                return hashes

    async def search(
        self,
        *,
        query_text: str,
        dense: list[float],
        sparse: SparseVector,
        mode: SearchMode,
        limit: int,
        candidate_k: int,
        category: str | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        flt = (
            models.Filter(must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))])
            if category
            else None
        )
        sparse_q = models.SparseVector(indices=sparse.indices, values=sparse.values)
        if mode == "dense":
            res = await self.client.query_points(
                self.collection, query=dense, using=DENSE, query_filter=flt, limit=limit, with_payload=True
            )
        elif mode == "sparse":
            res = await self.client.query_points(
                self.collection, query=sparse_q, using=SPARSE, query_filter=flt, limit=limit, with_payload=True
            )
        else:
            res = await self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE, limit=candidate_k, filter=flt),
                    models.Prefetch(query=sparse_q, using=SPARSE, limit=candidate_k, filter=flt),
                ],
                query=models.RrfQuery(rrf=models.Rrf(k=rrf_k, weights=[dense_weight, sparse_weight])),
                limit=limit,
                with_payload=True,
            )
        return [
            SearchHit(
                chunk_id=str(p.payload["chunk_id"]),
                doc_id=str(p.payload["doc_id"]),
                title=str(p.payload["title"]),
                url=str(p.payload.get("url") or ""),
                section=str(p.payload.get("section") or ""),
                category=str(p.payload.get("category") or ""),
                text=str(p.payload["text"]),
                score=float(p.score),
            )
            for p in res.points
            if p.payload
        ]

    async def count(self) -> int:
        if not await self.client.collection_exists(self.collection):
            return 0
        return (await self.client.count(self.collection, exact=True)).count

    async def close(self) -> None:
        await self.client.close()
