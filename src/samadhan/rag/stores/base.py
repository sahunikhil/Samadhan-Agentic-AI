"""Backend-agnostic vector store contract."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

from samadhan.rag.documents import Chunk
from samadhan.rag.embeddings import SparseVector

SearchMode = Literal["dense", "sparse", "hybrid"]


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    section: str
    category: str
    text: str
    score: float  # fusion / similarity score from the store (not comparable across modes)
    rerank_score: float | None = None  # cross-encoder relevance probability in [0, 1]
    product: str | None = None  # product scope of the article (e.g. "VoltBook Air 14, VoltBook Pro 16")

    @property
    def header(self) -> str:
        """The contextual header shown with the excerpt (generator *and* judge see the same)."""
        scope = f" (applies to: {self.product})" if self.product else ""
        return f"{self.title}{scope} > {self.section}"


class VectorStore(Protocol):
    name: str

    async def ensure_schema(self, dense_dim: int) -> None: ...

    async def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVector]) -> int: ...

    async def delete_documents(self, doc_ids: list[str]) -> None: ...

    async def document_hashes(self) -> dict[str, str]:
        """doc_id -> content hash of what is currently indexed (for incremental ingestion)."""
        ...

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
    ) -> list[SearchHit]: ...

    async def count(self) -> int: ...

    async def close(self) -> None: ...
