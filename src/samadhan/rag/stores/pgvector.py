"""pgvector hybrid store: HNSW cosine search + Postgres full-text search, fused by RRF in SQL.

Postgres doesn't have Qdrant's native BM25 sparse vectors, but it has something just
as good for lexical matching: ``tsvector`` full-text search with stemming and a GIN
index. We run both retrievers as CTEs and fuse them with Reciprocal Rank Fusion in a
single SQL statement - one round trip, no application-side merging.

Choosing this backend means one database for everything (checkpoints, long-term
memory, vectors) - the lowest-cost, lowest-ops production option up to a few
million chunks.
"""

from __future__ import annotations

import re
from typing import Any

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from samadhan.rag.documents import Chunk
from samadhan.rag.embeddings import SparseVector
from samadhan.rag.stores.base import SearchHit, SearchMode

_WORD = re.compile(r"[A-Za-z0-9]+")


def _or_tsquery(text: str) -> str:
    """Build an OR tsquery ('return | earbud | seal'). ``plainto_tsquery`` ANDs every word,
    which returns nothing for most natural-language questions."""
    words = [w.lower() for w in _WORD.findall(text) if len(w) > 1]
    return " | ".join(dict.fromkeys(words)) or "a"


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str, *, table: str = "kb_chunks") -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", table):
            raise ValueError(f"Invalid table name: {table!r}")
        self.table = table
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            dsn, min_size=1, max_size=5, open=False, kwargs={"autocommit": True}, configure=self._configure
        )
        self._opened = False

    @staticmethod
    async def _configure(conn: AsyncConnection[Any]) -> None:
        await register_vector_async(conn)

    async def _open(self) -> None:
        if not self._opened:
            # The extension must exist before `register_vector_async` runs on pooled connections.
            async with await AsyncConnection.connect(self._dsn, autocommit=True) as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await self._pool.open()
            self._opened = True

    def _t(self, suffix: str = "") -> sql.Identifier:
        return sql.Identifier(f"{self.table}{suffix}")

    async def ensure_schema(self, dense_dim: int) -> None:
        await self._open()
        async with self._pool.connection() as conn:
            await conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {t} (
                        chunk_id     uuid PRIMARY KEY,
                        doc_id       text NOT NULL,
                        title        text NOT NULL,
                        url          text,
                        section      text,
                        category     text,
                        product      text,
                        text         text NOT NULL,
                        embed_text   text NOT NULL,
                        position     int,
                        content_hash text NOT NULL,
                        embedding    vector({dim}) NOT NULL,
                        tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED
                    )"""
                ).format(t=self._t(), dim=sql.Literal(dense_dim))
            )
            await conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw (embedding vector_cosine_ops)").format(
                    i=self._t("_hnsw"), t=self._t()
                )
            )
            await conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING gin (tsv)").format(i=self._t("_tsv"), t=self._t())
            )
            await conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} (doc_id)").format(i=self._t("_doc"), t=self._t())
            )

    async def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVector]) -> int:
        import numpy as np

        await self._open()
        rows = [
            (c.chunk_id, c.doc_id, c.title, c.url, c.section, c.category, c.product, c.text, c.embed_text,
             c.position, c.content_hash, np.asarray(d, dtype=np.float32))
            for c, d in zip(chunks, dense, strict=True)
        ]  # fmt: skip
        query = sql.SQL(
            """
            INSERT INTO {t} (chunk_id, doc_id, title, url, section, category, product, text, embed_text,
                             position, content_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id, title = EXCLUDED.title, url = EXCLUDED.url, section = EXCLUDED.section,
                category = EXCLUDED.category, product = EXCLUDED.product, text = EXCLUDED.text,
                embed_text = EXCLUDED.embed_text, position = EXCLUDED.position,
                content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding
            """
        ).format(t=self._t())
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(query, rows)
        return len(rows)

    async def delete_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        await self._open()
        async with self._pool.connection() as conn:
            await conn.execute(sql.SQL("DELETE FROM {t} WHERE doc_id = ANY(%s)").format(t=self._t()), (doc_ids,))

    async def document_hashes(self) -> dict[str, str]:
        await self._open()
        async with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql.SQL("SELECT to_regclass(%s) IS NOT NULL AS exists"), (self.table,))
            row = await cur.fetchone()
            if not row or not row["exists"]:
                return {}
            await cur.execute(sql.SQL("SELECT DISTINCT doc_id, content_hash FROM {t}").format(t=self._t()))
            return {r["doc_id"]: r["content_hash"] for r in await cur.fetchall()}

    async def search(
        self,
        *,
        query_text: str,
        dense: list[float],
        sparse: SparseVector,  # unused: Postgres FTS plays the lexical role
        mode: SearchMode,
        limit: int,
        candidate_k: int,
        category: str | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        import numpy as np

        await self._open()
        qvec = np.asarray(dense, dtype=np.float32)
        cat = sql.SQL("AND category = %(category)s") if category else sql.SQL("")
        params: dict[str, Any] = {
            "q": qvec, "tsq": _or_tsquery(query_text), "k": candidate_k, "limit": limit,
            "wd": dense_weight if mode != "sparse" else 0.0, "ws": sparse_weight if mode != "dense" else 0.0,
            "rrf_k": rrf_k, "category": category,
        }  # fmt: skip
        query = sql.SQL(
            """
            WITH dense AS (
                SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %(q)s) AS r
                FROM {t} WHERE %(wd)s > 0 {cat}
                ORDER BY embedding <=> %(q)s LIMIT %(k)s
            ),
            lexical AS (
                SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, to_tsquery('english', %(tsq)s)) DESC) AS r
                FROM {t} WHERE %(ws)s > 0 AND tsv @@ to_tsquery('english', %(tsq)s) {cat}
                ORDER BY ts_rank_cd(tsv, to_tsquery('english', %(tsq)s)) DESC LIMIT %(k)s
            )
            SELECT c.chunk_id::text, c.doc_id, c.title, c.url, c.section, c.category, c.product, c.text,
                   COALESCE(%(wd)s / (%(rrf_k)s + dense.r), 0) + COALESCE(%(ws)s / (%(rrf_k)s + lexical.r), 0) AS score
            FROM {t} c
            LEFT JOIN dense ON dense.chunk_id = c.chunk_id
            LEFT JOIN lexical ON lexical.chunk_id = c.chunk_id
            WHERE dense.chunk_id IS NOT NULL OR lexical.chunk_id IS NOT NULL
            ORDER BY score DESC
            LIMIT %(limit)s
            """
        ).format(t=self._t(), cat=cat)
        async with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
        return [
            SearchHit(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                title=r["title"],
                url=r["url"] or "",
                section=r["section"] or "",
                category=r["category"] or "",
                product=r["product"],
                text=r["text"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def count(self) -> int:
        hashes = await self.document_hashes()
        if not hashes:
            return 0
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql.SQL("SELECT count(*) FROM {t}").format(t=self._t()))
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
