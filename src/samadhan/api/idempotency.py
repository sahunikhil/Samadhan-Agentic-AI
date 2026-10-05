"""``Idempotency-Key`` for ``POST /v1/chat`` (IETF draft "The Idempotency-Key HTTP Header Field").

Why: a mobile client times out after 30 s and retries. Without idempotency the turn runs twice -
two LLM bills, two tickets, in the worst case a second refund attempt. With it, the retry gets the
*stored* response of the first run.

Semantics (the same as Stripe's API):

* first request with a key             -> run, store the response (24 h)
* same key + same body, after success  -> the stored response, header ``Idempotent-Replayed: true``
* same key while the first is running  -> ``409 Conflict`` (retry later)
* same key + a *different* body        -> ``422`` (a key identifies one operation)
* the first run fails                  -> the key is released, a retry runs again

**Atomic across replicas.** Keys live in a SQL table whose primary key is (principal, key); a request
claims a key with ``INSERT ... ON CONFLICT DO NOTHING``, so when two replicas receive the same key at
the same moment, the database picks exactly one winner. Postgres in production (the same database as
the checkpoints), SQLite locally. A run that crashed mid-way is taken over after
``in_progress_timeout_s`` with a compare-and-set update, so a key never stays locked forever.

Keys are scoped per principal (one customer can't replay another's response). Defense in depth: the
*tools* are idempotent too - ``issue_refund`` refuses to exceed what is refundable, so even a
duplicate run could not pay twice.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException, status
from sqlalchemy import Column, Float, MetaData, String, Table, Text, delete, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_KEY = re.compile(r"^[A-Za-z0-9_\-:]{8,128}$")
_metadata = MetaData()
KEYS = Table(
    "idempotency_keys",
    _metadata,
    Column("principal", String(80), primary_key=True),
    Column("key", String(128), primary_key=True),
    Column("body_hash", String(64), nullable=False),
    Column("status", String(16), nullable=False),  # in_progress | done
    Column("response", Text),
    Column("created_at", Float, nullable=False),
)


def fingerprint(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def database_url(postgres_url: str | None, sqlite_path: Path) -> str:
    """The checkpoint database in production (``postgresql://`` -> async psycopg driver), else SQLite."""
    if postgres_url:
        return re.sub(r"^postgres(?:ql)?(?:\+\w+)?://", "postgresql+psycopg://", postgres_url)
    return f"sqlite+aiosqlite:///{sqlite_path.as_posix()}"


class IdempotencyStore:
    def __init__(self, url: str, *, ttl_s: float = 24 * 3600, in_progress_timeout_s: float = 300) -> None:
        self.engine: AsyncEngine = create_async_engine(url)
        self.ttl_s = ttl_s
        self.in_progress_timeout_s = in_progress_timeout_s  # a crashed run must not lock a key forever

    async def setup(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    @staticmethod
    def validate(key: str) -> str:
        if not _KEY.fullmatch(key):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Idempotency-Key must be 8-128 characters of [A-Za-z0-9_-:]"
            )
        return key

    async def begin(self, principal: str, key: str, body_hash: str) -> tuple[Literal["new", "replay"], Any]:
        now = time.time()
        pk = (KEYS.c.principal == principal) & (KEYS.c.key == key)
        dialect = postgresql if self.engine.dialect.name == "postgresql" else sqlite
        claim = (
            dialect.insert(KEYS)
            .values(principal=principal, key=key, body_hash=body_hash, status="in_progress", created_at=now)
            .on_conflict_do_nothing(index_elements=["principal", "key"])
        )
        async with self.engine.begin() as conn:
            if (await conn.execute(claim)).rowcount == 1:
                return "new", None  # we won the key
            row = (await conn.execute(select(KEYS).where(pk))).mappings().first()
            if row is None:  # released between our insert and select: claim it now
                await conn.execute(claim)
                return "new", None
            expired = now - row["created_at"] > self.ttl_s
            if not expired and row["body_hash"] != body_hash:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    "This Idempotency-Key was already used for a different request.",
                )
            if not expired and row["status"] == "done":
                return "replay", json.loads(row["response"])
            if not expired and now - row["created_at"] < self.in_progress_timeout_s:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "A request with this Idempotency-Key is still being processed.",
                    headers={"Retry-After": "5"},
                )
            # Expired, or a stale run that crashed: take it over - compare-and-set on the row we saw,
            # so only one of several concurrent retries wins.
            taken = await conn.execute(
                update(KEYS)
                .where(pk, KEYS.c.created_at == row["created_at"], KEYS.c.status == row["status"])
                .values(body_hash=body_hash, status="in_progress", response=None, created_at=now)
            )
            if taken.rowcount == 1:
                return "new", None
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "A request with this Idempotency-Key is still being processed.",
                headers={"Retry-After": "5"},
            )

    async def complete(self, principal: str, key: str, body_hash: str, response: Any) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                update(KEYS)
                .where(KEYS.c.principal == principal, KEYS.c.key == key, KEYS.c.body_hash == body_hash)
                .values(status="done", response=json.dumps(response, default=str), created_at=time.time())
            )

    async def release(self, principal: str, key: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(delete(KEYS).where(KEYS.c.principal == principal, KEYS.c.key == key))
