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

Keys are scoped per principal (one customer can't replay another's response) and stored in the
LangGraph ``BaseStore`` - Postgres in production, so every replica sees them.

Defense in depth: the *tools* are idempotent too - ``issue_refund`` refuses to exceed what is
refundable, so even a duplicate run could not pay twice. The API layer avoids the duplicate
work; the MCP server guarantees the business invariant.

Limitation, stated honestly: ``BaseStore`` has no atomic insert-if-absent, so two *simultaneous*
first requests on different replicas could both run. An in-process lock closes the window on
one replica; for a strict cross-replica guarantee put a unique constraint behind it
(``INSERT ... ON CONFLICT DO NOTHING`` in Postgres, or Redis ``SET NX``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any, Literal

from fastapi import HTTPException, status
from langgraph.store.base import BaseStore

NAMESPACE = "idempotency"
_KEY = re.compile(r"^[A-Za-z0-9_\-:]{8,128}$")


def fingerprint(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class IdempotencyStore:
    def __init__(self, store: BaseStore, *, ttl_minutes: int = 24 * 60, in_progress_timeout_s: float = 300) -> None:
        self.store = store
        self.ttl_minutes = ttl_minutes
        self.in_progress_timeout_s = in_progress_timeout_s  # a crashed run must not lock a key forever
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    @staticmethod
    def validate(key: str) -> str:
        if not _KEY.fullmatch(key):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Idempotency-Key must be 8-128 characters of [A-Za-z0-9_-:]"
            )
        return key

    def _ns(self, principal: str) -> tuple[str, ...]:
        return (NAMESPACE, principal.replace(".", "_"))

    async def begin(self, principal: str, key: str, body_hash: str) -> tuple[Literal["new", "replay"], Any]:
        lock_key = (principal, key)
        lock = self._locks.setdefault(lock_key, asyncio.Lock())
        try:
            async with lock:
                return await self._begin(principal, key, body_hash)
        finally:
            # The lock only guards check-then-set; once the record says "in_progress" the store
            # itself answers concurrent requests (409). Drop idle locks so the dict can't grow forever.
            if not lock.locked() and self._locks.get(lock_key) is lock:
                del self._locks[lock_key]

    async def _begin(self, principal: str, key: str, body_hash: str) -> tuple[Literal["new", "replay"], Any]:
        item = await self.store.aget(self._ns(principal), key)
        record = item.value if item else None
        now = time.time()
        if record and now - record["created_at"] > self.ttl_minutes * 60:
            record = None  # expired (stores without native TTL)
        if record:
            if record["body_hash"] != body_hash:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    "This Idempotency-Key was already used for a different request.",
                )
            if record["status"] == "done":
                return "replay", record["response"]
            if now - record["created_at"] < self.in_progress_timeout_s:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "A request with this Idempotency-Key is still being processed.",
                    headers={"Retry-After": "5"},
                )
        await self._put(principal, key, {"status": "in_progress", "body_hash": body_hash, "created_at": now})
        return "new", None

    async def complete(self, principal: str, key: str, body_hash: str, response: Any) -> None:
        record = {"status": "done", "body_hash": body_hash, "created_at": time.time(), "response": response}
        await self._put(principal, key, record)

    async def release(self, principal: str, key: str) -> None:
        await self.store.adelete(self._ns(principal), key)

    async def _put(self, principal: str, key: str, value: dict[str, Any]) -> None:
        kwargs: dict[str, Any] = {"ttl": self.ttl_minutes} if self.store.supports_ttl else {}
        await self.store.aput(self._ns(principal), key, value, index=False, **kwargs)
