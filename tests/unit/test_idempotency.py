"""Idempotency store: concurrent first requests, replay, conflicts, no lock leak."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from langgraph.store.memory import InMemoryStore

from caseflow.api.idempotency import IdempotencyStore, fingerprint


async def test_concurrent_first_requests_run_once() -> None:
    idem = IdempotencyStore(InMemoryStore())
    body = fingerprint({"message": "refund please"})
    results = await asyncio.gather(
        *(idem.begin("cust_1", "key-00000001", body) for _ in range(5)), return_exceptions=True
    )
    started = [r for r in results if r == ("new", None)]
    conflicts = [r for r in results if isinstance(r, HTTPException) and r.status_code == 409]
    assert len(started) == 1 and len(conflicts) == 4, "exactly one request may run the turn"
    assert idem._locks == {}, "idle per-key locks are dropped (no unbounded growth)"


async def test_replay_conflict_release_and_scoping() -> None:
    idem = IdempotencyStore(InMemoryStore())
    body, other = fingerprint({"m": 1}), fingerprint({"m": 2})
    assert await idem.begin("cust_1", "key-00000002", body) == ("new", None)
    await idem.complete("cust_1", "key-00000002", body, {"reply": "done"})
    assert await idem.begin("cust_1", "key-00000002", body) == ("replay", {"reply": "done"})
    with pytest.raises(HTTPException) as exc:
        await idem.begin("cust_1", "key-00000002", other)
    assert exc.value.status_code == 422
    assert await idem.begin("cust_2", "key-00000002", body) == ("new", None)  # per-principal keys

    assert await idem.begin("cust_1", "key-00000003", body) == ("new", None)
    await idem.release("cust_1", "key-00000003")  # the turn failed -> a retry may run again
    assert await idem.begin("cust_1", "key-00000003", body) == ("new", None)


async def test_feedback_listing_is_newest_first_on_any_backend() -> None:
    from caseflow.feedback import list_feedback

    store = InMemoryStore()  # returns items oldest-first natively
    for i in range(30):
        await store.aput(
            ("feedback", f"t{i}"), "turn-1-customer:x", {"rating": "down", "created_at": float(i)}, index=False
        )
    newest = await list_feedback(store, rating="down", limit=5)
    assert [r["created_at"] for r in newest] == [29.0, 28.0, 27.0, 26.0, 25.0]
