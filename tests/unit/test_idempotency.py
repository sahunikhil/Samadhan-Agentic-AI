"""Idempotency store: concurrent first requests, replay, conflicts, no lock leak."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException
from langgraph.store.memory import InMemoryStore

from samadhan.api.idempotency import IdempotencyStore, database_url, fingerprint


@pytest.fixture
async def idem(tmp_path: Any) -> Any:
    store = IdempotencyStore(database_url(None, tmp_path / "api.sqlite"))
    await store.setup()
    yield store
    await store.close()


async def test_concurrent_first_requests_run_once(idem: IdempotencyStore, tmp_path: Any) -> None:
    # Two "replicas" (separate engines on one database) racing on the same key: the database decides.
    other = IdempotencyStore(database_url(None, tmp_path / "api.sqlite"))
    body = fingerprint({"message": "refund please"})
    results = await asyncio.gather(
        *(store.begin("cust_1", "key-00000001", body) for store in (idem, other, idem, other, idem)),
        return_exceptions=True,
    )
    await other.close()
    started = [r for r in results if r == ("new", None)]
    conflicts = [r for r in results if isinstance(r, HTTPException) and r.status_code == 409]
    assert len(started) == 1 and len(conflicts) == 4, "exactly one request may run the turn"


async def test_replay_conflict_release_and_scoping(idem: IdempotencyStore) -> None:
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
    from samadhan.feedback import list_feedback

    store = InMemoryStore()  # returns items oldest-first natively
    for i in range(30):
        await store.aput(
            ("feedback", f"t{i}"), "turn-1-customer:x", {"rating": "down", "created_at": float(i)}, index=False
        )
    newest = await list_feedback(store, rating="down", limit=5)
    assert [r["created_at"] for r in newest] == [29.0, 28.0, 27.0, 26.0, 25.0]


async def test_a_crashed_run_is_taken_over_after_the_timeout(tmp_path: Any) -> None:
    store = IdempotencyStore(database_url(None, tmp_path / "api.sqlite"), in_progress_timeout_s=0)
    await store.setup()
    body = fingerprint({"m": 1})
    assert await store.begin("c", "key-00000009", body) == ("new", None)  # ...then the replica crashes
    assert await store.begin("c", "key-00000009", body) == ("new", None)  # a retry takes the stale key over
    await store.close()
