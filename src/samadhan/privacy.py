"""Right to erasure: delete everything the support service stores about one customer.

GDPR Art. 17 (and UK GDPR, India's DPDP Act s. 12) lets a customer have their personal data erased.
What this service keeps per customer, and where:

=====================================  ============================================================
conversation threads (checkpoints)     checkpointer - every thread whose run metadata names the customer
long-term memories                     Store namespaces under ``("customers", <id>)``
feedback (message, reply, comment)    Store ``("feedback", <thread>)``, field ``customer_id``
idempotency keys (stored responses)    SQL ``idempotency_keys``, ``principal`` = customer id
A2A tasks (agent-to-agent history)     SQL ``tasks``, ``owner`` = customer id
=====================================  ============================================================

Deliberately out of scope: the semantic answer cache holds only general knowledge answers (personal
questions bypass it); orders and tickets belong to the commerce and helpdesk systems of record, which
apply their own retention rules (invoices, for example, must be kept for tax law); logs carry ids but
no message content and expire with the log pipeline's retention.
"""

from __future__ import annotations

from typing import Any

from a2a.server.models import TaskModel
from sqlalchemy import delete, inspect
from sqlalchemy.ext.asyncio import AsyncEngine

from samadhan.api.idempotency import KEYS
from samadhan.feedback import NAMESPACE as FEEDBACK_NAMESPACE
from samadhan.service import owner_of

_PAGE = 100


async def _delete_items(store: Any, namespace: tuple[str, ...], filter: dict[str, Any] | None = None) -> int:
    deleted = 0
    while items := await store.asearch(namespace, filter=filter, limit=_PAGE):
        for item in items:
            await store.adelete(item.namespace, item.key)
        deleted += len(items)
    return deleted


async def erase_customer(
    checkpointer: Any, store: Any, customer_id: str, *, engine: AsyncEngine | None = None
) -> dict[str, Any]:
    """Delete the customer's threads, memories, feedback and API records; returns what was removed."""
    threads = {
        tid
        async for checkpoint in checkpointer.alist(None, filter={"customer_id": customer_id})
        if owner_of(tid := checkpoint.config["configurable"]["thread_id"]) == customer_id
    }
    for thread_id in threads:
        await checkpointer.adelete_thread(thread_id)

    memories = 0
    for namespace in await store.alist_namespaces(prefix=("customers", customer_id)):
        memories += await _delete_items(store, namespace)
    feedback = await _delete_items(store, (FEEDBACK_NAMESPACE,), {"customer_id": customer_id})

    sql_rows = 0
    if engine is not None:
        async with engine.begin() as conn:
            sql_rows += (await conn.execute(delete(KEYS).where(KEYS.c.principal == customer_id))).rowcount
            # The A2A task table exists only once the A2A endpoint has been enabled.
            if await conn.run_sync(lambda sync: inspect(sync).has_table(TaskModel.__tablename__)):
                sql_rows += (await conn.execute(delete(TaskModel).where(TaskModel.owner == customer_id))).rowcount

    return {
        "customer_id": customer_id,
        "threads": len(threads),
        "memories": memories,
        "feedback": feedback,
        "api_records": sql_rows,
    }
