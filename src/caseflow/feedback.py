"""User feedback -> online evaluation -> dataset flywheel.

Offline evals (``caseflow eval ...``) test what we *thought of*. Production shows what we didn't.
The loop that makes an agent better over time:

    1. capture  - thumbs up/down (+ reason) on a reply: ``POST /v1/threads/{id}/feedback``
    2. monitor  - ``caseflow_feedback_total{rating,reason,intent}``: thumbs-down rate per intent is
                  an *online* quality metric (alert when "refund_status" suddenly degrades)
    3. harvest  - ``caseflow feedback harvest``: every thumbs-down becomes a *candidate* eval case
                  (the customer's message, what the agent did, why the customer disliked it)
    4. label    - a human writes the expected behavior (``needs_label``); candidates are never
                  auto-promoted: a customer's thumbs-down is a signal, not ground truth
    5. promote  - labeled cases join ``evals/datasets/agent_scenarios.jsonl`` -> the failure is now a
                  regression test in CI, and the fix is measured, not assumed

Stored in the LangGraph ``BaseStore`` (Postgres in production) under ``("feedback", thread_id)``;
one record per (turn, actor), so re-submitting changes the rating instead of double counting.
The free-text comment is PII-masked (card numbers) before it is stored.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from caseflow.agents.guardrails import mask_card_numbers
from caseflow.observability import FEEDBACK
from caseflow.prompts import PROMPT_VERSION
from caseflow.service import owner_of

NAMESPACE = "feedback"
Rating = Literal["up", "down"]
Reason = Literal["wrong_answer", "not_helpful", "incomplete", "unsafe", "too_slow", "other"]


class FeedbackRequest(BaseModel):
    rating: Rating
    reason: Reason | None = None
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackError(Exception):
    pass


def _ns(thread_id: str) -> tuple[str, ...]:
    return (NAMESPACE, thread_id.replace(".", "_"))


async def record_feedback(
    store: BaseStore, graph: Any, thread_id: str, fb: FeedbackRequest, *, actor: str
) -> dict[str, Any]:
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}
    messages = values.get("messages", [])
    humans = [m for m in messages if isinstance(m, HumanMessage)]
    reply = next((m.text for m in reversed(messages) if isinstance(m, AIMessage) and m.text), None)
    if not humans or reply is None:
        raise FeedbackError("There is no reply to rate in this conversation yet.")
    triage = values.get("triage") or {}
    intent = (triage.get("intents") or ["none"])[0]
    record = {
        "thread_id": thread_id,
        "customer_id": owner_of(thread_id),
        "turn": len(humans),
        "rating": fb.rating,
        "reason": fb.reason,
        "comment": mask_card_numbers(fb.comment) if fb.comment else None,
        "actor": actor,
        "message": humans[-1].text,
        "reply": reply,
        "intents": triage.get("intents", []),
        "agents": sorted({r["agent"] for r in values.get("specialist_results", [])}),
        "outcome": values.get("outcome"),
        "prompt_version": PROMPT_VERSION,
        "created_at": time.time(),
    }
    await store.aput(_ns(thread_id), f"turn-{len(humans)}-{actor}", record, index=False)
    FEEDBACK.labels(rating=fb.rating, reason=fb.reason or "none", intent=intent).inc()
    return record


async def list_feedback(
    store: BaseStore, *, rating: Rating | None = None, limit: int = 100, max_scan: int = 10_000
) -> list[dict[str, Any]]:
    """Newest first, independent of the backend's native order (Postgres returns newest first, the
    in-memory store oldest first - taking the first ``limit`` items would differ by backend)."""
    flt = {"rating": rating} if rating else None
    records: list[dict[str, Any]] = []
    page = 500
    while len(records) < max_scan:
        items = await store.asearch((NAMESPACE,), filter=flt, limit=page, offset=len(records))
        records.extend(i.value for i in items)
        if len(items) < page:
            break
    return sorted(records, key=lambda r: r["created_at"], reverse=True)[:limit]


def to_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """A thumbs-down turn -> an *unlabeled* agent-scenario candidate for human review."""
    digest = hashlib.sha256(f"{record['thread_id']}:{record['turn']}".encode()).hexdigest()[:8]
    return {
        "id": f"C-{digest}",
        "customer_id": record["customer_id"],
        "message": record["message"],
        "needs_label": True,
        # Fill these in (see evals/datasets/agent_scenarios.jsonl), then move the row there:
        "expected_agents": record["agents"],
        "expected_outcome": "TODO",
        "reference_tool_calls": [],
        "forbidden_tools": [],
        "reference_goal": "TODO: what a correct resolution looks like",
        "observed": {k: record[k] for k in ("reply", "outcome", "intents", "reason", "comment", "prompt_version")},
        "source": {"thread_id": record["thread_id"], "turn": record["turn"]},
    }
