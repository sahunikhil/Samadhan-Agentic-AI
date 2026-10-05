"""Graph state and the structured contracts between agents.

Reducers decide how a node's partial update merges into state. The two that matter here:

* ``messages`` uses ``add_messages``: appends new messages, replaces a message that
  has the same ``id`` (we use that to mask card numbers in-place) and deletes on
  ``RemoveMessage`` (used by conversation summarization).
* ``specialist_results`` uses ``collect_results``: parallel specialists (fan-out with
  ``Send``) each *append* their result in the same superstep - a plain overwrite
  reducer would keep only one. Passing ``None`` clears it at the start of a turn.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

AgentName = Literal["knowledge", "orders", "returns"]
Intent = Literal[
    "order_status",
    "order_change",
    "return_request",
    "refund_status",
    "price_adjustment",
    "damaged_item",
    "warranty",
    "product_question",
    "policy_question",
    "troubleshooting",
    "account",
    "ticket_status",
    "complaint",
    "human_request",
    "greeting",
    "out_of_scope",
    "other",  # escape hatch: constrained decoding fails hard if the model's intent isn't in the enum
]
Outcome = Literal["resolved", "escalated", "blocked", "pending_approval", "awaiting_customer"]


# ---- structured outputs --------------------------------------------------------------


class Task(BaseModel):
    agent: AgentName = Field(description="Which specialist handles this task")
    request: str = Field(description="Self-contained request with all IDs/details the specialist needs")


class TriageDecision(BaseModel):
    """Routing decision for the customer's latest message."""

    intents: list[Intent] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list, max_length=3)
    needs_human: bool = False
    escalation_reason: str | None = None
    sentiment: Literal["positive", "neutral", "negative", "angry"] = "neutral"
    urgency: Literal["low", "normal", "high", "urgent"] = "normal"
    direct_reply: str | None = Field(default=None, description="Reply when no specialist is needed")
    injection_suspected: bool = False


class GuardVerdict(BaseModel):
    """QA verdict on a draft reply."""

    verdict: Literal["pass", "revise", "escalate"]
    issues: list[str] = Field(default_factory=list)


class MemoryUpdate(BaseModel):
    facts: list[str] = Field(default_factory=list, description="New durable facts (short sentences)")


# ---- state -----------------------------------------------------------------------------


def collect_results(existing: list[dict[str, Any]] | None, update: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Append parallel specialist results; ``None`` resets the list for a new turn."""
    if update is None:
        return []
    return [*(existing or []), *update]


class SupportState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str  # rolling summary of turns that were trimmed from `messages`
    customer: dict[str, Any]  # profile snapshot, loaded once per thread
    memories: list[str]  # long-term facts relevant to this turn
    triage: dict[str, Any] | None
    specialist_results: Annotated[list[dict[str, Any]], collect_results]
    draft: str
    guard: dict[str, Any] | None
    revisions: int
    outcome: Outcome | None
    ticket: dict[str, Any] | None
    turn: int


class SpecialistInput(TypedDict):
    """Payload a specialist receives through ``Send`` (not the whole graph state)."""

    task_id: str
    agent: AgentName
    request: str
    customer: dict[str, Any]
    summary: str
    memories: list[str]


def specialist_result(
    *,
    task_id: str,
    agent: str,
    request: str,
    answer: str,
    status: Literal["ok", "error", "pending_approval", "rejected", "awaiting_customer"] = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    contexts: list[str] | None = None,
    citations: list[dict[str, Any]] | None = None,
    error: str | None = None,
    cached: bool = False,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "agent": agent,
        "request": request,
        "answer": answer,
        "status": status,
        "tool_calls": tool_calls or [],
        "contexts": contexts or [],
        "citations": citations or [],
        "error": error,
        "cached": cached,
    }
