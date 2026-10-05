"""Short-term and long-term memory.

* **Short-term** = the thread's message history, persisted by the LangGraph
  *checkpointer* after every step. Long threads are compacted: older messages are
  summarized into ``state["summary"]`` and removed with ``RemoveMessage``, so the
  prompt stays small no matter how long the conversation runs (cost + latency).
* **Long-term** = facts about the customer that outlive a thread ("prefers store
  credit", "owns a VoltBook Pro 16"), kept in the LangGraph *Store* under the
  namespace ``("customers", <id>, "memories")``. The store has a semantic index
  (same local embedding model as RAG), so each turn recalls only the memories
  relevant to the current message.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import Runnable
from langgraph.store.base import BaseStore

from samadhan.agents.state import MemoryUpdate
from samadhan.prompts import MEMORY_EXTRACTOR, SUMMARIZER


def memory_namespace(customer_id: str) -> tuple[str, ...]:
    return ("customers", customer_id, "memories")


def render_messages(messages: list[AnyMessage], limit_chars: int = 600) -> str:
    lines = []
    for m in messages:
        role = "Customer" if isinstance(m, HumanMessage) else "Assistant" if isinstance(m, AIMessage) else m.type
        if isinstance(m, AIMessage) and m.tool_calls and not m.text:
            continue
        lines.append(f"{role}: {m.text[:limit_chars]}")
    return "\n".join(lines)


async def compact_history(
    messages: list[AnyMessage], summary: str, model: Runnable[Any, Any], *, keep_last: int, trigger: int
) -> dict[str, Any]:
    """Summarize + drop everything but the last ``keep_last`` messages once ``trigger`` is exceeded."""
    if len(messages) <= trigger:
        return {}
    old, _recent = messages[:-keep_last], messages[-keep_last:]
    response = await model.ainvoke(SUMMARIZER.format(summary=summary or "(none)", messages=render_messages(old)))
    return {
        "summary": response.text.strip(),
        "messages": [RemoveMessage(id=m.id) for m in old if m.id],
    }


async def recall_memories(store: BaseStore, customer_id: str, query: str, *, limit: int = 5) -> list[str]:
    items = await store.asearch(memory_namespace(customer_id), query=query or None, limit=limit)
    return [str(item.value.get("fact", "")) for item in items if item.value.get("fact")]


async def extract_and_store_memories(
    store: BaseStore,
    extractor: Runnable[Any, Any],
    *,
    customer_id: str,
    conversation_id: str,
    turn_text: str,
    known: list[str],
    max_new: int = 3,
) -> list[str]:
    """``extractor`` is a structured-output runnable returning ``MemoryUpdate``."""
    update: MemoryUpdate = await extractor.ainvoke(  # type: ignore[assignment]
        MEMORY_EXTRACTOR.format(
            company="Voltwise", known="\n".join(f"- {k}" for k in known) or "(nothing)", turn=turn_text
        )
    )
    known_lower = {k.lower() for k in known}
    saved: list[str] = []
    for fact in update.facts[:max_new]:
        fact = fact.strip()
        if len(fact) < 8 or fact.lower() in known_lower:
            continue
        await store.aput(
            memory_namespace(customer_id),
            uuid.uuid4().hex,
            {"fact": fact, "source": conversation_id, "created_at": datetime.now(UTC).isoformat()},
        )
        saved.append(fact)
    return saved
