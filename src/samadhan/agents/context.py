"""Runtime context: per-request, read-only dependencies for a graph run.

LangGraph separates three kinds of data - knowing which is which is the key to
clean agent design:

* **State** (``SupportState``) - what the graph *computes and remembers*; it is
  checkpointed after every step and changes as nodes run.
* **Context** (this class) - *who/what* the run is for (customer, channel). Passed
  as ``graph.ainvoke(..., context=SupportContext(...))``; immutable during the run,
  not checkpointed, available to nodes, tools and middleware via ``runtime.context``.
* **Services** (``GraphDeps``) - long-lived infrastructure (models, retriever, MCP
  toolkit) captured when the graph is built.

Putting ``customer_id`` in context (not state, not a prompt) means no node and no
model can change whose data the run operates on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SupportContext:
    customer_id: str
    conversation_id: str
    channel: Literal["web", "email", "api", "a2a", "cli", "eval"] = "web"
    locale: str = "en-US"
