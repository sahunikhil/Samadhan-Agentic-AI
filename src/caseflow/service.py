"""Application service: the one API every front door (HTTP, CLI, evals) uses to talk to the graph.

Responsibilities
----------------
* **Thread ownership** - thread IDs are ``<customer_id>--<random>`` so ownership is
  checkable without a lookup table; customers can only touch their own threads.
* **Streaming** - translates LangGraph's typed ``astream(version="v2")`` parts into
  a small, stable event vocabulary for the UI (status, tool, token, interrupt, final).
* **Human-in-the-loop routing** - classifies every pending ``interrupt`` by *who*
  must answer it (customer / supervisor / human agent) and validates decisions.
* **Signed approvals** - a supervisor's "approve" on a refund becomes an *edit*
  decision that injects an ``approval_code`` JWT into the tool call; the MCP server
  verifies it independently.
* **Graceful shutdown** - every run gets a ``RunControl``; on SIGTERM we request a
  drain so in-flight runs stop at a superstep boundary with a resumable checkpoint.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.types import Command, Interrupt, StreamMode

from caseflow.agents.context import SupportContext
from caseflow.bootstrap import Container
from caseflow.cost import ModelPrice, TurnCostTracker
from caseflow.mcp_servers.auth import mint_approval_code
from caseflow.observability import TURN_COST, TURN_OVER_BUDGET, get_logger, tracing_callbacks

log = get_logger(__name__)

InterruptKind = Literal["refund_approval", "customer_confirmation", "human_handoff", "unknown"]
Audience = Literal["customer", "supervisor", "agent"]
Actor = Literal["customer", "supervisor", "system"]

STREAM_MODES: list[StreamMode] = ["updates", "messages", "custom"]

NODE_LABELS = {
    "ingress": "Checking your message",
    "load_context": "Loading your account",
    "triage": "Understanding your request",
    "knowledge_agent": "Searching the help center",
    "orders_agent": "Looking into your orders",
    "returns_agent": "Checking returns & refunds",
    "aggregate": "Combining results",
    "synthesize": "Writing a reply",
    "output_guard": "Double-checking the answer",
    "escalate": "Connecting you with a specialist",
    "finalize": "Done",
    "remember": "Updating customer memory",
}


@dataclass
class PendingAction:
    interrupt_id: str
    kind: InterruptKind
    audience: Audience
    title: str
    detail: dict[str, Any]


@dataclass
class TurnResult:
    thread_id: str
    reply: str | None
    outcome: str | None
    pending: list[PendingAction] = field(default_factory=list)
    triage: dict[str, Any] | None = None
    specialist_results: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None


class ThreadAccessError(PermissionError):
    pass


def new_thread_id(customer_id: str) -> str:
    return f"{customer_id}--{uuid.uuid4().hex[:12]}"


def owner_of(thread_id: str) -> str:
    return thread_id.split("--", 1)[0]


def classify_interrupt(item: Interrupt) -> PendingAction:
    value = item.value if isinstance(item.value, dict) else {"value": item.value}
    if value.get("type") == "mcp_elicitation":
        req = (value.get("requests") or [{}])[0]
        return PendingAction(item.id, "customer_confirmation", "customer", req.get("message", "Please confirm"), value)
    if value.get("type") == "human_handoff":
        return PendingAction(item.id, "human_handoff", "agent", f"Human handoff: {value.get('reason', '')}", value)
    if "action_requests" in value:
        titles = [a.get("description", a.get("name", "")) for a in value["action_requests"]]
        return PendingAction(item.id, "refund_approval", "supervisor", " | ".join(titles), value)
    return PendingAction(item.id, "unknown", "supervisor", "Pending input", value)


class SupportService:
    def __init__(self, container: Container) -> None:
        self.c = container
        self._active: dict[str, RunControl] = {}

    # ---- config ----------------------------------------------------------------------
    def _config(self, thread_id: str, *, run_name: str = "support_turn") -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": tracing_callbacks(self.c.settings),
            "metadata": {"customer_id": owner_of(thread_id), "thread_id": thread_id},
            "run_name": run_name,
            "recursion_limit": 50,
        }

    @staticmethod
    def check_access(thread_id: str, customer_id: str | None) -> None:
        """customer_id=None means a staff caller (supervisor/agent) with access to all threads."""
        if customer_id is not None and owner_of(thread_id) != customer_id:
            raise ThreadAccessError("This conversation belongs to another account.")

    # ---- streaming turns ---------------------------------------------------------------
    async def stream_turn(
        self, *, customer_id: str, message: str, thread_id: str | None = None, channel: str = "web"
    ) -> AsyncIterator[dict[str, Any]]:
        thread_id = thread_id or new_thread_id(customer_id)
        self.check_access(thread_id, customer_id)
        payload = {"messages": [HumanMessage(content=message, id=f"h-{uuid.uuid4().hex[:10]}")]}
        async for event in self._run(payload, thread_id, channel):
            yield event

    async def stream_resume(
        self, *, thread_id: str, decisions: dict[str, Any], actor: Actor, actor_name: str, customer_id: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        self.check_access(thread_id, customer_id)
        resume = await self._build_resume(thread_id, decisions, actor=actor, actor_name=actor_name)
        async for event in self._run(Command(resume=resume), thread_id, "web"):
            yield event

    async def _run(self, payload: Any, thread_id: str, channel: str) -> AsyncIterator[dict[str, Any]]:
        customer_id = owner_of(thread_id)
        context = SupportContext(customer_id=customer_id, conversation_id=thread_id, channel=channel)  # type: ignore[arg-type]
        control = RunControl()
        self._active[thread_id] = control
        prices = {m: ModelPrice(**p) for m, p in self.c.settings.observability.prices.items()}
        cost = TurnCostTracker(prices)  # sees every nested LLM call of this run
        config = self._config(thread_id)
        config["callbacks"] = [*config["callbacks"], cost]  # type: ignore[misc]
        yield {"type": "run_started", "thread_id": thread_id}
        try:
            async for part in self.c.graph.astream(
                payload,
                config,
                context=context,
                stream_mode=STREAM_MODES,
                subgraphs=True,
                version="v2",
                control=control,
            ):
                for event in self._translate(part):
                    yield event
        except GraphDrained:
            yield {"type": "error", "message": "The service is restarting. Your conversation was saved - please retry."}
            return
        except Exception as exc:
            log.exception("run_failed", thread_id=thread_id)
            yield {"type": "error", "message": f"Something went wrong ({type(exc).__name__}). Please try again."}
            return
        finally:
            self._active.pop(thread_id, None)
            usage = self._account(cost, thread_id)
        view = await self.thread_view(thread_id)
        if view["pending"]:
            yield {"type": "interrupt", "thread_id": thread_id, "pending": view["pending"], "usage": usage}
        else:
            yield {
                "type": "final",
                "thread_id": thread_id,
                "reply": view["last_reply"],
                "outcome": view["outcome"],
                "usage": usage,
            }

    def _account(self, cost: TurnCostTracker, thread_id: str) -> dict[str, Any]:
        usage = cost.summary()
        TURN_COST.observe(usage["cost_usd"])
        if usage["cost_usd"] > self.c.settings.agent.turn_budget_usd:
            TURN_OVER_BUDGET.inc()
            log.warning("turn_over_budget", thread_id=thread_id, cost_usd=usage["cost_usd"], calls=usage["llm_calls"])
        return usage

    def _translate(self, part: Any) -> list[dict[str, Any]]:
        """Map a typed v2 ``StreamPart`` (``type``/``ns``/``data``) to UI events."""
        kind, ns, data = part["type"], part["ns"], part["data"]
        out: list[dict[str, Any]] = []
        if kind == "messages":
            chunk, meta = data
            # Only the customer-facing reply streams token by token; internal LLM calls don't.
            if not ns and meta.get("langgraph_node") == "synthesize" and isinstance(chunk.content, str | list):
                text = chunk.text
                if text:
                    out.append({"type": "token", "text": text})
        elif kind == "updates":
            if ns:
                agent = ns[0].split(":", 1)[0]
                for node, update in data.items():
                    if node in ("model", "tools"):
                        out.append({"type": "agent_step", "agent": agent, "step": node})
                    elif node == "cache_lookup" and isinstance(update, dict) and update.get("cached"):
                        out.append({"type": "agent_step", "agent": agent, "step": "answered from semantic cache"})
                return out
            for node, update in data.items():
                if node.startswith("__"):
                    continue
                out.append({"type": "status", "node": node, "label": NODE_LABELS.get(node, node)})
                if node == "triage" and isinstance(update, dict) and update.get("triage"):
                    t = update["triage"]
                    out.append(
                        {
                            "type": "triage",
                            "intents": t.get("intents", []),
                            "agents": [x["agent"] for x in t.get("tasks", [])],
                            "needs_human": t.get("needs_human", False),
                        }
                    )
                if node == "output_guard" and isinstance(update, dict):
                    guard = update.get("guard") or {}
                    if guard.get("verdict") == "revise":
                        out.append({"type": "draft_reset", "issues": guard.get("issues", [])})
        elif kind == "custom" and isinstance(data, dict):
            out.append({"type": "tool", **data})
        return out

    # ---- non-streaming convenience (CLI, evals, tests) ---------------------------------------
    async def run_turn(
        self, *, customer_id: str, message: str, thread_id: str | None = None, channel: str = "api"
    ) -> TurnResult:
        events = [
            e
            async for e in self.stream_turn(
                customer_id=customer_id, message=message, thread_id=thread_id, channel=channel
            )
        ]
        return await self._result(events)

    async def resume(
        self, *, thread_id: str, decisions: dict[str, Any], actor: Actor, actor_name: str, customer_id: str | None
    ) -> TurnResult:
        events = [
            e
            async for e in self.stream_resume(
                thread_id=thread_id, decisions=decisions, actor=actor, actor_name=actor_name, customer_id=customer_id
            )
        ]
        return await self._result(events)

    async def _result(self, events: list[dict[str, Any]]) -> TurnResult:
        thread_id = next(e["thread_id"] for e in events if e["type"] == "run_started")
        view = await self.thread_view(thread_id)
        errors = [e["message"] for e in events if e["type"] == "error"]
        return TurnResult(
            thread_id=thread_id,
            reply=view["last_reply"] if not view["pending"] else None,
            outcome=view["outcome"] if not errors else "error",
            pending=[PendingAction(**p) for p in view["pending"]],
            triage=view["triage"],
            specialist_results=view["specialist_results"],
            events=events,
            usage=next((e["usage"] for e in reversed(events) if "usage" in e), None),
        )

    # ---- state inspection --------------------------------------------------------------------
    async def thread_view(self, thread_id: str) -> dict[str, Any]:
        snapshot = await self.c.graph.aget_state(self._config(thread_id))
        values = snapshot.values or {}
        messages = values.get("messages", [])
        pending = [classify_interrupt(i).__dict__ for i in snapshot.interrupts]
        last_reply = next((m.text for m in reversed(messages) if isinstance(m, AIMessage)), None)
        return {
            "thread_id": thread_id,
            "messages": [
                {"role": "customer" if isinstance(m, HumanMessage) else "assistant", "content": m.text, "id": m.id}
                for m in messages
                if isinstance(m, HumanMessage | AIMessage) and m.text
            ],
            "summary": values.get("summary", ""),
            "customer": values.get("customer", {}),
            "outcome": values.get("outcome"),
            "triage": values.get("triage"),
            "specialist_results": values.get("specialist_results", []),
            "ticket": values.get("ticket"),
            "pending": pending,
            "last_reply": last_reply,
            "next": list(snapshot.next),
        }

    async def history(self, thread_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Checkpoint timeline (time travel / debugging): every superstep of the thread."""
        out = []
        async for snap in self.c.graph.aget_state_history(self._config(thread_id), limit=limit):
            out.append(
                {
                    "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
                    "step": snap.metadata.get("step") if snap.metadata else None,
                    "source": snap.metadata.get("source") if snap.metadata else None,
                    "next": list(snap.next),
                    "created_at": snap.created_at,
                    "writes": list(cast(dict[str, Any], (snap.metadata or {}).get("writes") or {})),
                }
            )
        return out

    # ---- human-in-the-loop -------------------------------------------------------------------
    async def _build_resume(
        self, thread_id: str, decisions: dict[str, Any], *, actor: Actor, actor_name: str
    ) -> dict[str, Any]:
        snapshot = await self.c.graph.aget_state(self._config(thread_id))
        pending = {i.id: (i, classify_interrupt(i)) for i in snapshot.interrupts}
        if not pending:
            raise ValueError("There is nothing waiting for input on this conversation.")
        resume: dict[str, Any] = {}
        for interrupt_id, decision in decisions.items():
            if interrupt_id not in pending:
                raise ValueError(f"Unknown or already-resolved interrupt {interrupt_id}.")
            item, action = pending[interrupt_id]
            allowed: dict[Actor, set[Audience]] = {
                "customer": {"customer"},  # only confirms their own actions
                "supervisor": {"supervisor", "agent"},  # never answers *for* the customer
                "system": {"customer", "supervisor", "agent"},  # eval harness / tests only
            }
            if action.audience not in allowed[actor]:
                raise ThreadAccessError(f"A {actor} cannot answer a '{action.kind}' request.")
            if action.kind == "refund_approval":
                resume[interrupt_id] = self._approval_resume(thread_id, item.value, decision, actor_name)
            elif action.kind == "customer_confirmation":
                resume[interrupt_id] = _confirmation_resume(item.value, decision)
            else:
                resume[interrupt_id] = decision
        return resume

    def _approval_resume(self, thread_id: str, value: dict[str, Any], decision: Any, approver: str) -> dict[str, Any]:
        """Turn a supervisor decision into HumanInTheLoopMiddleware decisions.

        ``approve`` -> ``edit`` that adds a short-lived signed ``approval_code`` scoped to this
        customer, order and amount. ``reject`` passes through with the reviewer's reason."""
        choice = decision.get("decision") if isinstance(decision, dict) else str(decision)
        reason = decision.get("message") if isinstance(decision, dict) else None
        out = []
        for req in value.get("action_requests", []):
            if choice == "approve" and req.get("name") == "issue_refund":
                args = dict(req.get("args", {}))
                args["approval_code"] = mint_approval_code(
                    self.c.settings.mcp,
                    customer_id=owner_of(thread_id),
                    order_id=str(args.get("order_id", "")),
                    max_amount=float(args.get("amount", 0)),
                    approver=approver,
                )
                out.append({"type": "edit", "edited_action": {"name": req["name"], "args": args}})
            elif choice == "approve":
                out.append({"type": "approve"})
            else:
                out.append({"type": "reject", "message": reason or "A support specialist declined this refund."})
        log.info("hitl_decision", thread_id=thread_id, decision=choice, approver=approver)
        return {"decisions": out}

    # ---- lifecycle ---------------------------------------------------------------------------
    def drain_all(self, reason: str = "shutdown") -> int:
        for control in self._active.values():
            control.request_drain(reason)
        return len(self._active)


def _confirmation_resume(value: dict[str, Any], decision: Any) -> dict[str, Any]:
    """Customer answer to an MCP elicitation -> ``{"responses": {key: ElicitResult-like}}``."""
    requests = value.get("requests", [])
    accepted = decision.get("accept") if isinstance(decision, dict) else bool(decision)
    responses: dict[str, Any] = {}
    for req in requests:
        if not accepted:
            responses[req["key"]] = {"action": "decline"}
            continue
        props = (req.get("requested_schema") or {}).get("properties", {})
        content = decision.get("content") if isinstance(decision, dict) and decision.get("content") else None
        if content is None:  # a yes/no confirmation form: set every boolean field to True
            content = {k: True for k, spec in props.items() if spec.get("type") == "boolean"}
        responses[req["key"]] = {"action": "accept", "content": content}
    return {"responses": responses}
