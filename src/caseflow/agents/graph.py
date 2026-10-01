"""The CaseFlow support graph - supervisor/router multi-agent orchestration.

::

    START
      |
    ingress ──(blocked)──────────────────────────────────────────┐
      |                                                           |
    load_context   profile (MCP) · history compaction · memories  |
      |                                                           |
    triage         structured routing decision (fast model)       |
      |──(needs human)────────────────────────────┐               |
      |──(no tasks: greeting / clarify)──┐        |               |
      | Send() fan-out, in parallel      |        |               |
      ├─> knowledge_agent (CRAG subgraph)|        |               |
      ├─> orders_agent    (create_agent + MCP)    |               |
      └─> returns_agent   (create_agent + MCP + HITL refund gate) |
                 |                       |        |               |
             aggregate ──(all failed)────┼──> escalate (ticket + human handoff interrupt)
                 |                       |        |               |
             synthesize <────────────────┘        |               |
                 |    ^ (revise, max N)           |               |
             output_guard ──(escalate)────────────┘               |
                 |                                                |
             finalize  <──────────────────────────────────────────┘
                 |
             remember   long-term memory extraction (Store)
                 |
                END

Patterns on display: supervisor routing with structured output, dynamic parallel
fan-out with ``Send`` + a merging reducer, subgraphs (a custom CRAG graph and
``create_agent`` graphs), ``Command``-based routing, three kinds of human-in-the-loop
(``interrupt``), per-node retry/timeout/error-handler policies (LangGraph 1.2), and
short- + long-term memory.
"""

from __future__ import annotations

import contextlib
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime, get_runtime
from langgraph.types import Command, RetryPolicy, Send, TimeoutPolicy, default_retry_on, interrupt

from caseflow.agents.context import SupportContext
from caseflow.agents.guardrails import REFUSAL_MESSAGE, InputVerdict, mask_card_numbers, output_issues, screen_input
from caseflow.agents.memory import compact_history, extract_and_store_memories, recall_memories, render_messages
from caseflow.agents.specialists import GraphDeps, new_task_id, run_knowledge, run_tool_agent
from caseflow.agents.state import (
    GuardVerdict,
    MemoryUpdate,
    SpecialistInput,
    SupportState,
    TriageDecision,
    specialist_result,
)
from caseflow.observability import AGENT_RUNS, ESCALATIONS, GUARDRAIL_BLOCKS, HITL_INTERRUPTS, get_logger
from caseflow.prompts import OUTPUT_GUARD, REVISION_NOTE, SYNTHESIZER, TRIAGE

log = get_logger(__name__)

SPECIALIST_NODES = {"knowledge": "knowledge_agent", "orders": "orders_agent", "returns": "returns_agent"}


def _llm_retry_on(exc: Exception) -> bool:
    return isinstance(exc, OutputParserException) or default_retry_on(exc)


def _last_human(state: SupportState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.text
    return ""


def _profile_block(customer: dict[str, Any] | None) -> str:
    c = customer or {}
    return (
        f"name: {c.get('name', 'unknown')}; tier: {c.get('tier', 'standard')}; country: {c.get('country', '?')}; "
        f"VoltPoints: {c.get('voltpoints', '?')}; store credit: ${c.get('store_credit', 0)}"
    )


def format_findings(results: list[dict[str, Any]]) -> str:
    blocks = []
    for r in results:
        head = f"### {r['agent']} specialist (status: {r['status']})\nTask: {r['request']}"
        body = r.get("answer") or ""
        if r.get("citations"):
            body += "\nSources: " + "; ".join(f"[{c['n']}] {c['title']}" for c in r["citations"])
        if r.get("error"):
            body += f"\nError: {r['error']}"
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks) or "(no findings)"


def build_support_graph(
    deps: GraphDeps, *, checkpointer: Any = None, store: Any = None
) -> CompiledStateGraph[SupportState, SupportContext, SupportState, SupportState]:
    s = deps.settings
    company = s.agent.company_name

    # ---- 1. ingress: deterministic guardrails -------------------------------------------
    async def ingress(state: SupportState) -> Command[Literal["load_context", "finalize"]]:
        last = state["messages"][-1]
        text = last.text
        update: dict[str, Any] = {
            # Reset per-turn fields; `None` tells the collect_results reducer to clear.
            "specialist_results": None,
            "triage": None,
            "draft": "",
            "guard": None,
            "revisions": 0,
            "outcome": None,
            "ticket": None,
            "turn": state.get("turn", 0) + 1,
        }
        masked = mask_card_numbers(text)
        if masked != text:
            # Same message id => add_messages *replaces* it: the raw card number is never checkpointed.
            update["messages"] = [HumanMessage(content=masked, id=last.id)]
        verdict = screen_input(masked)
        if not verdict.blocked and deps.prompt_guard is not None:
            score = await deps.prompt_guard.score(masked)
            if score >= s.agent.prompt_guard_threshold:
                verdict = InputVerdict(
                    blocked=True, reason="prompt_guard", signals=[f"score={score:.3f}"], message=REFUSAL_MESSAGE
                )
        if verdict.blocked:
            GUARDRAIL_BLOCKS.labels(stage="input", reason=verdict.reason or "unknown").inc()
            log.info("input_blocked", reason=verdict.reason, signals=verdict.signals[:3])
            update |= {"draft": verdict.message, "outcome": "blocked"}
            return Command(update=update, goto="finalize")
        return Command(update=update, goto="load_context")

    # ---- 2. context: profile, history compaction, long-term memory -----------------------
    async def load_context(state: SupportState, runtime: Runtime[SupportContext]) -> dict[str, Any]:
        ctx = runtime.context
        update: dict[str, Any] = {}
        if not state.get("customer"):
            try:
                update["customer"] = await deps.toolkit.call(ctx.customer_id, "commerce", "get_customer_profile", {})
            except Exception as exc:  # degrade gracefully: the conversation can continue without it
                log.warning("profile_unavailable", error=type(exc).__name__)
                update["customer"] = {"name": "there", "tier": "standard"}
        update |= await compact_history(
            state["messages"],
            state.get("summary", ""),
            deps.models.chat("fast"),
            keep_last=s.agent.history_window,
            trigger=s.agent.summarize_after_messages,
        )
        if runtime.store is not None and s.agent.enable_long_term_memory:
            update["memories"] = await recall_memories(runtime.store, ctx.customer_id, _last_human(state))
        return update

    # ---- 3. triage: the supervisor ----------------------------------------------------------
    async def triage(state: SupportState) -> dict[str, Any]:
        router = deps.models.structured("fast", TriageDecision)
        system = TRIAGE.format(
            company=company,
            customer=_profile_block(state.get("customer")),
            memories="\n".join(f"- {m}" for m in state.get("memories", [])) or "(none)",
            summary=state.get("summary") or "(none)",
        )
        recent = [m for m in state["messages"][-s.agent.history_window :] if isinstance(m, HumanMessage | AIMessage)]
        decision: TriageDecision = await router.ainvoke([SystemMessage(content=system), *recent])  # type: ignore[assignment]
        if decision.injection_suspected:
            GUARDRAIL_BLOCKS.labels(stage="triage", reason="injection_suspected").inc()
            # Suspicious turn: answer, but never hand it to tool-using specialists.
            decision.tasks = [t for t in decision.tasks if t.agent == "knowledge"]
        log.info(
            "triage", intents=decision.intents, tasks=[t.agent for t in decision.tasks], human=decision.needs_human
        )
        return {"triage": decision.model_dump()}

    def route_after_triage(state: SupportState) -> list[Send] | str:
        t = TriageDecision.model_validate(state["triage"] or {})
        if t.needs_human:
            return "escalate"
        if not t.tasks:
            return "synthesize"
        summary = state.get("summary", "")
        recent = render_messages(state["messages"][-6:-1]) if len(state["messages"]) > 1 else ""
        convo = "\n".join(x for x in (summary, recent) if x)
        return [
            Send(
                SPECIALIST_NODES[task.agent],
                SpecialistInput(
                    task_id=new_task_id(),
                    agent=task.agent,
                    request=task.request,
                    customer=state.get("customer", {}),
                    summary=convo,
                    memories=state.get("memories", []),
                ),
            )
            for task in t.tasks
        ]

    # ---- 4. specialists (each receives only its Send payload - context isolation) ----------
    async def knowledge_agent(state: SpecialistInput, config: RunnableConfig) -> dict[str, Any]:
        return {"specialist_results": [await run_knowledge(deps, state, config)]}

    async def orders_agent(state: SpecialistInput, config: RunnableConfig) -> dict[str, Any]:
        runtime = get_runtime(SupportContext)
        return {"specialist_results": [await run_tool_agent(deps, state, config, runtime.context)]}

    async def returns_agent(state: SpecialistInput, config: RunnableConfig) -> dict[str, Any]:
        runtime = get_runtime(SupportContext)
        return {"specialist_results": [await run_tool_agent(deps, state, config, runtime.context)]}

    async def specialist_failed(payload: SpecialistInput, error: NodeError) -> dict[str, Any]:
        """LangGraph 1.2 node error handler: a crashed specialist becomes an error *result*
        instead of failing the whole run, so the other specialists' work still reaches the customer.

        Must be async: error handlers run as internal nodes and inherit the graph's default
        timeout, and LangGraph only allows timeouts on async nodes."""
        log.error("specialist_failed", node=error.node, error=repr(error.error)[:300])
        return {
            "specialist_results": [
                specialist_result(
                    task_id=payload.get("task_id", "?"),
                    agent=payload.get("agent", error.node),
                    request=payload.get("request", ""),
                    answer="",
                    status="error",
                    error=f"{type(error.error).__name__}: internal failure",
                )
            ]
        }

    # ---- 5. fan-in -----------------------------------------------------------------------------
    async def aggregate(state: SupportState) -> Command[Literal["synthesize", "escalate"]]:
        results = state.get("specialist_results", [])
        tool_results = [r for r in results if r["agent"] != "knowledge"]
        if (
            tool_results
            and all(r["status"] == "error" for r in tool_results)
            and not any(r["status"] == "ok" for r in results)
        ):
            return Command(
                update={"triage": {**(state.get("triage") or {}), "escalation_reason": "Automated tools failed"}},
                goto="escalate",
            )
        return Command(goto="synthesize")

    # ---- 6. answer synthesis (the only node whose tokens stream to the customer) ---------------
    async def synthesize(state: SupportState) -> dict[str, Any]:
        results = state.get("specialist_results", [])
        triage_out = state.get("triage") or {}
        if not results:
            reply = triage_out.get("direct_reply") or (
                f"I can help with {company} orders, deliveries, returns, refunds, products and policies. "
                "What can I do for you?"
            )
            return {"draft": reply}
        guard = state.get("guard") or {}
        note = (
            REVISION_NOTE.format(issues="\n".join(f"- {i}" for i in guard.get("issues", [])))
            if guard.get("verdict") == "revise"
            else ""
        )
        system = SYNTHESIZER.format(
            company=company,
            revision_note=note,
            customer=_profile_block(state.get("customer")),
            findings=format_findings(results),
        )
        recent = [m for m in state["messages"][-4:] if isinstance(m, HumanMessage | AIMessage) and m.text]
        response = await deps.models.chat("smart").ainvoke([SystemMessage(content=system), *recent])
        return {"draft": response.text.strip(), "revisions": state.get("revisions", 0) + (1 if note else 0)}

    # ---- 7. output guard: deterministic checks + LLM QA -----------------------------------------
    async def output_guard(state: SupportState) -> Command[Literal["synthesize", "finalize", "escalate"]]:
        results = state.get("specialist_results", [])
        draft = state.get("draft", "")
        if not results or not s.agent.enable_output_guard:
            return Command(goto="finalize")
        issues = output_issues(draft)
        if issues:
            verdict = GuardVerdict(verdict="revise", issues=issues)
        else:
            reviewer = deps.models.structured("fast", GuardVerdict)
            verdict = await reviewer.ainvoke(  # type: ignore[assignment]
                OUTPUT_GUARD.format(
                    company=company, question=_last_human(state), findings=format_findings(results), draft=draft
                )
            )
        update = {"guard": verdict.model_dump()}
        if verdict.verdict == "pass":
            return Command(update=update, goto="finalize")
        GUARDRAIL_BLOCKS.labels(stage="output", reason=verdict.verdict).inc()
        if verdict.verdict == "revise" and state.get("revisions", 0) < s.agent.max_revisions:
            return Command(update=update, goto="synthesize")
        if verdict.verdict == "escalate":
            return Command(update=update, goto="escalate")
        # Revisions exhausted: deterministic leaks are never sent; LLM-judged nits are.
        if output_issues(draft):
            return Command(update=update, goto="escalate")
        return Command(update=update, goto="finalize")

    # ---- 8. human handoff ----------------------------------------------------------------------
    async def escalate(state: SupportState, runtime: Runtime[SupportContext]) -> dict[str, Any]:
        ctx = runtime.context
        t = state.get("triage") or {}
        reason = t.get("escalation_reason") or "The assistant could not resolve the request"
        priority = {"urgent": "urgent", "high": "high"}.get(
            t.get("urgency", ""), "high" if t.get("sentiment") == "angry" else "normal"
        )
        ticket: dict[str, Any] | None = None
        try:
            # Idempotent per conversation: safe even though this node re-runs when resumed.
            ticket = await deps.toolkit.call(
                ctx.customer_id,
                "helpdesk",
                "create_ticket",
                {
                    "subject": f"Escalation: {reason[:150]}",
                    "description": render_messages(state["messages"][-8:]),
                    "category": "complaint" if "complaint" in t.get("intents", []) else "other",
                    "priority": priority,
                    "conversation_id": ctx.conversation_id,
                },
            )
        except Exception as exc:
            log.warning("ticket_create_failed", error=type(exc).__name__)
        ESCALATIONS.labels(reason="triage" if t.get("needs_human") else "automation").inc()
        HITL_INTERRUPTS.labels(kind="human_handoff").inc()
        # Pause the run until a human agent picks the case up. Resume values:
        #   {"action": "reply", "message": "...", "agent_name": "Jordan"}  -> live answer
        #   {"action": "defer"}                                             -> async follow-up
        decision = interrupt(
            {
                "type": "human_handoff",
                "reason": reason,
                "priority": priority,
                "ticket": ticket,
                "customer": state.get("customer", {}),
                "last_message": _last_human(state),
            }
        )
        ticket_id = (ticket or {}).get("ticket_id", "pending")
        if isinstance(decision, dict) and decision.get("action") == "reply" and decision.get("message"):
            draft = str(decision["message"])
            name = str(decision.get("agent_name") or "Support specialist")
            with contextlib.suppress(Exception):  # best effort: the reply matters more than the note
                await deps.toolkit.call(
                    ctx.customer_id, "helpdesk", "add_ticket_comment",
                    {"ticket_id": ticket_id, "comment": draft, "author": "agent"},
                )  # fmt: skip
            return {"draft": f"{draft}\n\n- {name}, {company} Support", "outcome": "escalated", "ticket": ticket}
        target = (ticket or {}).get("first_response_target", "1 business day")
        return {
            "draft": (
                f"I've passed your case to a {company} specialist (ticket {ticket_id}). "
                f"You'll hear back within {target}, and you won't need to repeat anything."
            ),
            "outcome": "escalated",
            "ticket": ticket,
        }

    # ---- 9. finalize & remember -----------------------------------------------------------------
    async def finalize(state: SupportState, runtime: Runtime[SupportContext]) -> dict[str, Any]:
        ctx = runtime.context
        outcome = state.get("outcome") or "resolved"
        draft = state.get("draft") or "Sorry, something went wrong on my side. Please try again."
        AGENT_RUNS.labels(outcome=outcome).inc()
        if outcome != "blocked":
            try:  # analytics (automation rate, intent mix) - never block the reply on it
                await deps.toolkit.call(
                    ctx.customer_id,
                    "helpdesk",
                    "log_case",
                    {
                        "conversation_id": ctx.conversation_id,
                        "intents": (state.get("triage") or {}).get("intents", []),
                        "outcome": "escalated" if outcome == "escalated" else "resolved",
                        "summary": draft[:500],
                    },
                )
            except Exception as exc:
                log.warning("case_log_failed", error=type(exc).__name__)
        return {"messages": [AIMessage(content=draft, name="caseflow")], "outcome": outcome}

    async def remember(state: SupportState, runtime: Runtime[SupportContext]) -> dict[str, Any]:
        if runtime.store is None or not s.agent.enable_long_term_memory or state.get("outcome") == "blocked":
            return {}
        turn = render_messages(state["messages"][-2:])
        try:
            await extract_and_store_memories(
                runtime.store,
                deps.models.structured("fast", MemoryUpdate),
                customer_id=runtime.context.customer_id,
                conversation_id=runtime.context.conversation_id,
                turn_text=turn,
                known=state.get("memories", []),
            )
        except Exception as exc:  # memory is an enhancement - never fail the turn for it
            log.warning("memory_extraction_failed", error=type(exc).__name__, detail=str(exc)[:300])
        return {}

    # ---- wiring ---------------------------------------------------------------------------------
    llm_retry = RetryPolicy(max_attempts=2, initial_interval=1.0, retry_on=_llm_retry_on)
    node_timeout = TimeoutPolicy(run_timeout=s.agent.node_timeout_s)
    builder = StateGraph(SupportState, context_schema=SupportContext)
    builder.set_node_defaults(timeout=node_timeout)

    builder.add_node("ingress", ingress, destinations=("load_context", "finalize"))
    builder.add_node("load_context", load_context, retry_policy=llm_retry)
    builder.add_node("triage", triage, retry_policy=llm_retry)
    # Tool-using specialists get NO node-level retry: a retry would replay side effects
    # (refunds!). Their resilience lives inside the agent (model/tool retry middleware).
    builder.add_node("knowledge_agent", knowledge_agent, input_schema=SpecialistInput, error_handler=specialist_failed)
    # Send() targets receive a SpecialistInput payload, not the graph state: declare it.
    specialist_timeout = TimeoutPolicy(run_timeout=s.agent.node_timeout_s * 2)
    for name, fn in (("orders_agent", orders_agent), ("returns_agent", returns_agent)):
        builder.add_node(
            name, fn, input_schema=SpecialistInput, error_handler=specialist_failed, timeout=specialist_timeout
        )
    builder.add_node("aggregate", aggregate, destinations=("synthesize", "escalate"))
    builder.add_node("synthesize", synthesize, retry_policy=llm_retry)
    builder.add_node(
        "output_guard", output_guard, retry_policy=llm_retry, destinations=("synthesize", "finalize", "escalate")
    )
    builder.add_node("escalate", escalate)
    builder.add_node("finalize", finalize)
    builder.add_node("remember", remember)

    builder.add_edge(START, "ingress")
    builder.add_edge("load_context", "triage")
    builder.add_conditional_edges("triage", route_after_triage, [*SPECIALIST_NODES.values(), "escalate", "synthesize"])
    for node in SPECIALIST_NODES.values():
        builder.add_edge(node, "aggregate")
    builder.add_edge("synthesize", "output_guard")
    builder.add_edge("escalate", "finalize")
    builder.add_edge("finalize", "remember")
    builder.add_edge("remember", END)

    return builder.compile(checkpointer=checkpointer, store=store, name="caseflow_support")
