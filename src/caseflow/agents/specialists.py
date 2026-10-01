"""Specialist agents (LangChain ``create_agent`` + MCP tools + middleware).

Middleware order matters: the list is applied outside-in, so the first entry wraps
all the others. Our stack, from outermost to innermost::

    ToolAudit            observe every tool call (sees the final outcome, incl. errors)
    ModelCallLimit       hard budget: max LLM calls per run  -> no runaway loops / bills
    ToolCallLimit        hard budget: max tool calls per run
    PII                  mask card numbers in inputs and tool results
    RuntimeContextPrompt inject date/channel into the system prompt at call time
    ModelFallback        primary model fails -> try the next provider
    ModelRetry           transient model errors -> exponential backoff (per model)
    ToolError            unexpected tool exceptions -> readable error ToolMessage
    ToolRetry            transient tool/transport errors -> retry once
    HumanInTheLoop       (returns only) pause large refunds for a human decision

Agents are built per customer (the MCP tools hold that customer's token). That
costs a few milliseconds - graph compilation is cheap - and guarantees tools can
never leak between customers.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from caseflow.agents.context import SupportContext
from caseflow.agents.guardrails import PromptGuardClassifier
from caseflow.agents.middleware import RuntimeContextPromptMiddleware, ToolAuditMiddleware, refund_approval_gate
from caseflow.agents.state import SpecialistInput, specialist_result
from caseflow.agents.toolkit import (
    ORDERS_HELPDESK_TOOLS,
    ORDERS_SCOPES,
    ORDERS_TOOLS,
    RETURNS_SCOPES,
    RETURNS_TOOLS,
    MCPToolkit,
)
from caseflow.config import Settings
from caseflow.llm import ModelRegistry
from caseflow.prompts import ORDERS_AGENT, RETURNS_AGENT
from caseflow.rag.retriever import HybridRetriever
from caseflow.resilience import CircuitBreakerMiddleware


@dataclass
class GraphDeps:
    """Long-lived services captured by the graph at build time (dependency injection)."""

    settings: Settings
    models: ModelRegistry
    retriever: HybridRetriever
    toolkit: MCPToolkit
    knowledge_graph: CompiledStateGraph[Any, Any, Any, Any]
    prompt_guard: PromptGuardClassifier | None = None


_TOKEN_LIKE = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")


def _on_tool_error(exc: Exception, request: ToolCallRequest) -> str:
    # Sanitized: the model sees *that* the call failed and can apologize or retry,
    # but never a stack trace or internal hostname.
    return (
        f"The {request.tool_call['name']} service is temporarily unavailable ({type(exc).__name__}). Try again later."
    )


def _middleware(
    deps: GraphDeps, agent_name: str, hitl: HumanInTheLoopMiddleware[Any, Any, Any] | None = None
) -> list[AgentMiddleware[Any, Any, Any]]:
    a = deps.settings.agent
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        ToolAuditMiddleware(agent_name),
        ModelCallLimitMiddleware(run_limit=a.specialist_model_call_limit, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=a.specialist_tool_call_limit, exit_behavior="continue"),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True, apply_to_tool_results=True),
        RuntimeContextPromptMiddleware(),
    ]
    # Order = nesting (first is outermost). Model calls: Fallback -> Breaker -> Retry -> model.
    # Tool calls: ToolError -> Breaker -> ToolRetry -> MCP. Model-only and tool-only middleware
    # don't interact, so one breaker instance sits at the right depth for both.
    stack.append(ToolErrorMiddleware(_on_tool_error))
    if fallbacks := deps.models.fallbacks("smart"):
        stack.append(ModelFallbackMiddleware(*fallbacks))
    stack += [
        CircuitBreakerMiddleware(),
        ModelRetryMiddleware(max_retries=2, initial_delay=2.0, backoff_factor=2.0, on_failure="error"),
        ToolRetryMiddleware(max_retries=1, initial_delay=0.5, on_failure="error"),
    ]
    if hitl is not None:
        stack.append(hitl)
    return stack


async def build_orders_agent(deps: GraphDeps, customer_id: str) -> CompiledStateGraph[Any, Any, Any, Any]:
    commerce = await deps.toolkit.tools(customer_id, "commerce", scopes=ORDERS_SCOPES, names=ORDERS_TOOLS)
    helpdesk = await deps.toolkit.tools(customer_id, "helpdesk", scopes=ORDERS_SCOPES, names=ORDERS_HELPDESK_TOOLS)
    return create_agent(
        deps.models.get("smart"),
        [*commerce, *helpdesk],
        system_prompt=ORDERS_AGENT.format(company=deps.settings.agent.company_name, today=date.today().isoformat()),
        middleware=_middleware(deps, "orders"),
        context_schema=SupportContext,
        name="orders_agent",
    )


async def build_returns_agent(deps: GraphDeps, customer_id: str) -> CompiledStateGraph[Any, Any, Any, Any]:
    limit = deps.settings.agent.refund_auto_approve_limit
    tools = await deps.toolkit.tools(customer_id, "commerce", scopes=RETURNS_SCOPES, names=RETURNS_TOOLS)
    hitl = HumanInTheLoopMiddleware(
        # Gate by tool name *and* arguments: only refunds above the limit pause.
        interrupt_on={"issue_refund": refund_approval_gate(limit)},
        description_prefix="Refund approval required",
        # Our "edit" only attaches the signed approval credential - the model's intended call is
        # unchanged - so don't echo the edited args (and the credential) back into the model context.
        edit_notice=None,
    )
    return create_agent(
        deps.models.get("smart"),
        tools,
        system_prompt=RETURNS_AGENT.format(
            company=deps.settings.agent.company_name, today=date.today().isoformat(), auto_limit=limit
        ),
        middleware=_middleware(deps, "returns", hitl),
        context_schema=SupportContext,
        name="returns_agent",
    )


def _brief(payload: SpecialistInput) -> str:
    c = payload.get("customer") or {}
    lines = [
        f"Customer: {c.get('name', 'unknown')} | tier: {c.get('tier', 'standard')} | country: {c.get('country', 'US')}",
    ]
    if payload.get("summary"):
        lines.append(f"Conversation so far:\n{payload['summary']}")
    if payload.get("memories"):
        lines.append("Known about this customer: " + "; ".join(payload["memories"]))
    lines.append(f"\nTask: {payload['request']}")
    return "\n".join(lines)


def _extract(payload: SpecialistInput, messages: list[Any]) -> dict[str, Any]:
    tool_calls: list[dict[str, Any]] = []
    errors: list[str] = []
    rejected = False
    for m in messages:
        if isinstance(m, AIMessage):
            tool_calls += [{"name": tc["name"], "args": tc.get("args", {})} for tc in m.tool_calls]
        elif isinstance(m, ToolMessage) and m.status == "error":
            errors.append(f"{m.name}: {m.text[:300]}")
            if m.name == "issue_refund" and "reject" in m.text.lower():
                rejected = True
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    answer = final.text if final else "The specialist did not produce a final report."
    status: Literal["ok", "rejected"] = "rejected" if rejected else "ok"
    if errors:
        answer += "\n\nTool errors encountered:\n- " + "\n- ".join(errors)
    # Defense in depth: credentials must never flow on into the synthesis prompt.
    answer = _TOKEN_LIKE.sub("[redacted-credential]", answer)
    return specialist_result(
        task_id=payload["task_id"],
        agent=payload["agent"],
        request=payload["request"],
        answer=answer,
        status=status,
        tool_calls=tool_calls,
    )


async def run_tool_agent(
    deps: GraphDeps, payload: SpecialistInput, config: RunnableConfig, context: SupportContext
) -> dict[str, Any]:
    builder = build_orders_agent if payload["agent"] == "orders" else build_returns_agent
    agent = await builder(deps, context.customer_id)
    # Invoked inside a parent node => it runs as a subgraph: shares the parent's checkpointer
    # (so its interrupts pause/resume the whole run) and streams under its own namespace.
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=_brief(payload), id=f"task-{payload['task_id']}")]},
        config,
        context=context,
    )
    return _extract(payload, result["messages"])


async def run_knowledge(deps: GraphDeps, payload: SpecialistInput, config: RunnableConfig) -> dict[str, Any]:
    out = await deps.knowledge_graph.ainvoke(
        {"question": payload["request"], "context": payload.get("summary", "")}, config
    )
    return specialist_result(
        task_id=payload["task_id"],
        agent="knowledge",
        request=payload["request"],
        answer=out.get("answer", ""),
        status="ok" if out.get("answerable", False) else "error",
        contexts=out.get("contexts", []),
        citations=out.get("citations", []),
        error=None if out.get("answerable", False) else "not_answerable_from_kb",
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": q}} for q in out.get("search_queries", [])],
        cached=out.get("cached", False),
    )


def new_task_id() -> str:
    return uuid.uuid4().hex[:8]
