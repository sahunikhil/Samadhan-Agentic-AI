"""Logging, metrics and tracing.

Three complementary signals, each answering a different question:

* **Structured logs** (structlog, JSON in production) - *what happened* in one request.
* **Prometheus metrics** - *how the fleet behaves* over time (latency, error rate,
  tokens, escalations). Cheap to store, great for alerting and cost dashboards.
* **LLM traces** (Langfuse, open source / self-hostable, or LangSmith) - *why the
  agent did that*: every prompt, tool call and model response in a run tree.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from prometheus_client import Counter, Gauge, Histogram

from samadhan.config import Settings

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


def _add_trace_context(_: Any, __: str, event: dict[str, Any]) -> dict[str, Any]:
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event["trace_id"] = format(ctx.trace_id, "032x")
        event["span_id"] = format(ctx.span_id, "016x")
    return event


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.observability.log_level.upper(), logging.INFO)
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,  # request_id / thread_id bound per request
        _add_trace_context,  # trace_id/span_id: jump from a log line to its trace
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.observability.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    # Third-party libraries are chatty at INFO; keep them at WARNING.
    for noisy in ("httpx", "httpx2", "httpcore", "mcp", "fastmcp", "uvicorn.access", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


_audit_log = structlog.get_logger("samadhan.audit")


def audit(action: str, actor: str, **fields: Any) -> None:
    """Audit trail for staff actions (approvals, handoff replies, re-indexing). One JSON line per event
    with ``audit=true`` - the log shipper (Fluent Bit, Vector, the cloud agent) routes these lines to
    the SIEM; request_id/trace_id come from the bound context. Never put customer content here."""
    _audit_log.info("audit", audit=True, action=action, actor=actor, **fields)


# --------------------------------------------------------------------------------------
# Prometheus metrics (process-global, registered once at import time)
# --------------------------------------------------------------------------------------

HTTP_REQUESTS = Counter("samadhan_http_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_LATENCY = Histogram(
    "samadhan_http_request_seconds",
    "HTTP request latency",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
AGENT_RUNS = Counter("samadhan_agent_runs_total", "Graph runs by outcome", ["outcome"])
NODE_LATENCY = Histogram(
    "samadhan_graph_node_seconds", "Time spent per graph node", ["node"], buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
)
LLM_TOKENS = Counter("samadhan_llm_tokens_total", "LLM tokens consumed", ["model", "kind"])
TOOL_CALLS = Counter("samadhan_tool_calls_total", "Agent tool calls", ["tool", "status"])
ESCALATIONS = Counter("samadhan_escalations_total", "Cases handed to a human", ["reason"])
HITL_INTERRUPTS = Counter("samadhan_hitl_interrupts_total", "Human-in-the-loop pauses", ["kind"])
GUARDRAIL_BLOCKS = Counter("samadhan_guardrail_blocks_total", "Requests or answers blocked", ["stage", "reason"])
LLM_COST = Counter("samadhan_llm_cost_usd_total", "LLM spend at list prices (USD)", ["model"])
LLM_UNPRICED = Counter(
    "samadhan_llm_unpriced_calls_total", "LLM calls for models missing from the price table", ["model"]
)
TURN_COST = Histogram(
    "samadhan_turn_cost_usd",
    "LLM spend per graph run (USD, list prices)",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
TURN_OVER_BUDGET = Counter("samadhan_turn_over_budget_total", "Graph runs whose LLM spend exceeded the budget")
CIRCUIT_STATE = Gauge("samadhan_circuit_state", "Circuit breaker state: 0 closed, 1 half-open, 2 open", ["dependency"])
CIRCUIT_REJECTIONS = Counter(
    "samadhan_circuit_rejections_total", "Calls failed fast by an open circuit", ["dependency"]
)
FEEDBACK = Counter(
    "samadhan_feedback_total", "User ratings of replies (online quality signal)", ["rating", "reason", "intent"]
)
SEMANTIC_CACHE = Counter(
    "samadhan_semantic_cache_total",
    "Semantic cache lookups (hit_exact, hit_verified, rejected by verifier, miss, skipped as personal, error)",
    ["result"],
)
RETRIEVAL_LATENCY = Histogram(
    "samadhan_retrieval_seconds", "Hybrid retrieval + rerank latency", buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2)
)


class TokenUsageCallback(AsyncCallbackHandler):
    """Counts prompt/completion tokens per model from LangChain ``usage_metadata``.

    This is the foundation of cost tracking: tokens x provider price = spend,
    broken down by model so you can see which role (smart/fast) drives cost.
    """

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) if message is not None else None
                if not usage:
                    continue
                meta = getattr(message, "response_metadata", {}) or {}
                model = str(meta.get("model_name") or meta.get("model") or "unknown")
                LLM_TOKENS.labels(model=model, kind="input").inc(usage.get("input_tokens", 0))
                LLM_TOKENS.labels(model=model, kind="output").inc(usage.get("output_tokens", 0))


def tracing_callbacks(settings: Settings) -> list[Any]:
    """Callbacks attached to every graph run (token metrics + optional OpenTelemetry / Langfuse)."""
    callbacks: list[Any] = [TokenUsageCallback()]
    from samadhan.telemetry import OTelCallbackHandler, tracing_enabled

    if tracing_enabled():
        callbacks.append(OTelCallbackHandler())
    if settings.observability.langfuse_enabled and os.getenv("LANGFUSE_PUBLIC_KEY"):
        try:
            from langfuse.langchain import CallbackHandler  # optional dependency

            callbacks.append(CallbackHandler())
        except ImportError:  # pragma: no cover - only when the extra is missing
            get_logger(__name__).warning("langfuse_not_installed", hint="uv sync --extra observability")
    return callbacks
