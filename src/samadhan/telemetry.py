"""OpenTelemetry tracing with the GenAI semantic conventions.

Why OpenTelemetry when LangSmith/Langfuse already trace the agent?
    Those are LLM-specific tools. OTel is the vendor-neutral standard the *rest* of production uses
    (Jaeger, Tempo, Honeycomb, Datadog, Grafana, cloud tracing). One trace then spans the HTTP
    request, the graph, every LLM call, every MCP tool call *and* the MCP server handling it (W3C
    ``traceparent`` is propagated over httpx), so "why was this turn slow?" has one answer.

What is emitted (semconv ``gen_ai.*``, development status - names verified Sept 2026)
    =============================  ========  ==================================================
    span                           kind      source
    =============================  ========  ==================================================
    ``invoke_workflow {graph}``    INTERNAL  root LangGraph run (``gen_ai.conversation.id``)
    ``invoke_agent {agent}``       INTERNAL  orders / returns / knowledge specialists
    ``{node}``                     INTERNAL  LangGraph nodes (``samadhan.graph.node``)
    ``chat {model}``               CLIENT    each LLM call: provider, model, token usage, finish
    ``execute_tool {tool}``        INTERNAL  each agent tool call (``ToolAuditMiddleware``)
    ``tools/call {tool}``          CLIENT    FastMCP's native MCP client span (``mcp.method.name``)
      -> MCP server span           SERVER    in the MCP server process, same trace via ``_meta``
    ``retrieval {collection}``     CLIENT    hybrid search + rerank (``rag/retriever.py``)
    =============================  ========  ==================================================

    Prompt/response *content* is opt-in in the spec and deliberately NOT recorded: it contains
    customer PII. Traces carry shape, timing, tokens and errors; content lives in the access-
    controlled LLM-observability tool (LangSmith/Langfuse) if you enable one.

Enable: ``SAMADHAN_OBSERVABILITY__OTEL_ENABLED=true`` + the standard ``OTEL_EXPORTER_OTLP_ENDPOINT``
(e.g. ``http://jaeger:4318``; ``docker compose --profile tracing up``). Disabled = zero overhead:
no handler is attached and the global tracer is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.errors import GraphBubbleUp
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

from samadhan.config import Settings

_enabled = False
AGENT_NAMES = frozenset({"orders_agent", "returns_agent", "knowledge_agent"})
# LangChain's ``ls_provider`` -> semconv ``gen_ai.provider.name`` well-known values.
_PROVIDERS = {"groq": "groq", "google_genai": "gcp.gemini", "openai": "openai", "ollama": "ollama"}


def tracer() -> Tracer:
    return trace.get_tracer("samadhan")


def tracing_enabled() -> bool:
    return _enabled


def setup_tracing(settings: Settings, service_name: str) -> bool:
    """Install the SDK tracer provider + OTLP exporter once per process (idempotent)."""
    global _enabled
    if _enabled or not settings.observability.otel_enabled:
        return _enabled
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {"service.name": service_name, "service.version": "0.1.0", "deployment.environment.name": settings.environment}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # endpoint from OTEL_EXPORTER_OTLP_*
    trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument()  # traceparent on MCP + LLM HTTP calls
    _enabled = True
    return True


def instrument_asgi(app: Any, excluded: str = "healthz,readyz,metrics") -> Any:
    """Server spans for a FastAPI app (API) or a raw ASGI app (MCP servers)."""
    if not _enabled:
        return app
    from fastapi import FastAPI

    if isinstance(app, FastAPI):
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls=excluded)
        return app
    from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

    return OpenTelemetryMiddleware(app, excluded_urls=excluded)


class OTelCallbackHandler(BaseCallbackHandler):
    """Turns LangChain/LangGraph callback events into GenAI-semconv spans.

    Parenting follows LangChain's ``run_id``/``parent_run_id`` tree. Runs we don't render as spans
    (internal runnables, prompt templates, parsers) are transparent: their children attach to the
    nearest rendered ancestor, so the trace shows workflow -> agent -> node -> chat/tool.
    ``run_inline`` keeps span bookkeeping on the event loop thread (no executor hop).
    """

    run_inline = True

    def __init__(self, tracer_: Tracer | None = None) -> None:
        super().__init__()
        self._tracer = tracer_ or tracer()
        self._spans: dict[UUID, Span] = {}
        self._ctx: dict[UUID, otel_context.Context] = {}
        self._names: dict[UUID, str] = {}  # rendered-ancestor name per run (dedupe node vs agent graph)
        self._ns_ctx: dict[str, otel_context.Context] = {}  # LangGraph checkpoint ns -> node span
        self._run_ns: dict[UUID, str] = {}

    # ---- helpers ----------------------------------------------------------------------------
    def _parent(self, parent_run_id: UUID | None, metadata: dict[str, Any] | None = None) -> otel_context.Context:
        if parent_run_id is not None and parent_run_id in self._ctx:
            return self._ctx[parent_run_id]
        # Some runnables run untraced (``trace=False``): their children name a parent run we never
        # saw. LangGraph stamps every run with its checkpoint namespace - use the node span there.
        ns = (metadata or {}).get("langgraph_checkpoint_ns")
        if ns and ns in self._ns_ctx:
            return self._ns_ctx[ns]
        return otel_context.get_current()  # e.g. the FastAPI request span

    def _start(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str,
        kind: SpanKind,
        attrs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        parent = self._parent(parent_run_id, metadata)
        span = self._tracer.start_span(name, context=parent, kind=kind, attributes=attrs)
        self._spans[run_id] = span
        self._ctx[run_id] = trace.set_span_in_context(span)
        self._names[run_id] = name

    def _passthrough(self, run_id: UUID, parent_run_id: UUID | None) -> None:
        self._ctx[run_id] = self._parent(parent_run_id)
        if parent_run_id in self._names:
            self._names[run_id] = self._names[parent_run_id]  # type: ignore[index]

    def context_for(self, run_id: UUID | None) -> otel_context.Context | None:
        return self._ctx.get(run_id) if run_id is not None else None

    def _end(self, run_id: UUID, error: BaseException | None = None) -> None:
        self._ctx.pop(run_id, None)
        self._names.pop(run_id, None)
        if (ns := self._run_ns.pop(run_id, None)) is not None:
            self._ns_ctx.pop(ns, None)
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        if error is not None:
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR, str(error)[:200]))
        span.end()

    # ---- chains: workflow / agents / graph nodes ---------------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        conversation = metadata.get("thread_id")
        if parent_run_id is None:
            attrs = {"gen_ai.operation.name": "invoke_workflow", "gen_ai.workflow.name": name}
            if conversation:
                attrs["gen_ai.conversation.id"] = conversation
            self._start(run_id, parent_run_id, f"invoke_workflow {name}", SpanKind.INTERNAL, attrs)
        elif "." in name:  # middleware hooks ("ModelCallLimitMiddleware.before_model"): noise
            self._passthrough(run_id, parent_run_id)
        elif name in AGENT_NAMES and self._names.get(parent_run_id) == f"invoke_agent {name}":  # type: ignore[arg-type]
            self._passthrough(run_id, parent_run_id)  # the agent graph inside its own graph node
        elif name in AGENT_NAMES:
            attrs = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": name}
            if conversation:
                attrs["gen_ai.conversation.id"] = conversation
            self._start(run_id, parent_run_id, f"invoke_agent {name}", SpanKind.INTERNAL, attrs)
        elif metadata.get("langgraph_node") == name and not name.startswith("__"):
            self._start(run_id, parent_run_id, name, SpanKind.INTERNAL, {"samadhan.graph.node": name}, metadata)
            if ns := metadata.get("langgraph_checkpoint_ns"):
                self._ns_ctx[ns] = self._ctx[run_id]
                self._run_ns[run_id] = ns
        else:
            self._passthrough(run_id, parent_run_id)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        # GraphInterrupt (HITL pause) and ParentCommand (Command.PARENT routing) are control flow.
        self._end(run_id, None if isinstance(error, GraphBubbleUp) else error)

    # ---- LLM calls ---------------------------------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metadata, params = metadata or {}, invocation_params or {}
        model = str(params.get("model") or params.get("model_name") or metadata.get("ls_model_name") or "unknown")
        provider = str(metadata.get("ls_provider") or params.get("_type") or "unknown")
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": _PROVIDERS.get(provider, provider),
            "gen_ai.request.model": model,
        }
        if metadata.get("ls_temperature") is not None:
            attrs["gen_ai.request.temperature"] = float(metadata["ls_temperature"])
        if metadata.get("thread_id"):
            attrs["gen_ai.conversation.id"] = metadata["thread_id"]
        self._start(run_id, parent_run_id, f"chat {model}", SpanKind.CLIENT, attrs, metadata)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span is not None:
            reasons: list[str] = []
            for generations in response.generations:
                for gen in generations:
                    info = gen.generation_info or {}
                    if info.get("finish_reason"):
                        reasons.append(str(info["finish_reason"]))
                    message = getattr(gen, "message", None)
                    usage = getattr(message, "usage_metadata", None)
                    meta = getattr(message, "response_metadata", None) or {}
                    if usage:
                        span.set_attribute("gen_ai.usage.input_tokens", int(usage.get("input_tokens", 0)))
                        span.set_attribute("gen_ai.usage.output_tokens", int(usage.get("output_tokens", 0)))
                        cached = (usage.get("input_token_details") or {}).get("cache_read")
                        if cached:
                            span.set_attribute("gen_ai.usage.cache_read.input_tokens", int(cached))
                    if meta.get("model_name"):
                        span.set_attribute("gen_ai.response.model", str(meta["model_name"]))
                    if meta.get("finish_reason"):
                        reasons.append(str(meta["finish_reason"]))
            if reasons:
                span.set_attribute("gen_ai.response.finish_reasons", sorted(set(reasons)))
        self._end(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)


def current_context() -> otel_context.Context:
    """The OTel context of the LangChain run executing *right now* (node, agent, tool).

    Callback start/end events fire in different async contexts, so the handler cannot safely make
    its spans "current" (contextvar tokens would be detached in the wrong context). Code that opens
    its own spans (retrieval) or makes instrumented HTTP calls (MCP tools) asks for the parent here:
    LangChain keeps the active run's callback manager - and its run id - in a contextvar.
    """
    config = var_child_runnable_config.get() or {}
    manager = config.get("callbacks")
    run_id = getattr(manager, "parent_run_id", None)
    for handler in getattr(manager, "handlers", None) or []:
        if isinstance(handler, OTelCallbackHandler) and (ctx := handler.context_for(run_id)) is not None:
            return ctx
    return otel_context.get_current()


@contextmanager
def langchain_span_context() -> Iterator[None]:
    """Make the current LangChain run's span current for a block (attach/detach in one coroutine)."""
    if not _enabled:
        yield
        return
    token = otel_context.attach(current_context())
    try:
        yield
    finally:
        otel_context.detach(token)


@contextmanager
def tool_span(tool: str, call_id: str | None, agent: str) -> Iterator[Span | None]:
    """``execute_tool {tool}`` span, *current* while the tool runs (used by ToolAuditMiddleware).

    Created here rather than in the callback handler because the middleware wraps the tool call in
    one coroutine, so the span can safely be made current: FastMCP's client span
    (``tools/call {tool}``) nests under it and injects W3C ``traceparent`` into the MCP request
    ``_meta``; the MCP server extracts it, so the server-side span joins the same trace - over any
    transport (HTTP or stdio), without HTTP-level instrumentation.
    """
    if not _enabled:
        yield None
        return
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool,
        "gen_ai.tool.type": "function",
        "gen_ai.agent.name": agent,
    }
    if call_id:
        attrs["gen_ai.tool.call.id"] = call_id
    with tracer().start_as_current_span(
        f"execute_tool {tool}",
        context=current_context(),
        kind=SpanKind.INTERNAL,
        attributes=attrs,
        record_exception=False,
        set_status_on_exception=False,  # the caller decides: an HITL interrupt is not an error
    ) as span:
        yield span


def mark_error(span: Span | None, error_type: str, message: str = "") -> None:
    if span is not None:
        span.set_attribute("error.type", error_type)
        span.set_status(Status(StatusCode.ERROR, message[:200]))
