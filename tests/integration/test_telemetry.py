"""OpenTelemetry: one trace per turn, GenAI-semconv span names/attributes, correct nesting,
and W3C trace-context propagation on the HTTP calls to the MCP servers."""

from __future__ import annotations

import fastmcp.client.telemetry
import fastmcp.server.telemetry
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

import samadhan.rag.retriever as retriever_module
import samadhan.telemetry as telemetry
from samadhan.service import SupportService
from tests.integration.test_support_graph import service  # noqa: F401 - fixture


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    # A local provider instead of the process-global one (which can be set only once).
    monkeypatch.setattr(telemetry, "tracer", lambda: tracer)
    monkeypatch.setattr(retriever_module, "tracer", lambda: tracer)
    monkeypatch.setattr(telemetry, "_enabled", True)
    # FastMCP's native MCP spans (client here, server in the MCP server threads of this process).
    monkeypatch.setattr(fastmcp.client.telemetry, "get_tracer", lambda *a: tracer)
    monkeypatch.setattr(fastmcp.server.telemetry, "get_tracer", lambda *a: tracer)
    return exporter


def _ancestors(span: ReadableSpan, by_id: dict[int, ReadableSpan]) -> list[str]:
    names, parent = [], span.parent
    while parent is not None and parent.span_id in by_id:
        names.append(by_id[parent.span_id].name)
        parent = by_id[parent.span_id].parent
    return names


async def test_a_turn_is_one_trace_with_genai_spans(service: SupportService, spans: InMemorySpanExporter) -> None:  # noqa: F811
    result = await service.run_turn(customer_id="cust_001", message="Where is VW-10003? Also do you price match?")
    finished = spans.get_finished_spans()
    by_id = {s.context.span_id: s for s in finished}
    by_name: dict[str, list[ReadableSpan]] = {}
    for s in finished:
        by_name.setdefault(s.name, []).append(s)

    # One trace for the whole turn, MCP servers included. (FastMCP injects `traceparent` into `_meta`
    # on tools/call - the calls that matter - but not on tools/list discovery, whose server-side
    # spans therefore start their own trace.)
    in_turn = [s for s in finished if not (s.kind == SpanKind.SERVER and s.name == "tools/list")]
    assert len({s.context.trace_id for s in in_turn}) == 1, "the whole turn - incl. MCP servers - is one trace"
    [root] = by_name["invoke_workflow support_turn"]
    assert root.attributes["gen_ai.operation.name"] == "invoke_workflow"
    assert root.attributes["gen_ai.conversation.id"] == result.thread_id

    # Parallel specialists are agents; their LLM calls and tools nest under them.
    assert {"invoke_agent orders_agent", "invoke_agent knowledge_agent"} <= set(by_name)
    [tool] = by_name["execute_tool track_shipment"]
    assert tool.attributes["gen_ai.tool.name"] == "track_shipment"
    assert tool.attributes["gen_ai.agent.name"] == "orders" and "invoke_agent orders_agent" in _ancestors(tool, by_id)
    chats = [s for s in finished if s.name.startswith("chat ")]
    assert chats and all(s.attributes["gen_ai.operation.name"] == "chat" for s in chats)
    assert all(_ancestors(s, by_id)[-1] == "invoke_workflow support_turn" for s in chats), "no orphan LLM spans"

    [retrieval] = by_name["retrieval voltwise_kb"]
    assert retrieval.attributes["gen_ai.data_source.id"] == "voltwise_kb"
    assert _ancestors(retrieval, by_id)[:2] == ["retrieve", "invoke_agent knowledge_agent"]

    # The MCP tool call's HTTP request is a child span in the same trace -> traceparent propagates
    # to the MCP server, whose own spans (when instrumented) join this trace.
    # MCP trace propagation (W3C traceparent in the request `_meta`), two paths:
    def one(name: str, kind: SpanKind) -> ReadableSpan:
        return next(s for s in finished if s.name == name and s.kind == kind)

    # 1) agent tool call (langchain.mcp adapter): the MCP *server* span is a child of execute_tool.
    server = one("tools/call track_shipment", SpanKind.SERVER)
    assert _ancestors(server, by_id)[:2] == ["execute_tool track_shipment", "tools"]
    # 2) deterministic workflow call (toolkit.call -> FastMCP client span) -> server span beneath it.
    client = one("tools/call get_customer_profile", SpanKind.CLIENT)
    assert _ancestors(client, by_id)[0] == "load_context"
    assert one("tools/call get_customer_profile", SpanKind.SERVER).parent.span_id == client.context.span_id


async def test_tracing_disabled_adds_no_handler(service: SupportService) -> None:  # noqa: F811
    from samadhan.observability import tracing_callbacks

    assert not telemetry.tracing_enabled()
    assert not any(isinstance(cb, telemetry.OTelCallbackHandler) for cb in tracing_callbacks(service.c.settings))


async def test_hitl_pause_is_not_an_error_span(service: SupportService, spans: InMemorySpanExporter) -> None:  # noqa: F811
    from opentelemetry.trace import StatusCode

    result = await service.run_turn(customer_id="cust_002", message="Where is my refund for my headphones return?")
    assert [p.kind for p in result.pending] == ["refund_approval"]  # paused for a supervisor
    errors = [s.name for s in spans.get_finished_spans() if s.status.status_code == StatusCode.ERROR]
    assert errors == [], f"an interrupt is control flow, not a failure: {errors}"
