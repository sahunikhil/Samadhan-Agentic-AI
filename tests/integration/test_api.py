"""HTTP API: auth, SSE streaming, HITL resume authorization, ops endpoints, MCP mount."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from caseflow.api.app import create_app
from caseflow.llm import ModelRegistry
from tests.fakes import ScriptedChatModel
from tests.integration.test_support_graph import responder

STAFF = {"X-Admin-Key": "dev-admin-key", "X-Staff-Name": "sup_test"}


def sse_events(response: Any) -> list[tuple[str, dict[str, Any]]]:
    events, current = [], "message"
    for line in response.iter_lines():
        if line.startswith("event:"):
            current = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((current, json.loads(line.split(":", 1)[1])))
    return events


@pytest.fixture(scope="module")
def client(mcp_servers: Any) -> Iterator[TestClient]:
    settings = mcp_servers.model_copy(
        update={"retrieval": mcp_servers.retrieval.model_copy(update={"qdrant_path": ":memory:"})}
    )
    models = ModelRegistry(settings.llm)
    fake = ScriptedChatModel(responder=responder)
    models.override("smart", fake)
    models.override("fast", fake)
    app = create_app(settings, models=models, checkpointer=InMemorySaver(), store=InMemoryStore())
    with TestClient(app) as c:  # runs the lifespan: container, auto-ingest, MCP mount
        yield c


def _login(client: TestClient, customer: str) -> dict[str, str]:
    r = client.post("/v1/auth/demo-login", json={"customer_id": customer})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_ops_endpoints(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["vector_chunks"] > 50
    assert "caseflow_http_requests_total" in client.get("/metrics").text


def test_auth_is_required(client: TestClient) -> None:
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 401
    assert client.post("/v1/chat", json={"message": "hi"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/auth/demo-login", json={"customer_id": "cust_999"}).status_code == 404


def test_chat_json(client: TestClient) -> None:
    r = client.post("/v1/chat", json={"message": "hello"}, headers=_login(client, "cust_001"))
    body = r.json()
    assert r.status_code == 200 and body["outcome"] == "resolved" and "help" in body["reply"]


def test_streaming_refund_flow_with_supervisor_approval(client: TestClient) -> None:
    customer = _login(client, "cust_002")
    with client.stream(
        "POST", "/v1/chat/stream", json={"message": "Where is my refund for my headphones return?"}, headers=customer
    ) as r:
        events = sse_events(r)
    kinds = [e for e, _ in events]
    assert kinds[0] == "run_started" and "status" in kinds and "tool" in kinds and kinds[-1] == "interrupt"
    thread_id = events[0][1]["thread_id"]
    [pending] = events[-1][1]["pending"]
    assert pending["kind"] == "refund_approval"

    # The customer cannot approve their own refund; another customer cannot even see the thread.
    decision = {"decisions": {pending["interrupt_id"]: {"decision": "approve"}}}
    assert client.post(f"/v1/threads/{thread_id}/resume", json=decision, headers=customer).status_code == 403
    assert client.get(f"/v1/threads/{thread_id}", headers=_login(client, "cust_001")).status_code == 403

    with client.stream("POST", f"/v1/threads/{thread_id}/resume", json=decision, headers=STAFF) as r:
        resumed = sse_events(r)
    assert resumed[-1][0] == "final" and resumed[-1][1]["outcome"] == "resolved"
    assert {"llm_calls", "cost_usd", "by_model"} <= set(resumed[-1][1]["usage"])  # per-turn cost accounting

    view = client.get(f"/v1/threads/{thread_id}", headers=customer).json()
    assert "specialist_results" not in view  # internal traces are staff-only
    staff_view = client.get(f"/v1/threads/{thread_id}", headers=STAFF).json()
    assert staff_view["specialist_results"][0]["agent"] == "returns"
    assert len(client.get(f"/v1/threads/{thread_id}/history", headers=STAFF).json()) > 5


def test_resume_without_pending_interrupt_is_a_conflict(client: TestClient) -> None:
    customer = _login(client, "cust_001")
    thread_id = client.post("/v1/chat", json={"message": "hello"}, headers=customer).json()["thread_id"]
    r = client.post(f"/v1/threads/{thread_id}/resume", json={"decisions": {"x": {"accept": True}}}, headers=customer)
    assert r.status_code == 409


def test_knowledge_mcp_server_is_mounted(client: TestClient) -> None:
    r = client.post(
        "/mcp/knowledge/",
        headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        },
    )
    assert r.status_code == 200 and "voltwise-knowledge" in r.text


def test_idempotency_key_replays_instead_of_rerunning(client: TestClient) -> None:
    customer = _login(client, "cust_001")
    headers = {**customer, "Idempotency-Key": "retry-test-0001"}
    first = client.post("/v1/chat", json={"message": "hello"}, headers=headers)
    again = client.post("/v1/chat", json={"message": "hello"}, headers=headers)
    assert first.status_code == again.status_code == 200
    assert again.headers["idempotent-replayed"] == "true" and again.json() == first.json()  # same thread, no 2nd run
    other_body = client.post("/v1/chat", json={"message": "something else"}, headers=headers)
    assert other_body.status_code == 422
    # Keys are per principal: another customer's identical key is a different operation.
    bob = client.post(
        "/v1/chat",
        json={"message": "hello"},
        headers={**_login(client, "cust_003"), "Idempotency-Key": "retry-test-0001"},
    )
    assert bob.status_code == 200 and "idempotent-replayed" not in bob.headers
    assert (
        client.post("/v1/chat", json={"message": "hi"}, headers={**customer, "Idempotency-Key": "x"}).status_code == 400
    )


def test_feedback_flywheel(client: TestClient) -> None:
    customer = _login(client, "cust_001")
    thread_id = client.post("/v1/chat", json={"message": "Can I return opened earbuds?"}, headers=customer).json()[
        "thread_id"
    ]
    url = f"/v1/threads/{thread_id}/feedback"
    assert (
        client.post(
            url,
            json={"rating": "down", "reason": "incomplete", "comment": "card 4242 4242 4242 4242"},
            headers=customer,
        ).status_code
        == 201
    )
    assert client.post(url, json={"rating": "down"}, headers=_login(client, "cust_002")).status_code == 403
    assert client.post(url, json={"rating": "meh"}, headers=customer).status_code == 422

    assert client.get("/v1/admin/feedback", headers=customer).status_code == 403
    [record] = [
        r for r in client.get("/v1/admin/feedback?rating=down", headers=STAFF).json() if r["thread_id"] == thread_id
    ]
    assert record["message"] == "Can I return opened earbuds?" and record["reason"] == "incomplete"
    assert "4242 4242 4242 4242" not in record["comment"]  # PII masked before storage
    assert (
        'caseflow_feedback_total{intent="policy_question",rating="down",reason="incomplete"}'
        in client.get("/metrics").text
    )

    from caseflow.feedback import to_candidate

    candidate = to_candidate(record)
    assert candidate["needs_label"] and candidate["message"] == record["message"] and candidate["id"].startswith("C-")


def test_readiness_reports_circuit_state(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["status"] in {"ready", "degraded"} and isinstance(body["circuits"], dict)


def test_ui_escapes_model_output_and_sends_security_headers(client: TestClient) -> None:
    """Interrupt titles carry model output (refund/escalation reasons) into the *staff* console:
    unescaped innerHTML there is stored XSS against the most privileged user (OWASP LLM05)."""
    import re

    page = client.get("/")
    html = page.text
    unescaped = [m for m in re.findall(r"innerHTML = `[^`]*`", html) if re.search(r"\$\{(?!esc\()", m)]
    assert not unescaped, unescaped
    csp = page.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "connect-src 'self'" in csp
    assert page.headers["x-content-type-options"] == "nosniff"
