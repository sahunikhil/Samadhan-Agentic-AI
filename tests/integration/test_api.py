"""HTTP API: auth, SSE streaming, HITL resume authorization, ops endpoints, MCP mount."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from samadhan.api.app import create_app
from samadhan.llm import ModelRegistry
from tests.fakes import ScriptedChatModel
from tests.integration.test_support_graph import responder

STAFF = {"X-Admin-Key": "dev-admin-key", "X-Staff-Name": "sup_test"}
IDP_ISSUER = "https://idp.example.test/realms/voltwise"
CUSTOMER_IDP_ISSUER = "https://idp.example.test/realms/customers"
_IDP_KEY = ec.generate_private_key(ec.SECP256R1())
IDP_PUBLIC = (
    _IDP_KEY.public_key()
    .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    .decode()
)


def _sso(user: str, roles: list[str], key: Any = _IDP_KEY, issuer: str = IDP_ISSUER) -> dict[str, str]:
    """An IdP-issued staff access token (as Keycloak / Auth0 / Entra would mint it)."""
    claims = {"iss": issuer, "aud": "samadhan", "sub": user, "preferred_username": user,
              "roles": roles, "exp": int(time.time()) + 300}  # fmt: skip
    return {"Authorization": f"Bearer {jwt.encode(claims, key, algorithm='ES256')}"}


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
        update={
            "retrieval": mcp_servers.retrieval.model_copy(update={"qdrant_path": ":memory:"}),
            # Staff SSO against a test identity provider (static public key instead of a JWKS URL).
            "api": mcp_servers.api.model_copy(
                update={
                    "staff_oidc_issuer": IDP_ISSUER,
                    "staff_oidc_audience": "samadhan",
                    "staff_oidc_public_key": IDP_PUBLIC,
                    "customer_oidc_issuer": CUSTOMER_IDP_ISSUER,
                    "customer_oidc_audience": "samadhan",
                    "customer_oidc_public_key": IDP_PUBLIC,
                    "customer_id_claim": "customer_id",
                }
            ),
        }
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
    assert "samadhan_http_requests_total" in client.get("/metrics").text


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
        'samadhan_feedback_total{intent="policy_question",rating="down",reason="incomplete"}'
        in client.get("/metrics").text
    )

    from samadhan.feedback import to_candidate

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
    script = client.get("/static/app.js")
    assert script.status_code == 200 and client.get("/static/app.css").status_code == 200
    unescaped = [m for m in re.findall(r"innerHTML = `[^`]*`", script.text) if re.search(r"\$\{(?!esc\()", m)]
    assert not unescaped, unescaped
    assert "<script>" not in page.text and "<style>" not in page.text and "style=" not in page.text + script.text
    csp = page.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "connect-src 'self'" in csp
    assert "unsafe-inline" not in csp  # inline script/style injected into the page would not run
    assert page.headers["x-content-type-options"] == "nosniff"


def test_staff_sso_roles_gate_refund_approvals_and_are_audited(client: TestClient) -> None:
    from structlog.testing import capture_logs

    customer = _login(client, "cust_002")
    with client.stream(
        "POST", "/v1/chat/stream", json={"message": "Where is my refund for my headphones return?"}, headers=customer
    ) as r:
        events = sse_events(r)
    thread_id = events[0][1]["thread_id"]
    [pending] = events[-1][1]["pending"]
    decision = {"decisions": {pending["interrupt_id"]: {"decision": "approve"}}}

    agent = _sso("ana.agent", ["agent"])
    assert client.get(f"/v1/threads/{thread_id}/history", headers=agent).status_code == 200  # staff access
    assert client.post(f"/v1/threads/{thread_id}/resume", json=decision, headers=agent).status_code == 403
    assert client.post("/v1/admin/ingest", headers=agent).status_code == 403  # admin role required
    assert client.get("/v1/admin/feedback", headers=_sso("nobody", [])).status_code == 200  # staff, no roles
    assert client.post(f"/v1/threads/{thread_id}/resume", json=decision, headers=_sso("x", [])).status_code == 403

    forged = _sso("mallory", ["supervisor"], key=ec.generate_private_key(ec.SECP256R1()))
    assert client.get(f"/v1/threads/{thread_id}/history", headers=forged).status_code == 401
    wrong_idp = _sso("mallory", ["supervisor"], issuer="https://evil.example")
    assert client.get(f"/v1/threads/{thread_id}/history", headers=wrong_idp).status_code == 401

    with (
        capture_logs() as logs,
        client.stream(
            "POST", f"/v1/threads/{thread_id}/resume", json=decision, headers=_sso("sam.supervisor", ["supervisor"])
        ) as r,
    ):
        resumed = sse_events(r)
    assert resumed[-1][0] == "final" and resumed[-1][1]["outcome"] == "resolved"
    [entry] = [e for e in logs if e.get("audit") and e["action"] == "refund_approval_answered"]
    assert entry["actor"] == "sam.supervisor" and entry["thread_id"] == thread_id


def test_customer_tokens_from_the_identity_provider(client: TestClient) -> None:
    def idp_customer(customer_id: str, issuer: str = CUSTOMER_IDP_ISSUER) -> dict[str, str]:
        claims = {"iss": issuer, "aud": "samadhan", "sub": "auth0|42", "customer_id": customer_id,
                  "exp": int(time.time()) + 300}  # fmt: skip
        return {"Authorization": f"Bearer {jwt.encode(claims, _IDP_KEY, algorithm='ES256')}"}

    reply = client.post("/v1/chat", json={"message": "hello"}, headers=idp_customer("cust_001"))
    assert reply.status_code == 200
    thread_id = reply.json()["thread_id"]
    assert thread_id.startswith("cust_001--"), "identity (and ownership) comes from the verified claim"
    assert client.get(f"/v1/threads/{thread_id}", headers=idp_customer("cust_002")).status_code == 403
    # A *staff*-realm token is not a customer identity: the issuer is checked.
    staff_realm = idp_customer("cust_001", IDP_ISSUER)
    assert client.post("/v1/chat", json={"message": "hi"}, headers=staff_realm).status_code == 403
