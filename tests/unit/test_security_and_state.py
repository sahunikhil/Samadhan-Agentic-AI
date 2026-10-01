from __future__ import annotations

import time
from pathlib import Path

import jwt
import pytest
from langgraph.types import Interrupt
from pydantic import ValidationError

from caseflow.agents.state import collect_results
from caseflow.config import MCPSettings, Settings
from caseflow.mcp_servers.auth import ApprovalError, mint_approval_code, mint_service_token, verify_approval_code
from caseflow.rag.documents import chunk_document, parse_markdown
from caseflow.service import _confirmation_resume, classify_interrupt, new_thread_id, owner_of

MCP = MCPSettings()


def test_service_token_is_scoped_short_lived_and_audience_bound() -> None:
    token = mint_service_token(MCP, subject="cust_001", scopes=("orders:read",))
    claims = jwt.decode(token, MCP.jwt_secret.get_secret_value(), algorithms=["HS256"], audience="caseflow-mcp")
    assert claims["sub"] == "cust_001" and claims["scope"] == "orders:read"
    assert claims["exp"] - time.time() <= MCP.token_ttl_s + 1
    with pytest.raises(jwt.InvalidAudienceError):
        jwt.decode(token, MCP.jwt_secret.get_secret_value(), algorithms=["HS256"], audience="caseflow-approvals")


def test_approval_code_is_bound_to_customer_order_and_amount() -> None:
    code = mint_approval_code(MCP, customer_id="cust_002", order_id="VW-10004", max_amount=242.01, approver="sup")
    assert (
        verify_approval_code(MCP, code, customer_id="cust_002", order_id="VW-10004", amount=242.01)["approver"] == "sup"
    )
    with pytest.raises(ApprovalError):
        verify_approval_code(MCP, code, customer_id="cust_001", order_id="VW-10004", amount=242.01)
    with pytest.raises(ApprovalError):
        verify_approval_code(MCP, code, customer_id="cust_002", order_id="VW-10005", amount=242.01)
    with pytest.raises(ApprovalError):
        verify_approval_code(MCP, code, customer_id="cust_002", order_id="VW-10004", amount=500)
    with pytest.raises(ApprovalError):  # a service token is not an approval
        verify_approval_code(
            MCP, mint_service_token(MCP, subject="cust_002"), customer_id="cust_002", order_id="VW-10004", amount=1
        )


def test_production_refuses_insecure_defaults() -> None:
    with pytest.raises(ValidationError, match="Unsafe production configuration"):
        Settings(environment="prod")


def test_thread_ids_encode_ownership() -> None:
    tid = new_thread_id("cust_004")
    assert owner_of(tid) == "cust_004"


def test_interrupt_classification() -> None:
    elicit = Interrupt(value={"type": "mcp_elicitation", "tool_name": "cancel_order",
                              "requests": [{"key": "k", "message": "Cancel?", "mode": "form",
                                            "requested_schema": {"properties": {"confirm": {"type": "boolean"}}}}]}, id="1")  # fmt: skip
    hitl = Interrupt(
        value={
            "action_requests": [{"name": "issue_refund", "args": {}, "description": "Refund $200"}],
            "review_configs": [],
        },
        id="2",
    )
    handoff = Interrupt(value={"type": "human_handoff", "reason": "asked"}, id="3")
    assert classify_interrupt(elicit).audience == "customer"
    assert classify_interrupt(hitl).kind == "refund_approval" and classify_interrupt(hitl).audience == "supervisor"
    assert classify_interrupt(handoff).audience == "agent"
    assert _confirmation_resume(elicit.value, {"accept": True}) == {
        "responses": {"k": {"action": "accept", "content": {"confirm": True}}}
    }
    assert _confirmation_resume(elicit.value, {"accept": False}) == {"responses": {"k": {"action": "decline"}}}


def test_collect_results_appends_in_parallel_and_resets_per_turn() -> None:
    merged = collect_results(collect_results([], [{"agent": "orders"}]), [{"agent": "knowledge"}])
    assert [r["agent"] for r in merged] == ["orders", "knowledge"]
    assert collect_results(merged, None) == []


def test_chunks_carry_contextual_headers_and_stable_ids() -> None:
    doc = parse_markdown(Path("data/knowledge_base/returns-and-refunds-policy.md"))
    chunks = chunk_document(doc, chunk_size=600, chunk_overlap=80)
    assert doc.doc_id == "KB-003" and len(chunks) > 3
    fee = next(c for c in chunks if "Restocking fee" in c.section)
    assert fee.embed_text.startswith("Returns and Refunds Policy > Restocking fee")
    assert [c.chunk_id for c in chunk_document(doc, chunk_size=600, chunk_overlap=80)] == [c.chunk_id for c in chunks]
