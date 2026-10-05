"""MCP servers over real Streamable HTTP with JWT auth: isolation, scopes, annotations, policy enforcement."""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp.client import Client
from fastmcp.client.auth import BearerAuth
from pydantic import SecretStr

from samadhan.agents.toolkit import MCPToolkit, is_destructive
from samadhan.mcp_servers.auth import generate_key_pair, mint_service_token
from samadhan.mcp_servers.knowledge.server import create_knowledge_server


async def test_tool_annotations_and_hidden_identity_parameter(mcp_servers: Any) -> None:
    toolkit = MCPToolkit(mcp_servers)
    tools = {
        t.name: t
        for t in await toolkit.tools("cust_001", "commerce", scopes=("orders:read", "refunds:write", "profile:read"))
    }
    assert is_destructive(tools["issue_refund"]) and not is_destructive(tools["get_order"])
    for tool in tools.values():  # the model can never choose whose data to read
        schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
        assert "customer_id" not in schema.get("properties", {})


async def test_scopes_limit_which_tools_are_visible(mcp_servers: Any) -> None:
    toolkit = MCPToolkit(mcp_servers)
    read_only = {t.name for t in await toolkit.tools("cust_001", "commerce", scopes=("orders:read",))}
    assert "get_order" in read_only
    assert "issue_refund" not in read_only and "cancel_order" not in read_only


async def test_customers_cannot_read_each_others_orders(mcp_servers: Any) -> None:
    toolkit = MCPToolkit(mcp_servers)
    mine = await toolkit.call("cust_001", "commerce", "get_order", {"order_id": "VW-10001"}, scopes=("orders:read",))
    assert mine["order_id"] == "VW-10001"
    with pytest.raises(Exception, match="not found on this account"):
        await toolkit.call("cust_002", "commerce", "get_order", {"order_id": "VW-10001"}, scopes=("orders:read",))


async def test_forged_or_missing_tokens_are_rejected(mcp_servers: Any) -> None:
    # A validly-formed EdDSA token signed by a *different* key (e.g. an attacker's) is rejected.
    wrong_key = mcp_servers.mcp.model_copy(update={"jwt_private_key": SecretStr(generate_key_pair()[0])})
    for bad in ("not-a-jwt", mint_service_token(wrong_key, subject="cust_001")):
        with pytest.raises(Exception):
            async with Client(mcp_servers.mcp.commerce_url, auth=BearerAuth(bad)) as c:
                await c.list_tools()


async def test_refunds_above_limit_require_signed_approval_server_side(mcp_servers: Any) -> None:
    toolkit = MCPToolkit(mcp_servers)
    # A legitimate refund (return received) that is simply too large to auto-approve.
    with pytest.raises(Exception, match="require approval"):
        await toolkit.call("cust_002", "commerce", "issue_refund",
                           {"order_id": "VW-10004", "amount": 242.01, "reason": "return_received"},
                           scopes=("refunds:write", "orders:read"))  # fmt: skip
    # Business rules are enforced independently of approvals: stale price adjustments are refused.
    with pytest.raises(Exception, match="within 14 days"):
        await toolkit.call("cust_003", "commerce", "issue_refund",
                           {"order_id": "VW-10007", "amount": 50.0, "reason": "price_adjustment"},
                           scopes=("refunds:write", "orders:read"))  # fmt: skip


async def test_return_eligibility_is_computed_by_the_server(mcp_servers: Any) -> None:
    toolkit = MCPToolkit(mcp_servers)
    out = await toolkit.call("cust_003", "commerce", "check_return_eligibility",
                             {"order_id": "VW-10007", "sku": "VB-PRO-16"}, scopes=("orders:read",))  # fmt: skip
    assert out["eligible"] and out["refund_amount"] == pytest.approx(1607.16)


async def test_knowledge_server_tools_resources_and_prompts(retriever: Any, settings: Any) -> None:
    async def get_retriever() -> Any:
        return retriever

    server = create_knowledge_server(get_retriever, settings)
    async with Client(server) as client:  # in-process transport: no network needed
        result = await client.call_tool("search_knowledge_base", {"query": "can I return opened earbuds", "top_k": 3})
        hits = result.structured_content["result"]
        assert hits[0]["article_id"] in {"KB-017", "KB-003"} and hits[0]["relevance"] > 0.5
        index = await client.read_resource("kb://index")
        assert "KB-001" in index[0].text
        prompt = await client.get_prompt("grounded_answer", {"question": "warranty?"})
        assert "search_knowledge_base" in prompt.messages[0].content.text


async def test_price_adjustment_check_computes_difference_and_next_step(mcp_servers: Any) -> None:
    toolkit = MCPToolkit(mcp_servers)
    out = await toolkit.call(
        "cust_004", "commerce", "check_price_adjustment", {"order_id": "VW-10010"}, scopes=("orders:read",)
    )
    assert out["eligible"] and out["refundable_amount"] == pytest.approx(20.0)
    assert "issue_refund" in out["next_step"] and "price_adjustment" in out["next_step"]
    stale = await toolkit.call(
        "cust_002", "commerce", "check_price_adjustment", {"order_id": "VW-10006"}, scopes=("orders:read",)
    )
    assert not stale["eligible"] and "14 days" in stale["next_step"]


async def test_return_status_tells_the_agent_what_to_do_next(mcp_servers: Any) -> None:
    toolkit = MCPToolkit(mcp_servers)
    out = await toolkit.call(
        "cust_002", "commerce", "get_return_status", {"order_id": "VW-10004"}, scopes=("orders:read",)
    )
    [rma] = [r for r in out["result"] if r["type"] == "return"]
    assert rma["refund_ready"] and "call issue_refund" in rma["next_step"] and "242.01" in rma["next_step"]


RW = ("refunds:write", "returns:write", "orders:read")


async def create_return(settings: Any, customer: str, args: dict[str, Any], *, confirm: bool = True) -> Any:
    """Call create_return as the customer would through a chat client: the server asks for
    confirmation (MCP elicitation) and the handler answers it."""
    token = mint_service_token(settings.mcp, subject=customer, scopes=RW)
    asked: list[str] = []

    async def customer_answers(message: str, response_type: Any, params: Any, context: Any) -> dict[str, bool]:
        asked.append(message)
        return {"confirm": confirm}

    async with Client(settings.mcp.commerce_url, auth=BearerAuth(token), elicitation_handler=customer_answers) as c:
        result = await c.call_tool("create_return", args)
    data = result.structured_content or {}
    data = data.get("result", data)
    return {**data, "_asked": asked}


async def test_refund_invariants_hold_on_the_server(mcp_servers: Any) -> None:
    """Money rules the agent cannot talk its way past: idempotent RMA, cap, no double refund."""
    toolkit = MCPToolkit(mcp_servers)
    call = lambda tool, args: toolkit.call("cust_006", "commerce", tool, args, scopes=RW)  # noqa: E731
    rma = await create_return(
        mcp_servers, "cust_006", {"order_id": "VW-10013", "sku": "PULSE-CUSH", "reason": "damaged_on_arrival"}
    )
    assert rma["_asked"] and "$29.00" in rma["_asked"][0], "the customer confirms the exact amount first"
    assert rma["instant_refund_eligible"] and rma["refund_amount"] == pytest.approx(29.0)
    again = await create_return(
        mcp_servers, "cust_006", {"order_id": "VW-10013", "sku": "PULSE-CUSH", "reason": "damaged_on_arrival"}
    )
    assert again["rma_id"] == rma["rma_id"], "create_return is idempotent"
    refund_args = {"order_id": "VW-10013", "reason": "damaged_on_arrival"}
    with pytest.raises(Exception, match="exceeds the refundable amount"):
        await call("issue_refund", {**refund_args, "amount": 30.0})
    receipt = await call("issue_refund", {**refund_args, "amount": 29.0})
    assert receipt["approved_by"] == "auto"
    with pytest.raises(Exception, match="Nothing left to refund"):  # a retried or duplicated call can't pay twice
        await call("issue_refund", {**refund_args, "amount": 29.0})


async def test_price_adjustment_check_and_refund_agree(mcp_servers: Any) -> None:
    """The check tool and issue_refund share one definition - what the agent is told is what is accepted."""
    toolkit = MCPToolkit(mcp_servers)
    call = lambda tool, args: toolkit.call("cust_004", "commerce", tool, args, scopes=RW)  # noqa: E731
    check = await call("check_price_adjustment", {"order_id": "VW-10010"})
    assert check["eligible"] and check["refundable_amount"] == pytest.approx(20.0)
    with pytest.raises(Exception, match="exceeds"):
        await call("issue_refund", {"order_id": "VW-10010", "amount": 25.0, "reason": "price_adjustment"})
    await call("issue_refund", {"order_id": "VW-10010", "amount": 20.0, "reason": "price_adjustment"})
    after = await call("check_price_adjustment", {"order_id": "VW-10010"})
    assert (
        not after["eligible"] and after["refundable_amount"] == 0 and after["already_refunded"] == pytest.approx(20.0)
    )
    with pytest.raises(Exception, match="Nothing left"):
        await call("issue_refund", {"order_id": "VW-10010", "amount": 20.0, "reason": "price_adjustment"})


async def test_returned_items_are_not_price_adjusted(mcp_servers: Any) -> None:
    toolkit = MCPToolkit(mcp_servers)
    call = lambda tool, args: toolkit.call("cust_004", "commerce", tool, args, scopes=RW)  # noqa: E731
    await create_return(mcp_servers, "cust_004", {"order_id": "VW-10010", "sku": "VC-140W", "reason": "changed_mind"})
    check = await call("check_price_adjustment", {"order_id": "VW-10010"})
    assert not check["eligible"] and check["lines"][0]["returned"]
    with pytest.raises(Exception, match="Nothing left"):  # and issue_refund agrees with the check
        await call("issue_refund", {"order_id": "VW-10010", "amount": 20.0, "reason": "price_adjustment"})


async def test_a_return_is_only_created_when_the_customer_confirms(mcp_servers: Any) -> None:
    args = {"order_id": "VW-10007", "sku": "VB-PRO-16", "reason": "changed_mind"}
    declined = await create_return(mcp_servers, "cust_003", args, confirm=False)
    assert declined["created"] is False and "$1607.16" in declined["_asked"][0]
    toolkit = MCPToolkit(mcp_servers)
    status = await toolkit.call("cust_003", "commerce", "get_return_status", {"order_id": "VW-10007"}, scopes=RW)
    statuses = status if isinstance(status, list) else status.get("result", [])
    assert not [r for r in statuses if r.get("type") == "return"], "declining leaves no RMA behind"
