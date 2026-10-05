"""MCP client side: per-customer, least-privilege tool loading.

For every specialist run we:
  1. mint a short-lived service JWT for *this customer* with *only the scopes this
     specialist needs* (the orders agent cannot issue refunds even if prompted to),
  2. discover the server's tools through ``langchain.mcp.MCPAdapter``,
  3. keep only the tools this specialist is allowed to see (smaller tool lists
     also make small models pick tools more accurately).

Discovered tools are cached per (customer, server, scopes) for slightly less than
the token lifetime, so a multi-turn conversation doesn't re-run discovery on every
message. The cache key includes the customer, so tools (which carry the token)
are never shared between customers.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import Iterable
from typing import Any, Literal

from fastmcp.client import Client
from fastmcp.client.auth import BearerAuth
from langchain_core.tools import BaseTool

from samadhan.config import Settings
from samadhan.mcp_servers.auth import mint_service_token
from samadhan.resilience import BREAKERS
from samadhan.telemetry import langchain_span_context

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # langchain.mcp is beta; the warning is noise in logs
    from langchain.mcp import MCPAdapter

ServerName = Literal["commerce", "helpdesk"]

# Least privilege per caller type.
ORDERS_SCOPES = ("profile:read", "orders:read", "orders:write", "tickets:read")
RETURNS_SCOPES = ("profile:read", "orders:read", "returns:write", "refunds:write")
SYSTEM_SCOPES = ("profile:read", "orders:read", "tickets:read", "tickets:write")

ORDERS_TOOLS = {
    "get_customer_profile", "list_orders", "get_order", "track_shipment", "search_products",
    "cancel_order", "update_shipping_address",
}  # fmt: skip
ORDERS_HELPDESK_TOOLS = {"list_my_tickets", "get_ticket"}
RETURNS_TOOLS = {
    "list_orders", "get_order", "check_return_eligibility", "create_return", "get_return_status",
    "check_price_adjustment", "issue_refund",
}  # fmt: skip


def is_destructive(tool: BaseTool) -> bool:
    """Read the MCP ``destructiveHint`` that LangChain exposes under ``metadata['mcp']``."""
    annotations = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    return bool(annotations.get("destructive_hint", False))


class MCPToolkit:
    def __init__(self, settings: Settings, *, timeout_s: float = 30.0) -> None:
        self._settings = settings
        self._timeout = timeout_s
        self._cache: dict[tuple[str, str, tuple[str, ...]], tuple[float, list[BaseTool]]] = {}
        self._locks: dict[tuple[str, str, tuple[str, ...]], asyncio.Lock] = {}

    def url(self, server: ServerName) -> str:
        return {"commerce": self._settings.mcp.commerce_url, "helpdesk": self._settings.mcp.helpdesk_url}[server]

    def _client(self, server: ServerName, customer_id: str, scopes: tuple[str, ...]) -> Client[Any]:
        token = mint_service_token(self._settings.mcp, subject=customer_id, scopes=scopes)
        return Client(self.url(server), auth=BearerAuth(token), timeout=self._timeout)

    async def tools(
        self,
        customer_id: str,
        server: ServerName,
        *,
        scopes: tuple[str, ...],
        names: Iterable[str] | None = None,
    ) -> list[BaseTool]:
        key = (customer_id, server, scopes)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:  # concurrent specialists for one customer share one discovery
            cached = self._cache.get(key)
            if cached and cached[0] > time.monotonic():
                tools = cached[1]
            else:
                with langchain_span_context():  # MCP spans nest under the calling graph node
                    tools = await BREAKERS.get(f"mcp:{server}").call(
                        lambda: self._discover(server, customer_id, scopes)
                    )
                for t in tools:  # lets CircuitBreakerMiddleware map a tool call to its server
                    t.metadata = {**(t.metadata or {}), "mcp_server": server}
                ttl = min(self._settings.mcp.tool_cache_ttl_s, self._settings.mcp.token_ttl_s - 30)
                self._cache[key] = (time.monotonic() + max(ttl, 0), tools)
        wanted = set(names) if names is not None else None
        return [t for t in tools if wanted is None or t.name in wanted]

    async def call(
        self,
        customer_id: str,
        server: ServerName,
        tool: str,
        arguments: dict[str, Any],
        *,
        scopes: tuple[str, ...] = SYSTEM_SCOPES,
    ) -> Any:
        """Deterministic (non-LLM) MCP call from workflow code; returns structured content."""
        with langchain_span_context():
            result = await BREAKERS.get(f"mcp:{server}").call(
                lambda: self._call(server, customer_id, scopes, tool, arguments)
            )
        return result.structured_content if result.structured_content is not None else result.data

    async def _discover(self, server: ServerName, customer_id: str, scopes: tuple[str, ...]) -> list[BaseTool]:
        async with MCPAdapter(self._client(server, customer_id, scopes)) as adapter:
            return await adapter.list_tools()

    async def _call(
        self, server: ServerName, customer_id: str, scopes: tuple[str, ...], tool: str, arguments: dict[str, Any]
    ) -> Any:
        async with self._client(server, customer_id, scopes) as client:
            return await client.call_tool(tool, arguments)

    def invalidate(self, customer_id: str | None = None) -> None:
        for key in list(self._cache):
            if customer_id is None or key[0] == customer_id:
                self._cache.pop(key, None)
