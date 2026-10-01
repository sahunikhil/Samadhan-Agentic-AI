"""Edge hardening: body-size cap, per-IP limit on public CPU-heavy routes, production secret strength,
bounded public tool input."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastmcp.client import Client
from fastmcp.exceptions import ToolError

from caseflow.api.security import EdgeGuardMiddleware, SlidingWindowRateLimiter
from caseflow.config import Settings
from caseflow.mcp_servers.knowledge.server import create_knowledge_server


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    @app.post("/mcp/knowledge/")
    async def kb() -> dict[str, str]:
        return {"ok": "yes"}

    app.add_middleware(
        EdgeGuardMiddleware,
        max_body_bytes=1000,
        ip_limiter=SlidingWindowRateLimiter(3),
        ip_limited=("/mcp/knowledge",),
    )
    return TestClient(app)


def test_oversized_bodies_are_rejected_declared_or_chunked(client: TestClient) -> None:
    assert client.post("/echo", content=b"x" * 500).json() == {"size": 500}
    assert client.post("/echo", content=b"x" * 5000).status_code == 413  # Content-Length checked up front

    def chunks() -> Any:  # no Content-Length: counted as the body streams in
        for _ in range(10):
            yield b"x" * 500

    assert client.post("/echo", content=chunks()).status_code == 413


def test_public_cpu_heavy_routes_are_rate_limited_per_ip(client: TestClient) -> None:
    codes = [client.post("/mcp/knowledge/").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert client.post("/echo", content=b"ok").status_code == 200  # other routes are unaffected


def test_production_refuses_weak_secrets() -> None:
    weak = {
        "environment": "prod",
        "mcp": {"jwt_secret": "x" * 40},
        "api": {"admin_api_key": "short-key", "token_secret": "too-short", "demo_mode": False},
    }
    with pytest.raises(ValueError, match="TOKEN_SECRET must be at least 32") as exc:
        Settings(**weak)  # type: ignore[arg-type]
    assert "ADMIN_API_KEY must be at least 24" in str(exc.value)
    strong = {**weak, "api": {"admin_api_key": "a" * 24, "token_secret": "t" * 32, "demo_mode": False}}
    assert Settings(**strong).environment == "prod"  # type: ignore[arg-type]


async def test_public_knowledge_tool_bounds_query_size() -> None:
    async def never_called() -> Any:
        raise AssertionError("retrieval must not run for an oversized query")

    async with Client(create_knowledge_server(never_called, Settings())) as mcp:
        with pytest.raises(ToolError, match="at most 1000"):
            await mcp.call_tool("search_knowledge_base", {"query": "x" * 1001})
