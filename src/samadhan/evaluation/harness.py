"""Self-contained MCP environment for evaluations.

Agent scenarios *mutate* data (refunds, cancellations), so each scenario must start
from the same known state. The harness runs the real commerce + helpdesk MCP
servers on ephemeral ports against throwaway SQLite files and can reseed between
scenarios - evals are reproducible and never touch your dev or prod databases.
"""

from __future__ import annotations

import asyncio
import socket
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn

from samadhan.config import Settings
from samadhan.mcp_servers.commerce.seed import seed_commerce
from samadhan.mcp_servers.commerce.server import create_commerce_server
from samadhan.mcp_servers.db import Database
from samadhan.mcp_servers.helpdesk.server import create_helpdesk_server


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Server:
    def __init__(self, app: Any, port: int) -> None:
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 30
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("MCP server failed to start")
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


class MCPEnvironment:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def reseed(self) -> None:
        db = Database(self.settings.mcp.commerce_db_url)
        try:
            await seed_commerce(db, reset=True)
        finally:
            await db.dispose()


@asynccontextmanager
async def mcp_environment(base: Settings) -> AsyncIterator[MCPEnvironment]:
    """Yield settings wired to freshly-seeded MCP servers on ephemeral ports."""
    tmp = Path(tempfile.mkdtemp(prefix="samadhan-eval-"))
    cp, hp = free_port(), free_port()
    settings = base.model_copy(
        update={
            "mcp": base.mcp.model_copy(
                update={
                    "commerce_url": f"http://127.0.0.1:{cp}/mcp",
                    "helpdesk_url": f"http://127.0.0.1:{hp}/mcp",
                    "commerce_port": cp,
                    "helpdesk_port": hp,
                    "commerce_db_url": f"sqlite+aiosqlite:///{(tmp / 'commerce.db').as_posix()}",
                    "helpdesk_db_url": f"sqlite+aiosqlite:///{(tmp / 'helpdesk.db').as_posix()}",
                }
            ),
            "persistence": base.persistence.model_copy(update={"database_url": None, "sqlite_path": tmp / "cp.sqlite"}),
        }
    )
    env = MCPEnvironment(settings)
    await env.reseed()
    servers = [
        _Server(create_commerce_server(settings).http_app(stateless_http=True), cp),
        _Server(create_helpdesk_server(settings).http_app(stateless_http=True), hp),
    ]
    for s in servers:
        await asyncio.to_thread(s.start)
    try:
        yield env
    finally:
        for s in servers:
            await asyncio.to_thread(s.stop)
