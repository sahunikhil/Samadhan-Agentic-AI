"""Shared fixtures.

Integration tests run the *real* MCP servers (Streamable HTTP + JWT auth) on
ephemeral ports in background threads, seeded with the deterministic demo data,
and the real retrieval stack (embedded Qdrant in memory + local FastEmbed models).
Only the LLM is replaced by a scripted fake - so tests are free, fast and
deterministic, yet exercise every other layer for real.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn

warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta.*")
os.environ.setdefault("CASEFLOW_ENVIRONMENT", "test")
# Tests never ship traces to a SaaS (flaky network, leaks fixture data, burns quota).
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from caseflow.config import Settings  # noqa: E402
from caseflow.mcp_servers.commerce.seed import seed_commerce  # noqa: E402
from caseflow.mcp_servers.commerce.server import create_commerce_server  # noqa: E402
from caseflow.mcp_servers.db import Database  # noqa: E402
from caseflow.mcp_servers.helpdesk.server import create_helpdesk_server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _ServerThread:
    def __init__(self, app: Any, port: int) -> None:
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> _ServerThread:
        self.thread.start()
        deadline = time.time() + 20
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("MCP server did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("caseflow")


@pytest.fixture(scope="session")
def settings(data_dir: Path) -> Settings:
    commerce_port, helpdesk_port = _free_port(), _free_port()
    return Settings(
        environment="test",
        retrieval={"qdrant_path": ":memory:", "kb_dir": ROOT / "data" / "knowledge_base"},
        persistence={"sqlite_path": data_dir / "checkpoints.sqlite"},
        mcp={
            "commerce_url": f"http://127.0.0.1:{commerce_port}/mcp",
            "helpdesk_url": f"http://127.0.0.1:{helpdesk_port}/mcp",
            "commerce_port": commerce_port,
            "helpdesk_port": helpdesk_port,
            "commerce_db_url": f"sqlite+aiosqlite:///{(data_dir / 'commerce.db').as_posix()}",
            "helpdesk_db_url": f"sqlite+aiosqlite:///{(data_dir / 'helpdesk.db').as_posix()}",
        },
        agent={"enable_long_term_memory": True},
    )


@pytest.fixture(scope="session")
def mcp_servers(settings: Settings) -> Iterator[Settings]:
    commerce_db = Database(settings.mcp.commerce_db_url)
    asyncio.run(seed_commerce(commerce_db))
    asyncio.run(commerce_db.dispose())
    commerce = create_commerce_server(settings).http_app(stateless_http=True)
    helpdesk = create_helpdesk_server(settings).http_app(stateless_http=True)
    with _ServerThread(commerce, settings.mcp.commerce_port), _ServerThread(helpdesk, settings.mcp.helpdesk_port):
        yield settings


@pytest.fixture(autouse=True)
async def _fresh_commerce_data(request: pytest.FixtureRequest) -> None:
    """Tests that move money or cancel orders must not see each other's side effects:
    reseed the (tiny) SQLite demo database before every test that uses the MCP servers."""
    if "mcp_servers" not in request.fixturenames:
        return
    s: Settings = request.getfixturevalue("mcp_servers")
    db = Database(s.mcp.commerce_db_url)
    try:
        await seed_commerce(db, reset=True)
    finally:
        await db.dispose()


@pytest.fixture(scope="session")
async def embeddings(settings: Settings) -> Any:
    from caseflow.rag.embeddings import EmbeddingModels

    models = EmbeddingModels(settings.retrieval)
    await asyncio.to_thread(models.warmup)
    return models


@pytest.fixture(scope="session")
async def retriever(settings: Settings, embeddings: Any) -> AsyncIterator[Any]:
    from caseflow.rag.ingest import ingest_knowledge_base
    from caseflow.rag.retriever import HybridRetriever
    from caseflow.rag.stores.qdrant import QdrantHybridStore

    store = QdrantHybridStore(collection="kb_test", path=Path(":memory:"))
    await ingest_knowledge_base(settings, store, embeddings, force=True)
    yield HybridRetriever(store, embeddings, settings.retrieval)
    await store.close()
