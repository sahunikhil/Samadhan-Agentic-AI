"""Async SQLAlchemy plumbing shared by the commerce and helpdesk servers.

Both servers run on SQLite locally (zero setup) and on Postgres in production -
the same ORM code, only ``*_DB_URL`` changes. All timestamps are stored as naive
UTC to behave identically on both engines.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Database:
    def __init__(self, url: str) -> None:
        if url.startswith("sqlite") and ":///" in url:
            db_path = url.split(":///", 1)[1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        self.engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        self._sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_all(self, base: type[DeclarativeBase]) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def dispose(self) -> None:
        await self.engine.dispose()
