"""Helpdesk MCP server - tickets, human escalation and case analytics.

Used two ways, which is typical for MCP in production:

* by **LLM agents** (e.g. "what's the status of my ticket?"), and
* by **deterministic workflow code** in the graph (the escalation node creates a
  ticket; the finalize node logs every case) - MCP is just a well-typed RPC
  boundary, it doesn't require an LLM on the calling side.

``case_log`` is what product analytics runs on: automated resolution rate,
escalation rate and intent mix per day.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import require_scopes
from fastmcp.server.dependencies import TokenClaim
from mcp.types import ToolAnnotations
from pydantic import BaseModel
from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from starlette.requests import Request
from starlette.responses import JSONResponse

from caseflow.config import Settings, get_settings
from caseflow.mcp_servers.auth import build_verifier
from caseflow.mcp_servers.db import Database, utcnow

Priority = Literal["low", "normal", "high", "urgent"]
FIRST_RESPONSE_TARGET = {
    "urgent": "1 business hour",
    "high": "4 business hours",
    "normal": "1 business day",
    "low": "2 business days",
}


class HelpdeskBase(DeclarativeBase):
    pass


class Ticket(HelpdeskBase):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    subject: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32))
    priority: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | pending | solved
    comments: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class CaseLog(HelpdeskBase):
    __tablename__ = "case_log"

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    intents: Mapped[list[str]] = mapped_column(JSON, default=list)
    outcome: Mapped[str] = mapped_column(String(16))  # resolved | escalated | blocked
    automated: Mapped[bool] = mapped_column(Boolean, default=True)
    turns: Mapped[int] = mapped_column(Integer, default=1)
    summary: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class TicketOut(BaseModel):
    ticket_id: str
    subject: str
    status: str
    priority: str
    category: str
    created_at: str
    first_response_target: str
    comments: list[dict[str, Any]] = []


def _out(t: Ticket) -> TicketOut:
    return TicketOut(
        ticket_id=t.id, subject=t.subject, status=t.status, priority=t.priority, category=t.category,
        created_at=t.created_at.isoformat(timespec="minutes"),
        first_response_target=FIRST_RESPONSE_TARGET.get(t.priority, "1 business day"),
        comments=t.comments or [],
    )  # fmt: skip


def create_helpdesk_server(settings: Settings | None = None, db: Database | None = None) -> FastMCP:
    settings = settings or get_settings()
    database = db or Database(settings.mcp.helpdesk_db_url)

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncIterator[dict[str, Any]]:
        await database.create_all(HelpdeskBase)
        yield {"db": database}

    mcp = FastMCP(
        name="voltwise-helpdesk",
        instructions="Voltwise help desk: create and look up support tickets for the signed-in customer.",
        auth=build_verifier(settings.mcp),
        lifespan=lifespan,
        mask_error_details=True,
    )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
        ),
        auth=require_scopes("tickets:write"),
    )
    async def create_ticket(
        subject: str,
        description: str,
        category: Literal["orders", "returns", "billing", "technical", "account", "complaint", "other"] = "other",
        priority: Priority = "normal",
        conversation_id: str | None = None,
        customer_id: str = TokenClaim("sub"),
    ) -> TicketOut:
        """Open a support ticket for a human specialist. Idempotent per conversation: if an open ticket
        already exists for this conversation it is returned instead of creating a duplicate."""
        async with database.session() as s:
            if conversation_id:
                existing = await s.scalar(
                    select(Ticket).where(
                        Ticket.conversation_id == conversation_id,
                        Ticket.customer_id == customer_id,
                        Ticket.status != "solved",
                    )
                )
                if existing:
                    return _out(existing)
            now = utcnow()  # set explicitly: column defaults only apply at flush time
            ticket = Ticket(
                id=f"TCK-{secrets.randbelow(900000) + 100000}",
                customer_id=customer_id,
                conversation_id=conversation_id,
                subject=subject[:200],
                description=description[:4000],
                category=category,
                priority=priority,
                status="open",
                comments=[],
                created_at=now,
                updated_at=now,
            )
            s.add(ticket)
            return _out(ticket)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False), auth=require_scopes("tickets:read"))
    async def list_my_tickets(
        status: Literal["open", "pending", "solved"] | None = None, customer_id: str = TokenClaim("sub")
    ) -> list[TicketOut]:
        """List the customer's support tickets, newest first."""
        async with database.session() as s:
            q = select(Ticket).where(Ticket.customer_id == customer_id).order_by(Ticket.created_at.desc())
            if status:
                q = q.where(Ticket.status == status)
            return [_out(t) for t in (await s.scalars(q.limit(20))).all()]

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False), auth=require_scopes("tickets:read"))
    async def get_ticket(ticket_id: str, customer_id: str = TokenClaim("sub")) -> TicketOut:
        """Get a ticket's status, priority and conversation history."""
        async with database.session() as s:
            t = await s.get(Ticket, ticket_id.strip().upper())
            if t is None or t.customer_id != customer_id:
                raise ToolError(f"Ticket {ticket_id} was not found on this account.")
            return _out(t)

    @mcp.tool(
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False),
        auth=require_scopes("tickets:write"),
    )
    async def add_ticket_comment(
        ticket_id: str, comment: str, author: Literal["customer", "assistant", "agent"] = "assistant",
        customer_id: str = TokenClaim("sub"),
    ) -> TicketOut:  # fmt: skip
        """Append a comment to an existing ticket."""
        async with database.session() as s:
            t = await s.get(Ticket, ticket_id.strip().upper())
            if t is None or t.customer_id != customer_id:
                raise ToolError(f"Ticket {ticket_id} was not found on this account.")
            t.comments = [
                *(t.comments or []),
                {"at": utcnow().isoformat(timespec="minutes"), "author": author, "text": comment[:2000]},
            ]
            t.updated_at = utcnow()
            if author == "agent":
                t.status = "pending"
            return _out(t)

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
        ),
        auth=require_scopes("tickets:write"),
    )
    async def log_case(
        conversation_id: str,
        intents: list[str],
        outcome: Literal["resolved", "escalated", "blocked"],
        summary: str = "",
        customer_id: str = TokenClaim("sub"),
    ) -> dict[str, Any]:
        """Record the outcome of a conversation turn for analytics (upsert by conversation)."""
        async with database.session() as s:
            row = await s.get(CaseLog, conversation_id)
            if row is None:
                row = CaseLog(conversation_id=conversation_id, customer_id=customer_id, turns=0)
                s.add(row)
            row.intents = sorted(set((row.intents or []) + intents))
            row.outcome = outcome
            row.automated = outcome != "escalated" and (row.automated if row.turns else True)
            row.turns = (row.turns or 0) + 1
            row.summary = summary[:1000]
            row.updated_at = utcnow()
            return {"conversation_id": conversation_id, "turns": row.turns, "outcome": outcome}

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "helpdesk"})

    @mcp.custom_route("/stats", methods=["GET"])
    async def stats(_: Request) -> JSONResponse:
        """Operational KPIs (no PII): automation rate and escalations."""
        async with database.session() as s:
            total = await s.scalar(select(func.count()).select_from(CaseLog)) or 0
            escalated = (
                await s.scalar(select(func.count()).select_from(CaseLog).where(CaseLog.outcome == "escalated")) or 0
            )
            open_tickets = (
                await s.scalar(select(func.count()).select_from(Ticket).where(Ticket.status != "solved")) or 0
            )
        return JSONResponse({
            "conversations": total,
            "escalated": escalated,
            "automated_resolution_rate": round((total - escalated) / total, 3) if total else None,
            "open_tickets": open_tickets,
        })  # fmt: skip

    return mcp
