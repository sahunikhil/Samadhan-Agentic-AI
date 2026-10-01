"""Request / response models (the public HTTP contract, versioned under /v1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DemoLoginRequest(BaseModel):
    customer_id: str = Field(pattern=r"^cust_\d{3}$", examples=["cust_001"])


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth token type, not a secret
    customer: dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, examples=["Where is my order VW-10003?"])
    thread_id: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_\-]+$")


class ResumeRequest(BaseModel):
    """Answers to pending interrupts, keyed by interrupt id.

    * refund approval (supervisor): ``{"decision": "approve"}`` or ``{"decision": "reject", "message": "..."}``
    * customer confirmation:        ``{"accept": true}`` / ``{"accept": false}``
    * human handoff (agent):        ``{"action": "reply", "message": "...", "agent_name": "Jordan"}`` or ``{"action": "defer"}``
    """

    decisions: dict[str, dict[str, Any]] = Field(min_length=1)


class PendingActionOut(BaseModel):
    interrupt_id: str
    kind: str
    audience: str
    title: str
    detail: dict[str, Any]


class TurnResponse(BaseModel):
    thread_id: str
    reply: str | None
    outcome: str | None
    pending: list[PendingActionOut] = Field(default_factory=list)
    intents: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
