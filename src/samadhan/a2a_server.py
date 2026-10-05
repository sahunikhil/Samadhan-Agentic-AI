"""Samadhan as an A2A (Agent2Agent protocol 1.0) remote agent.

MCP vs A2A
----------
* **MCP** connects an agent to *tools and data*: a function call with a schema. The caller's
  model decides what to call; the server is a capability.
* **A2A** connects an agent to *another agent*. The remote agent is opaque (its prompts, tools and
  model are its own business), work is a **task with a lifecycle** (submitted -> working ->
  input-required / auth-required / completed / failed / canceled / rejected), turns are grouped by
  ``context_id``, and capabilities are advertised in an **agent card** at
  ``/.well-known/agent-card.json``.

Samadhan uses both: it *consumes* MCP servers (commerce, helpdesk) and *exposes* itself over A2A,
so a retailer's shopping assistant, a CRM copilot or a partner's agent can delegate "sort out this
customer's return" to it without knowing anything about LangGraph.

Mapping Samadhan onto A2A
-------------------------
========================  ===========================================================
A2A                       Samadhan
========================  ===========================================================
bearer token (the card    the same customer session token as the REST API. Missing or
lists an http bearer      invalid -> HTTP 401 before any JSON-RPC handling. The verified
scheme)                   ``sub`` becomes the customer - never a message field.
task owner                the customer id: the task store is scoped per customer, so one
                          customer cannot ``tasks/get`` another customer's task
``context_id``            a conversation thread ``<customer>--a2a-<sha256(context_id)>``
message                   one ``SupportService`` turn
``input-required``        a pending *customer* interrupt (MCP elicitation, e.g. confirm a
                          cancellation); the caller's next message ("yes"/"no") resumes it
``completed``             the reply as a text artifact. Staff-side waits (refund approval,
                          human handoff) also complete, with a status note: Voltwise staff
                          resolve them asynchronously, the calling agent cannot
``failed``                an error inside the turn (no stack trace crosses the boundary)
========================  ===========================================================
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import UnauthenticatedUser, User
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import DefaultServerCallContextBuilder, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    HTTPAuthSecurityScheme,
    Message,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from samadhan.api.security import verify_customer_token
from samadhan.config import Settings
from samadhan.observability import get_logger
from samadhan.service import SupportService

log = get_logger(__name__)

A2A_PATH = "/a2a"
_YES = re.compile(r"\b(yes|yeah|yep|confirm(ed)?|go ahead|do it|please do|sure|ok(ay)?)\b", re.IGNORECASE)
_NO = re.compile(r"\b(no|nope|don'?t|do not|keep it|stop)\b", re.IGNORECASE)


# ---- agent card ---------------------------------------------------------------------------------
def build_agent_card(settings: Settings) -> AgentCard:
    base = settings.api.public_url.rstrip("/")
    skills = [
        AgentSkill(
            id="order-support",
            name="Order status & changes",
            description="Track Voltwise orders, detect stalled parcels, cancel unshipped orders "
            "(the customer confirms first) and change shipping addresses.",
            tags=["orders", "tracking", "cancellation"],
            examples=["Where is my order VW-10003?", "Please cancel order VW-10005."],
        ),
        AgentSkill(
            id="returns-refunds",
            name="Returns & refunds",
            description="Return eligibility with exact fees, returns, refunds and price adjustments. "
            "Refunds above the auto-approve limit are approved by a Voltwise specialist.",
            tags=["returns", "refunds", "price-adjustment"],
            examples=["How much would I get back if I return my VoltBook Pro 16?"],
        ),
        AgentSkill(
            id="product-policy-qa",
            name="Product & policy answers",
            description="Grounded, cited answers from the Voltwise help center: specs, compatibility, "
            "troubleshooting, shipping, warranty and payment policies.",
            tags=["knowledge", "rag"],
            examples=["Can I bring the PowerCell 20K on a plane?"],
        ),
    ]
    return AgentCard(
        name="Voltwise Support Agent (Samadhan)",
        description="Resolves Voltwise customer-support requests end to end on behalf of an authenticated customer.",
        version="0.1.0",
        provider=AgentProvider(organization="Voltwise (fictional retailer)", url=base),
        documentation_url=f"{base}/docs",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=f"{base}{A2A_PATH}", protocol_version="1.0")
        ],
        security_schemes={
            "customerBearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer", bearer_format="JWT", description="Voltwise customer session token"
                )
            )
        },
        security_requirements=[SecurityRequirement(schemes={"customerBearer": StringList()})],  # bearer, no scopes
        skills=skills,
    )


# ---- identity -----------------------------------------------------------------------------------
class CustomerUser(User):
    """An authenticated customer. ``user_name`` is the task-store owner key."""

    def __init__(self, customer_id: str) -> None:
        self._customer_id = customer_id

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self._customer_id


class CustomerContextBuilder(DefaultServerCallContextBuilder):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_user(self, request: Request) -> User:
        customer_id = verify_customer_token(self._settings, request.headers.get("authorization"))
        return CustomerUser(customer_id) if customer_id else UnauthenticatedUser()


class A2AAuthMiddleware:
    """Transport-level guard for the JSON-RPC endpoint: 401 without a valid token, 429 when over the
    per-customer rate limit (the same limiter as the REST API). The agent card stays public - discovery
    must work before authentication."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == A2A_PATH and scope["method"] == "POST":
            customer_id = verify_customer_token(self.settings, Headers(scope=scope).get("authorization"))
            if customer_id is None:
                response = JSONResponse(
                    {"detail": "A valid customer bearer token is required."},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="samadhan-a2a"'},
                )
                await response(scope, receive, send)
                return
            limiter = getattr(scope["app"].state, "rate_limiter", None)
            if limiter is not None:
                try:
                    limiter.check(f"customer:{customer_id}")
                except HTTPException as exc:
                    await JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)(
                        scope, receive, send
                    )
                    return
        await self.app(scope, receive, send)


# ---- execution ----------------------------------------------------------------------------------
def a2a_thread_id(customer_id: str, context_id: str) -> str:
    """Namespaced by customer: the same ``context_id`` from two customers never shares a thread."""
    return f"{customer_id}--a2a-{hashlib.sha256(context_id.encode()).hexdigest()[:12]}"


def parse_confirmation(text: str) -> bool | None:
    """Map a free-text answer to yes/no; None when unclear (ask again rather than guess)."""
    yes, no = bool(_YES.search(text)), bool(_NO.search(text))
    if yes != no:
        return yes
    return None


class SamadhanAgentExecutor(AgentExecutor):
    def __init__(self, get_service: Callable[[], SupportService]) -> None:
        self._get_service = get_service

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None:
            raise ValueError("message/send without a message")
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)

        user = context.call_context.user
        if not user.is_authenticated:  # defense in depth; A2AAuthMiddleware already returns 401
            await updater.requires_auth(message=new_text_message("A Voltwise customer bearer token is required."))
            return
        customer_id = user.user_name
        thread_id = a2a_thread_id(customer_id, task.context_id)
        text = get_message_text(message) or ""
        await updater.start_work(message=new_text_message("Samadhan is working on it..."))
        svc = self._get_service()

        try:
            pending = (await svc.thread_view(thread_id))["pending"]
            confirmations = [p for p in pending if p["kind"] == "customer_confirmation"]
            if confirmations:
                answer = parse_confirmation(text)
                if answer is None:
                    await updater.requires_input(message=_confirm_prompt(confirmations[0]["title"]))
                    return
                result = await svc.resume(
                    thread_id=thread_id,
                    decisions={p["interrupt_id"]: {"accept": answer} for p in confirmations},
                    actor="customer",
                    actor_name=customer_id,
                    customer_id=customer_id,
                )
            elif pending:
                # Waiting for Voltwise staff; a new message must not start a parallel run on the thread.
                await updater.complete(message=new_text_message(_STAFF_NOTE))
                return
            else:
                result = await svc.run_turn(customer_id=customer_id, message=text, thread_id=thread_id, channel="a2a")
        except Exception as exc:  # the remote agent gets a failed task, never a stack trace
            log.exception("a2a_turn_failed", thread_id=thread_id)
            await updater.failed(
                message=new_text_message(f"Samadhan could not complete the request ({type(exc).__name__}).")
            )
            return

        customer_waits = [p for p in result.pending if p.kind == "customer_confirmation"]
        if customer_waits:
            await updater.requires_input(message=_confirm_prompt(customer_waits[0].title))
        elif result.pending:
            await updater.add_artifact(parts=[new_text_part(text=_STAFF_NOTE, media_type="text/plain")], name="status")
            await updater.complete()
        else:
            reply = result.reply or "Done."
            await updater.add_artifact(parts=[new_text_part(text=reply, media_type="text/plain")], name="reply")
            await updater.complete()
        log.info("a2a_turn", thread_id=thread_id, pending=[p.kind for p in result.pending])

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is not None:
            updater = TaskUpdater(event_queue, context.current_task.id, context.current_task.context_id)
            await updater.cancel(message=new_text_message("Canceled by the caller."))


_STAFF_NOTE = (
    "This request is with a Voltwise specialist (for example a refund above the auto-approve limit, "
    "or a human handoff). The customer will be notified when it is resolved."
)


def _confirm_prompt(title: str) -> Message:
    return new_text_message(f"{title} Please answer yes or no.")


def create_a2a_routes(
    settings: Settings, get_service: Callable[[], SupportService], task_store: TaskStore | None = None
) -> list[Any]:
    """Starlette routes for the agent card and the JSON-RPC endpoint (mount on the FastAPI app).

    ``task_store``: the API passes a2a-sdk's SQL ``DatabaseTaskStore`` on its database (Postgres in
    production), so every replica sees every task; owner scoping per customer is kept."""
    card = build_agent_card(settings)
    handler = DefaultRequestHandler(
        agent_executor=SamadhanAgentExecutor(get_service),
        task_store=task_store or InMemoryTaskStore(),
        agent_card=card,
    )
    return [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, A2A_PATH, context_builder=CustomerContextBuilder(settings)),
    ]
