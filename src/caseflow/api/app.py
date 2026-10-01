"""FastAPI application.

Run:  ``uv run caseflow serve api``  (or ``uvicorn caseflow.api.app:create_app --factory``)

Endpoints (OpenAPI docs at ``/docs``):

====================================  =========  ==================================================
``POST /v1/auth/demo-login``          public     demo customer session (demo_mode only)
``POST /v1/chat``                     customer   one turn, JSON response (``Idempotency-Key`` supported)
``POST /v1/chat/stream``              customer   one turn, Server-Sent Events
``POST /v1/threads/{id}/resume``      cust/staff answer pending interrupts (SSE)
``GET  /v1/threads/{id}``             cust/staff thread state + pending actions
``GET  /v1/threads/{id}/history``     staff      checkpoint timeline (time travel)
``POST /v1/threads/{id}/feedback``    cust/staff thumbs up/down on the latest reply
``GET  /v1/admin/feedback``           staff      ratings for review / dataset harvesting
``POST /v1/admin/ingest``             staff      re-index the knowledge base
``/mcp/knowledge``                    public     knowledge MCP server (Streamable HTTP)
``GET  /.well-known/agent-card.json`` public     A2A agent card (discovery)
``POST /a2a``                         customer   A2A JSON-RPC endpoint (agent-to-agent)
``GET  /healthz`` ``/readyz``          public     liveness / readiness probes
``GET  /metrics``                     public     Prometheus metrics
====================================  =========  ==================================================
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse

from caseflow.a2a_server import A2AAuthMiddleware, create_a2a_routes
from caseflow.api.idempotency import IdempotencyStore, fingerprint
from caseflow.api.schemas import ChatRequest, DemoLoginRequest, LoginResponse, ResumeRequest, TurnResponse
from caseflow.api.security import (
    EdgeGuardMiddleware,
    Principal,
    SlidingWindowRateLimiter,
    get_principal,
    issue_customer_token,
    rate_limited,
    require_staff,
)
from caseflow.bootstrap import Container, build_container
from caseflow.config import Settings, get_settings
from caseflow.feedback import FeedbackError, FeedbackRequest, list_feedback, record_feedback
from caseflow.llm import ModelRegistry
from caseflow.mcp_servers.commerce.seed import CUSTOMERS
from caseflow.mcp_servers.knowledge.server import create_knowledge_server
from caseflow.observability import HTTP_LATENCY, HTTP_REQUESTS, get_logger
from caseflow.rag.ingest import ingest_knowledge_base
from caseflow.rag.retriever import HybridRetriever
from caseflow.resilience import BREAKERS
from caseflow.service import Actor, SupportService, ThreadAccessError, TurnResult
from caseflow.telemetry import instrument_asgi, setup_tracing

log = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


# Second layer behind output escaping in the UI: even if an XSS slipped through, the page can't
# send data to another origin (connect-src / img-src 'self') and can't be framed - the staff console
# approves refunds, so clickjacking matters (frame-ancestors 'none'). The single-file UI uses inline
# script/style, hence 'unsafe-inline' there; a bundled UI should switch to nonces.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _service(request: Request) -> SupportService:
    return request.app.state.service  # type: ignore[no-any-return]


# Module level on purpose: with `from __future__ import annotations` FastAPI resolves
# annotation strings against module globals - a dependency alias defined inside
# create_app() would silently become a query parameter.
Svc = Annotated[SupportService, Depends(_service)]


def _turn_response(result: TurnResult) -> TurnResponse:
    triage = result.triage or {}
    return TurnResponse(
        thread_id=result.thread_id,
        reply=result.reply,
        outcome=result.outcome,
        pending=[p.__dict__ for p in result.pending],  # type: ignore[misc]
        intents=triage.get("intents", []),
        agents=sorted({r["agent"] for r in result.specialist_results}),
    )


def _sse(events: AsyncIterator[dict[str, Any]]) -> EventSourceResponse:
    async def gen() -> AsyncIterator[dict[str, str]]:
        async for event in events:
            yield {"event": event["type"], "data": json.dumps(event, default=str)}

    # ping keeps proxies/load balancers from closing idle streams during long tool calls
    return EventSourceResponse(gen(), ping=15)


def create_app(
    settings: Settings | None = None, *, models: ModelRegistry | None = None, **container_kwargs: Any
) -> FastAPI:
    """App factory. ``models`` / ``container_kwargs`` exist for tests (fake LLMs, in-memory persistence)."""
    settings = settings or get_settings()
    state: dict[str, Any] = {}

    async def get_retriever() -> HybridRetriever:
        container: Container = state["container"]
        return container.retriever

    knowledge_mcp_app = create_knowledge_server(get_retriever, settings).http_app(path="/", stateless_http=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with (
            build_container(settings, models=models, **container_kwargs) as container,
            knowledge_mcp_app.lifespan(app),
        ):
            state["container"] = container
            app.state.container = container
            app.state.service = SupportService(container)
            app.state.idempotency = IdempotencyStore(container.store)
            log.info("api_ready", port=settings.api.port)
            yield
            drained = app.state.service.drain_all("shutdown")
            if drained:
                # Give in-flight runs a moment to reach a superstep boundary and checkpoint.
                log.info("draining_runs", count=drained)
                await asyncio.sleep(2)

    app = FastAPI(
        title="CaseFlow API",
        version="0.1.0",
        description="Multi-agent customer-support resolution platform (LangGraph + MCP + hybrid RAG).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rate_limiter = SlidingWindowRateLimiter(settings.api.rate_limit_per_minute)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Admin-Key", "X-Staff-Name"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        response: Response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(route=path, method=request.method, status=str(response.status_code)).inc()
        HTTP_LATENCY.labels(route=path).observe(time.perf_counter() - started)
        response.headers["x-request-id"] = request_id
        response.headers.update(SECURITY_HEADERS)
        structlog.contextvars.clear_contextvars()
        return response

    # ---- ops ------------------------------------------------------------------------------
    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request) -> dict[str, Any]:
        container: Container = request.app.state.container
        checks: dict[str, Any] = {"vector_chunks": await container.vector_store.count()}
        async with httpx.AsyncClient(timeout=3) as client:
            for name, url in (("commerce", settings.mcp.commerce_url), ("helpdesk", settings.mcp.helpdesk_url)):
                try:
                    r = await client.get(url.rsplit("/mcp", 1)[0] + "/health")
                    checks[name] = r.status_code == 200
                except httpx.HTTPError:
                    checks[name] = False
        # Readiness = can *this pod* serve? Only local capabilities gate it (the knowledge index).
        # Shared downstream dependencies (MCP servers, open circuits) only *degrade* it: failing
        # readiness on them would pull every replica out of the load balancer at once - a total
        # outage, including the knowledge answers that still work.
        if checks["vector_chunks"] == 0:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, checks)
        circuits = BREAKERS.snapshot()
        degraded = not (checks["commerce"] and checks["helpdesk"]) or any(s != "closed" for s in circuits.values())
        return {"status": "degraded" if degraded else "ready", **checks, "circuits": circuits}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ---- auth (demo) --------------------------------------------------------------------------
    @app.get("/v1/demo/customers", tags=["auth"])
    async def demo_customers() -> list[dict[str, str]]:
        if not settings.api.demo_mode:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return [{"customer_id": c[0], "name": c[1], "tier": c[3]} for c in CUSTOMERS]

    @app.post("/v1/auth/demo-login", response_model=LoginResponse, tags=["auth"])
    async def demo_login(body: DemoLoginRequest, request: Request) -> LoginResponse:
        if not settings.api.demo_mode:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        container: Container = request.app.state.container
        try:
            profile = await container.toolkit.call(body.customer_id, "commerce", "get_customer_profile", {})
        except Exception as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown customer") from exc
        return LoginResponse(access_token=issue_customer_token(settings, body.customer_id), customer=profile)

    # ---- chat ---------------------------------------------------------------------------------
    @app.post("/v1/chat", response_model=TurnResponse, tags=["chat"])
    async def chat(
        body: ChatRequest,
        svc: Svc,
        request: Request,
        response: Response,
        principal: Annotated[Principal, Depends(rate_limited)],
        idempotency_key: Annotated[str | None, Header(description="Makes client retries safe (24 h)")] = None,
    ) -> TurnResponse:
        if principal.kind != "customer":
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Customer session required")
        idem: IdempotencyStore = request.app.state.idempotency
        body_hash = fingerprint(body.model_dump())
        if idempotency_key is not None:
            IdempotencyStore.validate(idempotency_key)
            kind, stored = await idem.begin(principal.id, idempotency_key, body_hash)
            if kind == "replay":
                response.headers["Idempotent-Replayed"] = "true"
                return TurnResponse.model_validate(stored)
        try:
            result = await svc.run_turn(customer_id=principal.id, message=body.message, thread_id=body.thread_id)
        except BaseException as exc:
            if idempotency_key is not None:
                await idem.release(principal.id, idempotency_key)
            if isinstance(exc, ThreadAccessError):
                raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
            raise
        out = _turn_response(result)
        if idempotency_key is not None:
            if result.outcome == "error":  # a failed turn is not a result worth replaying: allow a retry
                await idem.release(principal.id, idempotency_key)
            else:
                await idem.complete(principal.id, idempotency_key, body_hash, out.model_dump(mode="json"))
        return out

    @app.post("/v1/chat/stream", tags=["chat"])
    async def chat_stream(
        body: ChatRequest, svc: Svc, principal: Annotated[Principal, Depends(rate_limited)]
    ) -> EventSourceResponse:
        if principal.kind != "customer":
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Customer session required")
        try:
            if body.thread_id:
                svc.check_access(body.thread_id, principal.id)
        except ThreadAccessError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        return _sse(svc.stream_turn(customer_id=principal.id, message=body.message, thread_id=body.thread_id))

    # ---- threads & human-in-the-loop ------------------------------------------------------------
    @app.get("/v1/threads/{thread_id}", tags=["threads"])
    async def get_thread(
        thread_id: str, svc: Svc, principal: Annotated[Principal, Depends(get_principal)]
    ) -> dict[str, Any]:
        try:
            svc.check_access(thread_id, principal.customer_id)
        except ThreadAccessError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        view = await svc.thread_view(thread_id)
        if principal.kind == "customer":  # customers never see internal traces
            view.pop("specialist_results", None)
            view.pop("triage", None)
        return view

    @app.get("/v1/threads/{thread_id}/history", tags=["threads"])
    async def thread_history(
        thread_id: str, svc: Svc, _: Annotated[Principal, Depends(require_staff)]
    ) -> list[dict[str, Any]]:
        return await svc.history(thread_id)

    @app.post("/v1/threads/{thread_id}/resume", tags=["threads"])
    async def resume(
        thread_id: str, body: ResumeRequest, svc: Svc, principal: Annotated[Principal, Depends(rate_limited)]
    ) -> EventSourceResponse:
        actor: Actor = "supervisor" if principal.kind == "staff" else "customer"
        try:
            svc.check_access(thread_id, principal.customer_id)
            # Validate before streaming so bad requests get a proper HTTP status, not an SSE error.
            await svc._build_resume(thread_id, body.decisions, actor=actor, actor_name=principal.id)
        except ThreadAccessError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return _sse(
            svc.stream_resume(
                thread_id=thread_id,
                decisions=body.decisions,
                actor=actor,
                actor_name=principal.id,
                customer_id=principal.customer_id,
            )
        )

    @app.post("/v1/threads/{thread_id}/feedback", status_code=status.HTTP_201_CREATED, tags=["threads"])
    async def feedback(
        thread_id: str,
        body: FeedbackRequest,
        svc: Svc,
        principal: Annotated[Principal, Depends(get_principal)],
    ) -> dict[str, str]:
        try:
            svc.check_access(thread_id, principal.customer_id)
            await record_feedback(svc.c.store, svc.c.graph, thread_id, body, actor=f"{principal.kind}:{principal.id}")
        except ThreadAccessError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        except FeedbackError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return {"status": "recorded"}

    @app.get("/v1/admin/feedback", tags=["admin"])
    async def feedback_list(
        svc: Svc,
        _: Annotated[Principal, Depends(require_staff)],
        rating: Literal["up", "down"] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return await list_feedback(svc.c.store, rating=rating, limit=min(limit, 500))

    # ---- admin ----------------------------------------------------------------------------------
    @app.post("/v1/admin/ingest", tags=["admin"])
    async def reingest(
        request: Request, _: Annotated[Principal, Depends(require_staff)], force: bool = False
    ) -> dict[str, Any]:
        c: Container = request.app.state.container
        report = await ingest_knowledge_base(settings, c.vector_store, c.embeddings, force=force)
        return report.__dict__

    # ---- MCP + A2A + UI -------------------------------------------------------------------------
    app.mount("/mcp/knowledge", knowledge_mcp_app)
    if settings.api.a2a_enabled:
        # MCP above exposes a *tool server*; A2A exposes the whole support agent to other agents.
        app.router.routes.extend(create_a2a_routes(settings, lambda: app.state.service))
        app.add_middleware(A2AAuthMiddleware, settings=settings)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # Added last = outermost: oversized bodies and per-IP floods are rejected before anything else runs.
    app.add_middleware(
        EdgeGuardMiddleware,
        max_body_bytes=settings.api.max_body_bytes,
        ip_limiter=SlidingWindowRateLimiter(settings.api.ip_rate_limit_per_minute),
        ip_limited=("/mcp/knowledge", "/v1/auth/demo-login"),
    )
    setup_tracing(settings, "caseflow-api")
    return instrument_asgi(app)
