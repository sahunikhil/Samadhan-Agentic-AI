"""Authentication, authorization and rate limiting for the HTTP API.

Two kinds of callers:

* **Customers** - ``Authorization: Bearer <customer session JWT>``. In production this
  token comes from your identity provider (Auth0, Keycloak, Cognito, Entra ...); the
  demo issues one from ``/v1/auth/demo-login`` when ``demo_mode`` is on.
* **Staff** (supervisors / human agents) - ``X-Admin-Key``. Staff can approve refunds,
  answer human handoffs and read any thread. Replace with SSO + roles in production.
"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Annotated, Literal

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from caseflow.config import Settings

ALGORITHM = "HS256"


@dataclass(frozen=True, slots=True)
class Principal:
    kind: Literal["customer", "staff"]
    id: str

    @property
    def customer_id(self) -> str | None:
        return self.id if self.kind == "customer" else None


def issue_customer_token(settings: Settings, customer_id: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": "caseflow-api",
            "aud": settings.api.customer_token_audience,
            "sub": customer_id,
            "iat": now,
            "exp": now + settings.api.customer_token_ttl_s,
            "role": "customer",
        },
        settings.api.token_secret.get_secret_value(),
        algorithm=ALGORITHM,
    )


def verify_customer_token(settings: Settings, authorization: str | None) -> str | None:
    """Return the customer id of a valid ``Bearer`` customer token, else None.

    Shared by the REST API and the A2A endpoint so both enforce exactly the same rules
    (signature, expiry, audience, issuer).
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        claims = jwt.decode(
            authorization.split(" ", 1)[1],
            settings.api.token_secret.get_secret_value(),
            algorithms=[ALGORITHM],
            audience=settings.api.customer_token_audience,
            issuer="caseflow-api",
        )
    except jwt.PyJWTError:
        return None
    return str(claims["sub"])


def _settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


async def get_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_admin_key: Annotated[str | None, Header()] = None,
    x_staff_name: Annotated[str | None, Header()] = None,
) -> Principal:
    settings = _settings(request)
    if x_admin_key is not None:
        # Constant-time comparison: no timing side channel on the key.
        if not hmac.compare_digest(x_admin_key, settings.api.admin_api_key.get_secret_value()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid staff key")
        return Principal(kind="staff", id=(x_staff_name or "supervisor")[:40])
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    customer_id = verify_customer_token(settings, authorization)
    if customer_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return Principal(kind="customer", id=customer_id)


def require_staff(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    if principal.kind != "staff":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Staff access required")
    return principal


def require_customer(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    if principal.kind != "customer":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Customer session required")
    return principal


class SlidingWindowRateLimiter:
    """Per-principal sliding-window limiter (in-process).

    Protects the free-tier LLM quota from a single noisy client. With several API
    replicas, move this to Redis (or the API gateway) so the limit is global.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            retry = int(60 - (now - window[0])) + 1
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Too many messages - please slow down.",
                headers={"Retry-After": str(retry)},
            )
        window.append(now)


def rate_limited(request: Request, principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    limiter: SlidingWindowRateLimiter = request.app.state.rate_limiter
    limiter.check(f"{principal.kind}:{principal.id}")
    return principal


class EdgeGuardMiddleware:
    """Cheap, early rejections before any route or mounted app runs (pure ASGI, so it also covers
    the mounted MCP server and streaming responses):

    * request bodies over ``max_body_bytes`` -> 413 (declared ``Content-Length`` checked up front;
      chunked bodies counted as they arrive), so a huge payload is never buffered for validation;
    * per-IP sliding-window limit on unauthenticated, CPU-heavy routes (the public knowledge MCP
      server runs embedding + a cross-encoder per call; demo login mints tokens) -> 429.

    Behind a reverse proxy, run uvicorn with ``--forwarded-allow-ips`` set to the proxy so the
    client IP is the real one; otherwise all traffic shares the proxy's bucket (fails safe: stricter).
    """

    def __init__(
        self, app: ASGIApp, *, max_body_bytes: int, ip_limiter: SlidingWindowRateLimiter, ip_limited: tuple[str, ...]
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.ip_limiter = ip_limiter
        self.ip_limited = ip_limited

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_body_bytes:
            await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, send)
            return
        if scope["path"].startswith(self.ip_limited):
            client = scope.get("client") or ("unknown", 0)
            try:
                self.ip_limiter.check(f"ip:{client[0]}")
            except HTTPException as exc:
                await JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)(
                    scope, receive, send
                )
                return
        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Request body too large")
            return message

        await self.app(scope, counted_receive, send)
