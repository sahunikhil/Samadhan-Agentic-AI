"""Authentication, authorization and rate limiting for the HTTP API.

Two kinds of callers:

* **Customers** - ``Authorization: Bearer <customer session JWT>``. In production this
  token comes from your identity provider (Auth0, Keycloak, Cognito, Entra ...); the
  demo issues one from ``/v1/auth/demo-login`` when ``demo_mode`` is on.
* **Staff** (supervisors / human agents) - an SSO token from your identity provider (OIDC; roles
  ``supervisor`` / ``agent`` / ``admin``), or the break-glass ``X-Admin-Key``. Staff can approve refunds,
  answer human handoffs and read any thread. Replace with SSO + roles in production.
"""

from __future__ import annotations

import asyncio
import hmac
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from samadhan.config import Settings

ALGORITHM = "HS256"


StaffRole = Literal["supervisor", "agent", "admin"]
STAFF_ROLES: frozenset[str] = frozenset({"supervisor", "agent", "admin"})


@dataclass(frozen=True, slots=True)
class Principal:
    kind: Literal["customer", "staff"]
    id: str
    roles: frozenset[str] = frozenset()

    @property
    def customer_id(self) -> str | None:
        return self.id if self.kind == "customer" else None


def issue_customer_token(settings: Settings, customer_id: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": "samadhan-api",
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
            issuer="samadhan-api",
        )
    except jwt.PyJWTError:
        # Not one of our session tokens: maybe an access token from the customer identity provider.
        api = settings.api
        idp_claims = decode_oidc(
            authorization.split(" ", 1)[1],
            issuer=api.customer_oidc_issuer,
            audience=api.customer_oidc_audience,
            jwks_uri=api.customer_oidc_jwks_uri,
            public_key=api.customer_oidc_public_key,
        )
        value = idp_claims.get(api.customer_id_claim) if idp_claims else None
        return str(value)[:80] if value else None
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
        if not settings.api.admin_key_enabled:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Staff key login is disabled - use SSO")
        # Constant-time comparison: no timing side channel on the key.
        if not hmac.compare_digest(x_admin_key, settings.api.admin_api_key.get_secret_value()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid staff key")
        return Principal(kind="staff", id=(x_staff_name or "supervisor")[:40], roles=STAFF_ROLES)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    customer_id = verify_customer_token(settings, authorization)
    if customer_id is not None:
        return Principal(kind="customer", id=customer_id)
    staff = await verify_staff_token(settings, authorization.split(" ", 1)[1])
    if staff is not None:
        return staff
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


_jwks_clients: dict[str, jwt.PyJWKClient] = {}


def jwks_client(uri: str) -> jwt.PyJWKClient:
    # Keys cached for an hour (and warmed at startup): verification normally never waits on the IdP;
    # an unknown `kid` (key rotation) triggers a refetch.
    return _jwks_clients.setdefault(uri, jwt.PyJWKClient(uri, cache_keys=True, lifespan=3600))


def decode_oidc(
    token: str, *, issuer: str | None, audience: str | None, jwks_uri: str | None, public_key: str | None
) -> dict[str, Any] | None:
    """Verify an IdP-issued JWT: signature (IdP JWKS or static key), issuer, audience, expiry.
    None when not configured or invalid - unverified claims are never trusted."""
    if not issuer or not (jwks_uri or public_key):
        return None
    try:
        key: Any = public_key or jwks_client(jwks_uri or "").get_signing_key_from_jwt(token).key
        claims: dict[str, Any] = jwt.decode(
            token,
            key,
            algorithms=["RS256", "ES256", "EdDSA", "PS256"],  # asymmetric only: no HS256 confusion
            issuer=issuer,
            audience=audience,
            options={"verify_aud": audience is not None},
        )
    except (jwt.PyJWTError, OSError):
        return None
    return claims


async def verify_staff_token(settings: Settings, token: str) -> Principal | None:
    """Staff SSO: an IdP-issued JWT whose roles claim maps to supervisor / agent / admin."""
    api = settings.api
    claims = await asyncio.to_thread(
        decode_oidc,
        token,
        issuer=api.staff_oidc_issuer,
        audience=api.staff_oidc_audience,
        jwks_uri=api.staff_oidc_jwks_uri,
        public_key=api.staff_oidc_public_key,
    )
    if claims is None:
        return None
    raw = claims.get(api.staff_roles_claim, [])
    roles = frozenset(raw.split() if isinstance(raw, str) else raw) & STAFF_ROLES
    name = str(claims.get("preferred_username") or claims.get("email") or claims["sub"])[:80]
    return Principal(kind="staff", id=name, roles=roles)


def require_staff(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    if principal.kind != "staff":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Staff access required")
    return principal


def require_admin(principal: Annotated[Principal, Depends(require_staff)]) -> Principal:
    if "admin" not in principal.roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "The admin role is required")
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
