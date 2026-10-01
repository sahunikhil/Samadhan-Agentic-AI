"""Service tokens between the CaseFlow API and the MCP servers.

Security model (defense in depth)
---------------------------------
1. The **customer** authenticates to the API (``/v1/auth/...``).
2. For every agent run the API mints a **short-lived, down-scoped service token**
   whose ``sub`` is that customer (a token-exchange pattern, cf. RFC 8693).
3. MCP tools never accept ``customer_id`` as an argument. They read it from the
   verified token via ``TokenClaim("sub")``. A prompt-injected model therefore
   *cannot* read or modify another customer's data - the parameter doesn't exist.
4. Money-moving tools additionally require a **signed human approval** above the
   auto-approve limit. The approval code is minted by the API only after a human
   approves, and verified by the MCP server - so the rule holds even if the agent
   layer is bypassed entirely.

HS256 with a shared secret keeps the demo dependency-free. In production swap to
RS256/EdDSA + JWKS (``JWTVerifier(jwks_uri=...)``) so MCP servers hold no signing key.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import jwt
from fastmcp.server.auth.providers.jwt import JWTVerifier

from caseflow.config import MCPSettings

ALGORITHM = "HS256"
APPROVAL_AUDIENCE = "caseflow-approvals"

# Scopes a customer-facing agent run receives. An internal "agent console" or batch
# job would receive a different set - least privilege per caller type.
CUSTOMER_SCOPES: tuple[str, ...] = (
    "profile:read",
    "orders:read",
    "orders:write",
    "returns:write",
    "refunds:write",
    "tickets:read",
    "tickets:write",
)


def mint_service_token(
    settings: MCPSettings,
    *,
    subject: str,
    scopes: tuple[str, ...] | list[str] = CUSTOMER_SCOPES,
    ttl_s: int | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "sub": subject,
        "iat": now,
        "nbf": now - 5,
        "exp": now + (ttl_s or settings.token_ttl_s),
        "jti": uuid.uuid4().hex,
        "scope": " ".join(scopes),
        "client_id": "caseflow-agent",
    }
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=ALGORITHM)


def build_verifier(settings: MCPSettings) -> JWTVerifier:
    return JWTVerifier(
        public_key=settings.jwt_secret.get_secret_value(),
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        algorithm=ALGORITHM,
    )


# ---- signed human approvals -------------------------------------------------------


def mint_approval_code(
    settings: MCPSettings, *, customer_id: str, order_id: str, max_amount: float, approver: str, ttl_s: int = 900
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": settings.jwt_issuer,
            "aud": APPROVAL_AUDIENCE,
            "sub": customer_id,
            "order_id": order_id,
            "max_amount": round(float(max_amount), 2),
            "approver": approver,
            "iat": now,
            "exp": now + ttl_s,
            "jti": uuid.uuid4().hex,
        },
        settings.jwt_secret.get_secret_value(),
        algorithm=ALGORITHM,
    )


class ApprovalError(ValueError):
    pass


def verify_approval_code(
    settings: MCPSettings, code: str, *, customer_id: str, order_id: str, amount: float
) -> dict[str, Any]:
    try:
        claims: dict[str, Any] = jwt.decode(
            code,
            settings.jwt_secret.get_secret_value(),
            algorithms=[ALGORITHM],
            audience=APPROVAL_AUDIENCE,
            issuer=settings.jwt_issuer,
        )
    except jwt.PyJWTError as exc:
        raise ApprovalError(f"Invalid or expired approval code ({type(exc).__name__}).") from exc
    if claims.get("sub") != customer_id or claims.get("order_id") != order_id:
        raise ApprovalError("Approval code was issued for a different customer or order.")
    if amount > float(claims.get("max_amount", 0)) + 0.005:
        raise ApprovalError(f"Approved amount ${claims.get('max_amount')} is lower than the requested ${amount:.2f}.")
    return claims
