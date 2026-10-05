"""Service tokens between the Samadhan API and the MCP servers.

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

Tokens are signed with **EdDSA (Ed25519)**. Only the API holds the private key; MCP servers hold
the public key (or fetch it from the API's ``/.well-known/jwks.json``), so a compromised MCP server
cannot mint tokens or approvals. Every token carries a ``kid`` so keys can be rotated via JWKS.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from fastmcp.server.auth.providers.jwt import JWTVerifier

from samadhan.config import MCPSettings

# RFC 9864 deprecates the polymorphic "EdDSA" name for the fully-specified "Ed25519"; PyJWT 2.15 does not
# implement "Ed25519" yet (FastMCP accepts both), so we keep "EdDSA" until it does.
ALGORITHM = "EdDSA"
_PRIVATE_FILE = "service_ed25519.pem"
APPROVAL_AUDIENCE = "samadhan-approvals"

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


# ---- keys ----------------------------------------------------------------------------


def _pem(value: str) -> bytes:
    # Env vars often carry PEM on one line with literal "\n" escapes.
    return value.replace("\\n", "\n").strip().encode()


def generate_key_pair() -> tuple[str, str]:
    """A fresh Ed25519 key pair as (private PEM, public PEM)."""
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return private.decode(), public.decode()


def _dev_private_key(key_dir: Path) -> bytes:
    """Load (or atomically create) the dev key pair shared by local processes - never used in prod,
    where the settings validator requires explicitly configured keys."""
    path = key_dir / _PRIVATE_FILE
    if not path.exists():
        key_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=key_dir)
        with os.fdopen(fd, "wb") as fh:
            fh.write(generate_key_pair()[0].encode())
        try:
            os.link(tmp, path)  # atomic and exclusive: concurrently starting servers agree on one key
        except FileExistsError:
            pass
        finally:
            os.unlink(tmp)
    return path.read_bytes()


def _no_keys_configured(settings: MCPSettings) -> bool:
    return not (settings.jwt_private_key or settings.jwt_public_key or settings.jwt_jwks_uri)


def signing_key(settings: MCPSettings) -> Ed25519PrivateKey:
    if settings.jwt_private_key is not None:
        pem = _pem(settings.jwt_private_key.get_secret_value())
    elif _no_keys_configured(settings):
        pem = _dev_private_key(settings.jwt_key_dir)
    else:
        raise RuntimeError("SAMADHAN_MCP__JWT_PRIVATE_KEY is required to mint service tokens (API only)")
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("the service-token private key must be an Ed25519 key")
    return key


def verification_key(settings: MCPSettings) -> Ed25519PublicKey:
    if settings.jwt_public_key:
        key = serialization.load_pem_public_key(_pem(settings.jwt_public_key))
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("the service-token public key must be an Ed25519 key")
        return key
    return signing_key(settings).public_key()


def public_pem(settings: MCPSettings) -> str:
    return (
        verification_key(settings)
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


def key_id(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def jwks(settings: MCPSettings) -> dict[str, Any]:
    """The public key as a JWK Set (served by the API at ``/.well-known/jwks.json``)."""
    key = verification_key(settings)
    jwk = jwt.algorithms.OKPAlgorithm.to_jwk(key, as_dict=True)
    return {"keys": [{**jwk, "kid": key_id(key), "use": "sig", "alg": ALGORITHM}]}


def _sign(settings: MCPSettings, claims: dict[str, Any]) -> str:
    key = signing_key(settings)
    return jwt.encode(claims, key, algorithm=ALGORITHM, headers={"kid": key_id(key.public_key())})


def _verify(settings: MCPSettings, token: str, *, audience: str) -> dict[str, Any]:
    key: Any = (
        jwt.PyJWKClient(settings.jwt_jwks_uri).get_signing_key_from_jwt(token).key
        if settings.jwt_jwks_uri and not settings.jwt_public_key
        else verification_key(settings)
    )
    return jwt.decode(token, key, algorithms=[ALGORITHM], audience=audience, issuer=settings.jwt_issuer)


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
        "client_id": "samadhan-agent",
    }
    if extra_claims:
        claims.update(extra_claims)
    return _sign(settings, claims)


def build_verifier(settings: MCPSettings) -> JWTVerifier:
    """MCP-server side: verify with the public key, or the API's JWKS (enables key rotation)."""
    if settings.jwt_jwks_uri and not settings.jwt_public_key:
        return JWTVerifier(
            jwks_uri=settings.jwt_jwks_uri,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            algorithm=ALGORITHM,
        )
    return JWTVerifier(
        public_key=public_pem(settings),
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        algorithm=ALGORITHM,
    )


# ---- signed human approvals -------------------------------------------------------


def mint_approval_code(
    settings: MCPSettings, *, customer_id: str, order_id: str, max_amount: float, approver: str, ttl_s: int = 900
) -> str:
    now = int(time.time())
    return _sign(
        settings,
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
    )


class ApprovalError(ValueError):
    pass


def verify_approval_code(
    settings: MCPSettings, code: str, *, customer_id: str, order_id: str, amount: float
) -> dict[str, Any]:
    try:
        claims = _verify(settings, code, audience=APPROVAL_AUDIENCE)
    except jwt.PyJWTError as exc:
        raise ApprovalError(f"Invalid or expired approval code ({type(exc).__name__}).") from exc
    if claims.get("sub") != customer_id or claims.get("order_id") != order_id:
        raise ApprovalError("Approval code was issued for a different customer or order.")
    if amount > float(claims.get("max_amount", 0)) + 0.005:
        raise ApprovalError(f"Approved amount ${claims.get('max_amount')} is lower than the requested ${amount:.2f}.")
    return claims
