"""Supabase Auth JWT verification (signature, issuer, audience, exp)."""

from __future__ import annotations

import os
from typing import Any

import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError

from .principal import AuthPrincipal  # noqa: F401 — typing docs


class AuthError(Exception):
    """JWT or membership failure."""


_jwks_client: PyJWKClient | None = None


def _supabase_url() -> str:
    return (
        os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        or os.getenv("SUPABASE_URL")
        or ""
    ).rstrip("/")


def jwt_secret() -> str:
    return (
        os.getenv("SUPABASE_JWT_SECRET")
        or os.getenv("WTR_JWT_SECRET")
        or ""
    ).strip()


def jwt_issuer() -> str:
    explicit = (os.getenv("SUPABASE_JWT_ISSUER") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    base = _supabase_url()
    if base:
        return f"{base}/auth/v1"
    return "http://localhost/auth/v1"


def jwt_audience() -> str:
    return (os.getenv("SUPABASE_JWT_AUD") or "authenticated").strip()


def _jwks_url() -> str:
    base = _supabase_url()
    if not base:
        return ""
    return f"{base}/auth/v1/.well-known/jwks.json"


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    url = _jwks_url()
    if not url:
        raise AuthError("Supabase URL not configured for JWKS")
    if _jwks_client is None:
        _jwks_client = PyJWKClient(url, cache_keys=True)
    return _jwks_client


def _require_sub(claims: dict[str, Any]) -> dict[str, Any]:
    sub = str(claims.get("sub") or "").strip()
    if not sub:
        raise AuthError("token missing sub")
    return claims


def _decode_options() -> dict[str, Any]:
    return {
        "require": ["exp", "sub", "aud"],
        "verify_signature": True,
        "verify_exp": True,
        "verify_aud": True,
        "verify_iss": True,
    }


def _verify_hs256(raw: str) -> dict[str, Any]:
    """Legacy / test path: HS256 shared secret."""
    secret = jwt_secret()
    if not secret:
        raise AuthError("JWT secret not configured")
    try:
        claims = jwt.decode(
            raw,
            secret,
            algorithms=["HS256"],
            audience=jwt_audience(),
            issuer=jwt_issuer(),
            options=_decode_options(),
        )
    except InvalidTokenError as exc:
        raise AuthError(f"invalid token: {exc}") from exc
    return _require_sub(claims)


def _verify_asymmetric(raw: str, alg: str) -> dict[str, Any]:
    """Production Supabase signing keys (ES256/RS256) via JWKS."""
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(raw)
        claims = jwt.decode(
            raw,
            signing_key.key,
            algorithms=[alg],
            audience=jwt_audience(),
            issuer=jwt_issuer(),
            options=_decode_options(),
        )
    except (InvalidTokenError, PyJWKClientError, AuthError) as exc:
        raise AuthError(f"invalid token: {exc}") from exc
    except Exception as exc:  # network / jwks
        raise AuthError(f"invalid token: {exc}") from exc
    return _require_sub(claims)


def verify_access_token(token: str) -> dict[str, Any]:
    """
    Verify Bearer access token.

    Supports:
    - HS256 with SUPABASE_JWT_SECRET / WTR_JWT_SECRET (tests / legacy)
    - ES256/RS256 via Supabase JWKS (current hosted Auth signing keys)

    Always validates signature, exp, audience, issuer, and sub (fail-closed).
    """
    raw = (token or "").strip()
    if not raw:
        raise AuthError("missing token")
    try:
        header = jwt.get_unverified_header(raw)
    except Exception as exc:
        raise AuthError(f"invalid token header: {exc}") from exc

    alg = str(header.get("alg") or "").strip()
    if alg in ("ES256", "RS256"):
        return _verify_asymmetric(raw, alg)
    if alg == "HS256":
        return _verify_hs256(raw)
    raise AuthError(f"unsupported token alg: {alg or 'missing'}")


def mint_test_token(
    user_id: str,
    *,
    secret: str | None = None,
    issuer: str | None = None,
    audience: str | None = None,
    ttl_seconds: int = 3600,
    extra: dict[str, Any] | None = None,
) -> str:
    """HS256 test JWT for security suite (not for production)."""
    import time

    key = (secret or jwt_secret() or "wtr-test-jwt-secret-key-32b!!").strip()
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user_id,
        "aud": audience or jwt_audience(),
        "iss": issuer or jwt_issuer(),
        "iat": now,
        "exp": now + int(ttl_seconds),
        "role": "authenticated",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, key, algorithm="HS256")
