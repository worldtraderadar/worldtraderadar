"""FastAPI helpers for authenticated principal."""

from __future__ import annotations

from fastapi import HTTPException, Request

from .jwt_verify import AuthError, verify_access_token
from .membership import resolve_principal
from .principal import AuthPrincipal


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization") or request.headers.get(
        "Authorization"
    )
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def get_principal(request: Request) -> AuthPrincipal | None:
    return getattr(request.state, "auth", None)


def require_principal(request: Request) -> AuthPrincipal:
    principal = get_principal(request)
    if principal is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return principal


async def authenticate_request(
    request: Request, supabase: object | None
) -> AuthPrincipal:
    token = bearer_token(request)
    if not token:
        raise AuthError("missing bearer token")
    claims = verify_access_token(token)
    user_id = str(claims["sub"])
    principal = await resolve_principal(user_id, supabase)
    # Prefer live account.plan_id over stale membership cache.
    try:
        from billing.engine import get_snapshot

        snap = await get_snapshot(supabase, principal.account_slug)
        if snap and snap.plan_id and snap.plan_id != principal.plan_id:
            principal = AuthPrincipal(
                user_id=principal.user_id,
                account_id=principal.account_id,
                account_slug=principal.account_slug,
                role=principal.role,
                plan_id=str(snap.plan_id),
            )
    except Exception:
        pass
    request.state.auth = principal
    request.state.account_slug = principal.account_slug
    request.state.account_id = principal.account_id
    request.state.user_id = principal.user_id
    return principal
