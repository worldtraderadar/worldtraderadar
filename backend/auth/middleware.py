"""Auth middleware — customer HTTP paths require verified JWT + membership."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from .deps import authenticate_request
from .jwt_verify import AuthError

GetSupabase = Callable[[Request], Awaitable[Any]]

PUBLIC_EXACT = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def _normalized_path(path: str) -> str:
    return path.rstrip("/") or "/"


def is_public_path(path: str, method: str) -> bool:
    if method.upper() == "OPTIONS":
        return True
    return _normalized_path(path) in PUBLIC_EXACT


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, get_supabase: GetSupabase) -> None:
        super().__init__(app)
        self.get_supabase = get_supabase

    async def dispatch(self, request: Request, call_next) -> Response:
        if is_public_path(request.url.path, request.method):
            return await call_next(request)

        try:
            supabase = None
            try:
                supabase = await self.get_supabase(request)
            except Exception:
                supabase = None
            await authenticate_request(request, supabase)
        except AuthError as exc:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required.", "reason": str(exc)},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        except Exception as exc:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required.", "reason": str(exc)},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        return await call_next(request)
