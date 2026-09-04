from __future__ import annotations

from typing import Any, Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from .engine import (
    QuotaExceeded,
    account_slug_from_headers,
    consume_quota,
    get_snapshot,
)
from .meter import init_token_meter, take_tokens

METERED_PATHS = {"/search", "/consult", "/consult/stream"}
GetSupabase = Callable[[Request], Awaitable[Any]]


def quota_json_response(exc: QuotaExceeded, status_code: int = 429) -> JSONResponse:
    snapshot = exc.snapshot
    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.to_detail()},
        headers={
            "Retry-After": "86400",
            "X-Plan": snapshot.plan_id,
            "X-Quota-Searches": (
                f"{snapshot.searches_used}/{snapshot.daily_search_limit}"
            ),
            "X-Quota-Tokens": (
                f"{snapshot.tokens_used}/{snapshot.daily_token_limit}"
            ),
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": (
                "X-Plan, X-Quota-Searches, X-Quota-Tokens, Retry-After"
            ),
        },
    )


def apply_quota_headers(response: Response, snapshot: Any) -> None:
    response.headers["X-Plan"] = snapshot.plan_id
    response.headers["X-Quota-Searches"] = (
        f"{snapshot.searches_used}/{snapshot.daily_search_limit}"
    )
    response.headers["X-Quota-Tokens"] = (
        f"{snapshot.tokens_used}/{snapshot.daily_token_limit}"
    )


class QuotaMiddleware(BaseHTTPMiddleware):
    """Günlük arama ve AI token kotalarını metered uçlarda denetler."""

    def __init__(self, app: ASGIApp, get_supabase: GetSupabase) -> None:
        super().__init__(app)
        self.get_supabase = get_supabase

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"
        if request.method != "POST" or path not in METERED_PATHS:
            return await call_next(request)

        supabase = await self.get_supabase(request)
        slug = account_slug_from_headers(request.headers)
        request.state.account_slug = slug
        init_token_meter()

        try:
            snapshot = await consume_quota(supabase, slug, searches=1, tokens=0)
        except QuotaExceeded as exc:
            return quota_json_response(exc)

        request.state.quota = snapshot
        response = await call_next(request)

        content_type = response.headers.get("content-type", "")
        is_stream = "text/event-stream" in content_type
        if not is_stream:
            used = take_tokens()
            if used:
                try:
                    snapshot = await consume_quota(
                        supabase, slug, searches=0, tokens=used
                    )
                except QuotaExceeded as exc:
                    snapshot = exc.snapshot
            else:
                snapshot = await get_snapshot(supabase, slug)
            apply_quota_headers(response, snapshot)

        return response
