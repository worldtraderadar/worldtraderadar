from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from supabase import AsyncClient

DEFAULT_ACCOUNT_SLUG = "demo"
DEMO_ACCOUNT_ID = "00000000-0000-4000-8000-000000000001"

PLAN_CATALOG: list[dict[str, Any]] = [
    {
        "id": "free",
        "name": "Free",
        "daily_search_limit": 8,
        "daily_token_limit": 15_000,
        "monthly_price_usd": 0,
        "features": [
            "Günde 8 arama / danışmanlık",
            "15.000 AI token / gün",
            "Orchestrator + 3 ajan",
            "Şeffaf ajan akışı",
        ],
    },
    {
        "id": "pro",
        "name": "Pro",
        "daily_search_limit": 250,
        "daily_token_limit": 500_000,
        "monthly_price_usd": 49,
        "features": [
            "Günde 250 arama / danışmanlık",
            "500.000 AI token / gün",
            "Öncelikli kota",
            "Tüm ajanlar ve geçmiş raporlar",
            "Paket yükseltme dahil",
        ],
    },
]

PLAN_BY_ID = {plan["id"]: plan for plan in PLAN_CATALOG}


def _catalog(plan_id: str) -> dict[str, Any]:
    return PLAN_BY_ID.get(plan_id, PLAN_BY_ID["free"])


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _missing_relation(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "does not exist",
        "schema cache",
        "could not find the table",
        "pgrst205",
    )
    return any(marker in text for marker in markers)


@dataclass
class QuotaSnapshot:
    account_id: str
    slug: str
    display_name: str
    plan_id: str
    plan_name: str
    daily_search_limit: int
    daily_token_limit: int
    searches_used: int
    tokens_used: int
    usage_date: str
    monthly_price_usd: float
    features: list[str] = field(default_factory=list)
    backend: str = "memory"

    @property
    def searches_remaining(self) -> int:
        return max(self.daily_search_limit - self.searches_used, 0)

    @property
    def tokens_remaining(self) -> int:
        return max(self.daily_token_limit - self.tokens_used, 0)

    @property
    def search_limited(self) -> bool:
        return self.searches_used >= self.daily_search_limit

    @property
    def token_limited(self) -> bool:
        return self.tokens_used >= self.daily_token_limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "slug": self.slug,
            "display_name": self.display_name,
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "usage_date": self.usage_date,
            "backend": self.backend,
            "monthly_price_usd": self.monthly_price_usd,
            "features": self.features,
            "usage": {
                "searches": self.searches_used,
                "tokens": self.tokens_used,
            },
            "limits": {
                "daily_searches": self.daily_search_limit,
                "daily_tokens": self.daily_token_limit,
            },
            "remaining": {
                "searches": self.searches_remaining,
                "tokens": self.tokens_remaining,
            },
        }


class QuotaExceeded(Exception):
    def __init__(self, reason: str, snapshot: QuotaSnapshot) -> None:
        self.reason = reason
        self.snapshot = snapshot
        if reason == "search_limit":
            message = (
                f"Günlük arama kotası doldu "
                f"({snapshot.searches_used}/{snapshot.daily_search_limit}). "
                "Pro pakete yükselterek devam edin."
            )
        else:
            message = (
                f"Günlük AI token kotası doldu "
                f"({snapshot.tokens_used}/{snapshot.daily_token_limit}). "
                "Pro pakete yükselterek devam edin."
            )
        self.message = message
        super().__init__(message)

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": "quota_exceeded",
            "reason": self.reason,
            "message": self.message,
            "upgrade_path": "/paket",
            **self.snapshot.to_dict(),
        }


class MemoryBillingStore:
    def __init__(self) -> None:
        self.accounts: dict[str, dict[str, Any]] = {
            DEFAULT_ACCOUNT_SLUG: {
                "id": DEMO_ACCOUNT_ID,
                "slug": DEFAULT_ACCOUNT_SLUG,
                "display_name": "World Trade Radar Demo",
                "plan_id": "free",
            }
        }
        self.usage: dict[tuple[str, str], dict[str, int]] = {}
        self.events: list[dict[str, Any]] = []

    def account(self, slug: str) -> dict[str, Any]:
        row = self.accounts.get(slug)
        if row:
            return row
        row = {
            "id": str(uuid4()),
            "slug": slug,
            "display_name": slug,
            "plan_id": "free",
        }
        self.accounts[slug] = row
        return row

    def usage_row(self, slug: str) -> dict[str, int]:
        key = (slug, _utc_today())
        if key not in self.usage:
            self.usage[key] = {"search_count": 0, "token_count": 0}
        return self.usage[key]


_memory = MemoryBillingStore()
_checkouts: dict[str, dict[str, Any]] = {}
_db_ready: bool | None = None
_seeded = False


def reset_db_ready_cache() -> None:
    global _db_ready, _seeded
    _db_ready = None
    _seeded = False


def account_slug_from_headers(headers: Any) -> str:
    slug = (
        headers.get("x-account-slug")
        or headers.get("X-Account-Slug")
        or ""
    ).strip()
    if slug:
        return slug[:64]
    return DEFAULT_ACCOUNT_SLUG


def snapshot_from_rows(
    *,
    account: dict[str, Any],
    plan: dict[str, Any],
    usage: dict[str, Any],
    backend: str,
) -> QuotaSnapshot:
    features = plan.get("features") or []
    if isinstance(features, str):
        features = [features]
    return QuotaSnapshot(
        account_id=str(account["id"]),
        slug=str(account["slug"]),
        display_name=str(account.get("display_name") or account["slug"]),
        plan_id=str(plan["id"]),
        plan_name=str(plan.get("name") or plan["id"]),
        daily_search_limit=int(plan["daily_search_limit"]),
        daily_token_limit=int(plan["daily_token_limit"]),
        searches_used=int(usage.get("search_count") or 0),
        tokens_used=int(usage.get("token_count") or 0),
        usage_date=_utc_today(),
        monthly_price_usd=float(plan.get("monthly_price_usd") or 0),
        features=[str(item) for item in features],
        backend=backend,
    )


def _memory_snapshot(slug: str) -> QuotaSnapshot:
    account = _memory.account(slug)
    plan = _catalog(account["plan_id"])
    usage = _memory.usage_row(slug)
    return snapshot_from_rows(
        account=account, plan=plan, usage=usage, backend="memory"
    )


async def billing_db_ready(supabase: AsyncClient | None) -> bool:
    global _db_ready
    if supabase is None:
        return False
    if _db_ready is True:
        return True
    try:
        await supabase.table("plans").select("id").limit(1).execute()
        _db_ready = True
    except Exception:
        _db_ready = False
    return _db_ready


async def seed_billing(supabase: AsyncClient) -> None:
    global _seeded
    if _seeded:
        return
    try:
        await supabase.table("plans").upsert(PLAN_CATALOG, on_conflict="id").execute()
        await (
            supabase.table("accounts")
            .upsert(
                {
                    "id": DEMO_ACCOUNT_ID,
                    "slug": DEFAULT_ACCOUNT_SLUG,
                    "display_name": "World Trade Radar Demo",
                    "email": "demo@worldtraderadar.local",
                    "plan_id": "free",
                },
                on_conflict="slug",
            )
            .execute()
        )
        _seeded = True
    except Exception:
        reset_db_ready_cache()


async def ensure_account(supabase: AsyncClient | None, slug: str) -> dict[str, Any]:
    if not await billing_db_ready(supabase):
        return _memory.account(slug)
    assert supabase is not None
    await seed_billing(supabase)
    existing = (
        await supabase.table("accounts").select("*").eq("slug", slug).limit(1).execute()
    )
    if existing.data:
        return existing.data[0]
    created = await (
        supabase.table("accounts")
        .insert(
            {
                "slug": slug,
                "display_name": slug,
                "plan_id": "free",
            }
        )
        .execute()
    )
    if created.data:
        return created.data[0]
    return _memory.account(slug)


async def _plan_row(supabase: AsyncClient, plan_id: str) -> dict[str, Any]:
    result = (
        await supabase.table("plans").select("*").eq("id", plan_id).limit(1).execute()
    )
    if result.data:
        row = result.data[0]
        catalog = _catalog(plan_id)
        row.setdefault("features", catalog["features"])
        return row
    return _catalog(plan_id)


async def _usage_row(
    supabase: AsyncClient, account_id: str
) -> dict[str, Any]:
    today = _utc_today()
    result = (
        await supabase.table("quota_usage")
        .select("*")
        .eq("account_id", account_id)
        .eq("usage_date", today)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]
    inserted = await (
        supabase.table("quota_usage")
        .insert(
            {
                "account_id": account_id,
                "usage_date": today,
                "search_count": 0,
                "token_count": 0,
            }
        )
        .execute()
    )
    if inserted.data:
        return inserted.data[0]
    return {"search_count": 0, "token_count": 0, "usage_date": today}


async def get_snapshot(
    supabase: AsyncClient | None, slug: str
) -> QuotaSnapshot:
    if not await billing_db_ready(supabase):
        return _memory_snapshot(slug)
    assert supabase is not None
    account = await ensure_account(supabase, slug)
    plan = await _plan_row(supabase, account.get("plan_id") or "free")
    usage = await _usage_row(supabase, str(account["id"]))
    return snapshot_from_rows(
        account=account, plan=plan, usage=usage, backend="supabase"
    )


async def consume_quota(
    supabase: AsyncClient | None,
    slug: str,
    *,
    searches: int = 0,
    tokens: int = 0,
) -> QuotaSnapshot:
    if searches == 0 and tokens == 0:
        return await get_snapshot(supabase, slug)

    if not await billing_db_ready(supabase):
        snapshot = _memory_snapshot(slug)
        if searches and snapshot.search_limited:
            raise QuotaExceeded("search_limit", snapshot)
        if tokens and snapshot.token_limited:
            raise QuotaExceeded("token_limit", snapshot)
        usage = _memory.usage_row(slug)
        usage["search_count"] += searches
        usage["token_count"] += tokens
        return _memory_snapshot(slug)

    assert supabase is not None
    snapshot = await get_snapshot(supabase, slug)
    if searches and snapshot.search_limited:
        raise QuotaExceeded("search_limit", snapshot)
    if tokens and snapshot.token_limited:
        raise QuotaExceeded("token_limit", snapshot)

    try:
        rpc = await supabase.rpc(
            "consume_quota",
            {
                "p_account_id": snapshot.account_id,
                "p_searches": searches,
                "p_tokens": tokens,
            },
        ).execute()
        payload = rpc.data
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if isinstance(payload, dict):
            if payload.get("allowed") is False:
                reason = str(payload.get("reason") or "search_limit")
                latest = await get_snapshot(supabase, slug)
                raise QuotaExceeded(reason, latest)
            return await get_snapshot(supabase, slug)
    except QuotaExceeded:
        raise
    except Exception:
        pass

    usage = await _usage_row(supabase, snapshot.account_id)
    await (
        supabase.table("quota_usage")
        .update(
            {
                "search_count": int(usage.get("search_count") or 0) + searches,
                "token_count": int(usage.get("token_count") or 0) + tokens,
            }
        )
        .eq("account_id", snapshot.account_id)
        .eq("usage_date", _utc_today())
        .execute()
    )
    return await get_snapshot(supabase, slug)


async def list_plans(supabase: AsyncClient | None) -> list[dict[str, Any]]:
    if not await billing_db_ready(supabase):
        return list(PLAN_CATALOG)
    assert supabase is not None
    await seed_billing(supabase)
    result = await supabase.table("plans").select("*").order("monthly_price_usd").execute()
    rows = result.data or []
    if not rows:
        return list(PLAN_CATALOG)
    merged: list[dict[str, Any]] = []
    for row in rows:
        catalog = _catalog(str(row.get("id") or "free"))
        features = row.get("features") or catalog["features"]
        merged.append({**catalog, **row, "features": features})
    return merged


async def log_billing_event(
    supabase: AsyncClient | None,
    *,
    account_id: str | None,
    event_type: str,
    plan_id: str | None = None,
    amount_usd: float | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    event = {
        "id": str(uuid4()),
        "account_id": account_id,
        "event_type": event_type,
        "plan_id": plan_id,
        "amount_usd": amount_usd,
        "payload": payload or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _memory.events.append(event)
    if not await billing_db_ready(supabase):
        return
    assert supabase is not None
    try:
        await supabase.table("billing_events").insert(
            {
                "account_id": account_id,
                "event_type": event_type,
                "plan_id": plan_id,
                "amount_usd": amount_usd,
                "payload": payload or {},
            }
        ).execute()
    except Exception:
        reset_db_ready_cache()


async def set_plan(
    supabase: AsyncClient | None, slug: str, plan_id: str
) -> QuotaSnapshot:
    if plan_id not in PLAN_BY_ID:
        raise ValueError(f"Bilinmeyen plan: {plan_id}")
    if not await billing_db_ready(supabase):
        account = _memory.account(slug)
        account["plan_id"] = plan_id
        return _memory_snapshot(slug)
    assert supabase is not None
    account = await ensure_account(supabase, slug)
    await (
        supabase.table("accounts")
        .update(
            {
                "plan_id": plan_id,
                "plan_started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("id", account["id"])
        .execute()
    )
    try:
        await (
            supabase.table("subscriptions")
            .insert(
                {
                    "account_id": account["id"],
                    "plan_id": plan_id,
                    "status": "active" if plan_id == "pro" else "canceled",
                    "provider": "internal",
                    "current_period_end": None,
                }
            )
            .execute()
        )
    except Exception:
        pass
    return await get_snapshot(supabase, slug)


async def start_checkout(
    supabase: AsyncClient | None, slug: str, plan_id: str
) -> dict[str, Any]:
    if plan_id != "pro":
        raise ValueError("Yalnızca Pro paketine yükseltme desteklenir.")
    snapshot = await get_snapshot(supabase, slug)
    if snapshot.plan_id == "pro":
        raise ValueError("Hesap zaten Pro planında.")
    session = {
        "id": str(uuid4()),
        "account_slug": slug,
        "account_id": snapshot.account_id,
        "plan_id": "pro",
        "plan_name": "Pro",
        "amount_usd": float(_catalog("pro")["monthly_price_usd"]),
        "currency": "USD",
        "status": "pending",
        "provider": "internal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limits": {
            "daily_searches": _catalog("pro")["daily_search_limit"],
            "daily_tokens": _catalog("pro")["daily_token_limit"],
        },
    }
    _checkouts[session["id"]] = session
    await log_billing_event(
        supabase,
        account_id=snapshot.account_id,
        event_type="checkout_started",
        plan_id="pro",
        amount_usd=session["amount_usd"],
        payload={"checkout_id": session["id"]},
    )
    return session


async def confirm_checkout(
    supabase: AsyncClient | None, slug: str, checkout_id: str
) -> tuple[dict[str, Any], QuotaSnapshot]:
    session = _checkouts.get(checkout_id)
    if session is None or session.get("account_slug") != slug:
        raise KeyError("Ödeme oturumu bulunamadı veya süresi doldu.")
    if session["status"] != "paid":
        snapshot = await set_plan(supabase, slug, "pro")
        session["status"] = "paid"
        session["paid_at"] = datetime.now(timezone.utc).isoformat()
        await log_billing_event(
            supabase,
            account_id=snapshot.account_id,
            event_type="checkout_paid",
            plan_id="pro",
            amount_usd=session["amount_usd"],
            payload={"checkout_id": checkout_id},
        )
        await log_billing_event(
            supabase,
            account_id=snapshot.account_id,
            event_type="plan_changed",
            plan_id="pro",
            payload={"from": "free", "to": "pro"},
        )
    else:
        snapshot = await get_snapshot(supabase, slug)
    return session, snapshot
