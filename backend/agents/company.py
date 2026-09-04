"""Kullanıcının kendi şirket profili. organizations hedef/kaynak firmadır."""

from __future__ import annotations

import logging
from typing import Any

from .session import SessionState

log = logging.getLogger("wtr.company")


def profile_from_session(session: SessionState | None) -> dict[str, Any]:
    if session is None:
        return {}
    data: dict[str, Any] = {}
    if session.product:
        data["products"] = [session.product]
    if session.capacity:
        data["capacity"] = session.capacity
    if session.stock:
        data["moq_or_stock"] = session.stock
    if session.market:
        data["target_markets"] = session.market
    if session.name:
        data["contact_name"] = session.name
    return data


def format_profile(profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    lines: list[str] = []
    labels = {
        "company_name": "Şirket",
        "sector": "Sektör",
        "products": "Ürünler",
        "capacity": "Kapasite",
        "current_markets": "Mevcut pazarlar",
        "target_markets": "Hedef pazarlar",
        "sales_channels": "Satış kanalları",
        "pricing_notes": "Fiyatlandırma",
        "moq": "Minimum sipariş",
        "logistics": "Lojistik",
        "brand": "Marka",
        "target_customer": "Hedef müşteri",
        "commercial_goals": "Ticari hedefler",
        "contact_name": "Hitap",
        "moq_or_stock": "Stok / MOQ",
    }
    for key, label in labels.items():
        value = profile.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value if item)
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


async def load_company_profile(
    supabase: Any,
    *,
    account_slug: str,
    session: SessionState | None = None,
) -> dict[str, Any]:
    profile = profile_from_session(session)
    slug = (account_slug or "demo").strip() or "demo"
    if supabase is None:
        return profile
    try:
        result = (
            await supabase.table("consultant_profiles")
            .select("*")
            .eq("account_slug", slug)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if rows:
            stored = {key: value for key, value in rows[0].items() if value not in (None, "", [], {})}
            stored.update({key: value for key, value in profile.items() if value})
            return stored
    except Exception as exc:
        log.debug("consultant_profiles okunamadı: %s", exc)
    return profile


async def upsert_company_profile(
    supabase: Any,
    *,
    account_slug: str,
    patch: dict[str, Any],
) -> None:
    slug = (account_slug or "demo").strip() or "demo"
    if supabase is None or not patch:
        return
    payload = {"account_slug": slug, **patch}
    try:
        await supabase.table("consultant_profiles").upsert(payload).execute()
    except Exception as exc:
        log.debug("consultant_profiles yazılamadı: %s", exc)
