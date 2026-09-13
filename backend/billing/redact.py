from __future__ import annotations

from typing import Any, Iterable

from supabase import AsyncClient
from starlette.requests import Request

from .engine import get_snapshot

VOLUME_FIELDS = ("quantity", "value_usd")
CONTACT_FIELDS = ("contact_email", "website", "tax_id")


async def plan_id_for_request(request: Request, supabase: AsyncClient | None) -> str:
    principal = getattr(request.state, "auth", None)
    if principal is not None and getattr(principal, "plan_id", None):
        return str(principal.plan_id)
    # Fallback only when auth middleware has set trusted slug (never raw header).
    slug = getattr(request.state, "account_slug", None)
    if not slug:
        return "free"
    snapshot = await get_snapshot(supabase, slug)
    return snapshot.plan_id


def _has_any(row: dict[str, Any], fields: Iterable[str]) -> bool:
    for field in fields:
        value = row.get(field)
        if value is not None and value != "":
            return True
    organization = row.get("organization")
    if isinstance(organization, dict):
        for field in fields:
            value = organization.get(field)
            if value is not None and value != "":
                return True
    return False


def _clear(row: dict[str, Any], fields: Iterable[str]) -> None:
    for field in fields:
        if field in row:
            row[field] = None
    organization = row.get("organization")
    if isinstance(organization, dict):
        for field in fields:
            if field in organization:
                organization[field] = None


def redact_record(row: dict[str, Any], plan_id: str) -> dict[str, Any]:
    """Free planda iletişim ve hacim alanlarını siler; kilit bayraklarını ekler."""
    payload = dict(row)
    organization = payload.get("organization")
    if isinstance(organization, dict):
        payload["organization"] = dict(organization)
        payload.setdefault("organization_name", organization.get("name"))
        payload.setdefault("contact_email", organization.get("contact_email"))
        payload.setdefault("website", organization.get("website"))

    has_volume = _has_any(payload, VOLUME_FIELDS)
    has_contact = _has_any(payload, CONTACT_FIELDS)
    locked = plan_id != "pro"
    if locked:
        _clear(payload, VOLUME_FIELDS)
        _clear(payload, CONTACT_FIELDS)

    payload["plan_id"] = plan_id
    payload["has_volume"] = has_volume
    payload["has_contact"] = has_contact
    payload["locked"] = {
        "contact": locked,
        "volume": locked,
    }
    return payload


def redact_records(
    rows: list[dict[str, Any]] | None, plan_id: str
) -> list[dict[str, Any]]:
    return [redact_record(row, plan_id) for row in (rows or [])]
