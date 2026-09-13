"""account_members resolution — JWT sub → account (never X-Account-Slug)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from .jwt_verify import AuthError
from .principal import AuthPrincipal, Role

RoleName = Role


@dataclass
class MembershipRow:
    user_id: str
    account_id: str
    account_slug: str
    role: RoleName
    plan_id: str


# Process-local seed for tests / offline; production prefers Supabase.
_MEMORY: dict[str, list[MembershipRow]] = {}


def reset_membership_memory() -> None:
    _MEMORY.clear()


def seed_membership(
    *,
    user_id: str,
    account_id: str | None = None,
    account_slug: str,
    role: RoleName = "owner",
    plan_id: str = "free",
) -> MembershipRow:
    aid = (account_id or str(uuid4())).strip()
    row = MembershipRow(
        user_id=user_id.strip(),
        account_id=aid,
        account_slug=account_slug.strip(),
        role=role,
        plan_id=(plan_id or "free").strip() or "free",
    )
    bucket = _MEMORY.setdefault(row.user_id, [])
    for existing in bucket:
        if existing.account_id == row.account_id:
            existing.role = row.role
            existing.account_slug = row.account_slug
            existing.plan_id = row.plan_id
            return existing
    bucket.append(row)
    return row


def memory_memberships_for_user(user_id: str) -> list[MembershipRow]:
    return list(_MEMORY.get(user_id.strip(), []))


async def _from_supabase(supabase: Any, user_id: str) -> MembershipRow | None:
    if supabase is None:
        return None
    try:
        result = await (
            supabase.table("account_members")
            .select("user_id, account_id, role, accounts(id, slug, plan_id)")
            .eq("user_id", user_id)
            .limit(8)
            .execute()
        )
    except Exception:
        return None
    rows = result.data or []
    if not rows:
        return None
    # Prefer owner, then admin, then first member.
    def rank(role: str) -> int:
        return {"owner": 0, "admin": 1, "member": 2}.get(role, 9)

    rows = sorted(rows, key=lambda r: rank(str(r.get("role") or "member")))
    row = rows[0]
    accounts = row.get("accounts")
    if isinstance(accounts, list):
        accounts = accounts[0] if accounts else None
    if not isinstance(accounts, dict):
        return None
    role_raw = str(row.get("role") or "member")
    role: RoleName = (
        role_raw if role_raw in ("owner", "member", "admin") else "member"
    )
    return MembershipRow(
        user_id=user_id,
        account_id=str(row.get("account_id") or accounts.get("id")),
        account_slug=str(accounts.get("slug") or ""),
        role=role,
        plan_id=str(accounts.get("plan_id") or "free"),
    )


async def resolve_principal(
    user_id: str,
    supabase: Any = None,
) -> AuthPrincipal:
    uid = (user_id or "").strip()
    if not uid:
        raise AuthError("missing user_id")

    mem = memory_memberships_for_user(uid)
    if mem:
        # Prefer owner/admin ordering like DB path.
        mem = sorted(
            mem,
            key=lambda r: {"owner": 0, "admin": 1, "member": 2}.get(r.role, 9),
        )
        row = mem[0]
        return AuthPrincipal(
            user_id=row.user_id,
            account_id=row.account_id,
            account_slug=row.account_slug,
            role=row.role,
            plan_id=row.plan_id,
        )

    db_row = await _from_supabase(supabase, uid)
    if db_row is None or not db_row.account_slug:
        raise AuthError("no account membership")
    return AuthPrincipal(
        user_id=db_row.user_id,
        account_id=db_row.account_id,
        account_slug=db_row.account_slug,
        role=db_row.role,
        plan_id=db_row.plan_id,
    )
