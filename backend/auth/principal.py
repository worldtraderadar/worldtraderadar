"""Trusted request identity — never from client account slug headers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["owner", "member", "admin"]


@dataclass(frozen=True)
class AuthPrincipal:
    """Resolved after JWT verify + account_members lookup."""

    user_id: str
    account_id: str
    account_slug: str
    role: Role
    plan_id: str

    @property
    def is_admin(self) -> bool:
        return self.role in ("admin", "owner")
