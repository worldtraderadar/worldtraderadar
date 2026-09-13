"""Future SaaS plan → permissions foundation (not enforcing image limits yet)."""

from __future__ import annotations

from dataclasses import dataclass

from billing.engine import PLAN_BY_ID


@dataclass(frozen=True)
class PlanPolicy:
    """Central policy shape for quotas + future feature gates."""

    plan_id: str
    consult_limit: int
    token_limit: int
    can_view_company_contacts: bool
    can_edit_company_profile: bool
    can_upload_images: bool
    max_images: int


def policy_for_plan(plan_id: str) -> PlanPolicy:
    """Map existing free/pro catalog into extensible policy object."""
    pid = (plan_id or "free").strip() or "free"
    catalog = PLAN_BY_ID.get(pid) or PLAN_BY_ID["free"]
    is_pro = pid == "pro"
    return PlanPolicy(
        plan_id=str(catalog["id"]),
        consult_limit=int(catalog["daily_search_limit"]),
        token_limit=int(catalog["daily_token_limit"]),
        can_view_company_contacts=is_pro,
        can_edit_company_profile=True,
        # Image upload not shipped yet — reserved for SaaS Feature Phase.
        can_upload_images=False,
        max_images=0,
    )
