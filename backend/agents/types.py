from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Literal
from uuid import UUID

Intent = Literal[
    "chat",
    "intake",
    "sector_chat",
    "trade_advisor",
    "buyer_finder",
    "supplier_finder",
    "product_matching",
]

AGENT_LABELS: dict[str, str] = {
    "orchestrator": "Orchestrator",
    "chat": "Sohbet",
    "intake": "Keşif",
    "sector_chat": "Sektör sohbeti",
    "trade_advisor": "Trade Advisor",
    "buyer_finder": "Buyer Finder",
    "supplier_finder": "Supplier Finder",
    "product_matching": "Product Matching",
}


@dataclass
class AgentStep:
    key: str
    agent: str
    label: str
    status: Literal["pending", "running", "success", "error"] = "running"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentOutcome:
    intent: Intent
    selected_agent: str
    advice: str
    context: list[dict[str, Any]] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)


@dataclass
class AgentDeps:
    http: Any
    supabase: Any
    embed: Callable[[Any, str], Awaitable[list[float]]]
    match: Callable[..., Awaitable[list[dict[str, Any]]]]
    generate: Callable[..., Awaitable[str]]
    format_context: Callable[[list[Any]], str]
    match_model: Any
    organization_id: UUID | None
    match_count: int
    match_threshold: float
    session: Any = None
    account_slug: str = "demo"
    account_id: str = ""
    goal: str = ""
    file_names: list[str] = field(default_factory=list)
