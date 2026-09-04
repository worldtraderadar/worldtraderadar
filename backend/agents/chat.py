from __future__ import annotations

from typing import Any

from prompts import GREETING_REPLY

from .session import continue_intake, looks_like_product
from .types import AgentDeps, AgentStep


async def run_chat(
    question: str,
    deps: AgentDeps,
    emit,
) -> tuple[str, list[dict[str, Any]], list[AgentStep]]:
    """Selam / hal hatır: arama yok. Hitap sorulur."""
    steps: list[AgentStep] = []

    current = AgentStep(
        key="greet",
        agent="chat",
        label="Sohbet",
        status="running",
        detail="Karşılama.",
    )
    steps.append(current)
    await emit(current)

    session = deps.session
    if session is not None and (
        session.product
        or session.last_ask in ("product", "choice")
        or looks_like_product(question)
    ):
        advice = continue_intake(session, question)
    else:
        if session is not None:
            session.last_ask = "name"
        advice = GREETING_REPLY

    current.status = "success"
    current.detail = "Cevap hazır."
    await emit(current)
    return advice, [], steps
