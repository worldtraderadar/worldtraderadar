from __future__ import annotations

from typing import Any

from prompts import INTAKE_REPLY

from .session import continue_intake
from .trade_advisor import run_trade_advisor
from .types import AgentDeps, AgentStep


async def run_intake(
    question: str,
    deps: AgentDeps,
    emit,
) -> tuple[str, list[dict[str, Any]], list[AgentStep]]:
    """Keşif bitince eşleşen firmalar ve taslak teklifi için advisor'a geçer."""
    steps: list[AgentStep] = []

    current = AgentStep(
        key="brief",
        agent="intake",
        label="Keşif",
        status="running",
        detail="Yönlendirme. Ardışık keşif sorusu yok.",
    )
    steps.append(current)
    await emit(current)

    session = deps.session
    if session is None:
        current.status = "success"
        current.detail = "Cevap hazır."
        await emit(current)
        return INTAKE_REPLY, [], steps

    advice = continue_intake(session, question)
    if not session.product:
        current.status = "success"
        current.detail = "Cevap hazır."
        await emit(current)
        return advice, [], steps

    current.status = "success"
    current.detail = "Ürün alındı. Eşleşen firmalar açılıyor."
    await emit(current)
    return await run_trade_advisor(question, deps, emit)
