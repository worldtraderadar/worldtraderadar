from __future__ import annotations

from typing import Any

from prompts import SECTOR_PROMPT, build_user_turn

from .context_pack import build_context_pack
from .response_engine import compose_consultant_reply
from .session import session_notes
from .types import AgentDeps, AgentStep


async def run_sector_chat(
    question: str,
    deps: AgentDeps,
    emit,
) -> tuple[str, list[dict[str, Any]], list[AgentStep]]:
    """Genel sektör sohbeti: arama yok; doğal, kısa değerlendirme."""
    steps: list[AgentStep] = []

    current = AgentStep(
        key="brief",
        agent="sector_chat",
        label="Sektör sohbeti",
        status="running",
        detail="Genel değerlendirme.",
    )
    steps.append(current)
    current.detail = "Şirket bilgileri ve bellek kontrol ediliyor."
    await emit(current)

    pack = await build_context_pack(
        supabase=deps.supabase,
        session=deps.session,
        account_slug=getattr(deps, "account_slug", "demo"),
        question=question,
    )
    facts = session_notes(deps.session, question=question) or "Kayıtlı şirket slotu yok."
    advice = await compose_consultant_reply(
        generate=deps.generate,
        http=deps.http,
        question=question,
        facts=facts,
        pack=pack,
        task="Genel sektör sohbeti. Şirket context varsa kullan. Serbest internet cevabı yok.",
        system_prompt=SECTOR_PROMPT,
        session=deps.session,
    )
    if not (advice or "").strip():
        advice = await deps.generate(
            deps.http,
            build_user_turn(question, pack.as_system_block() or session_notes(deps.session, question=question)),
            system_prompt=SECTOR_PROMPT,
        )
    current.status = "success"
    current.detail = "Cevap hazır."
    await emit(current)
    return advice, [], steps
