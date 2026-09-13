from __future__ import annotations

from typing import Any

from prompts import INTAKE_REPLY, MATCHING_PROMPT, build_user_turn, is_general_intake

from .context_pack import build_context_pack
from .response_engine import compose_consultant_reply
from .retrieve import (
    active_retrieval_product,
    enrich_matches_for_product,
    gather_match_rows,
    retrieval_query,
)
from .session import in_progress, session_notes
from .tools_registry import matches_as_tool
from .types import AgentDeps, AgentStep


async def run_product_matching(
    question: str,
    deps: AgentDeps,
    emit,
) -> tuple[str, list[dict[str, Any]], list[AgentStep]]:
    steps: list[AgentStep] = []

    async def step(key: str, label: str, detail: str = "") -> AgentStep:
        current = AgentStep(
            key=key,
            agent="product_matching",
            label=label,
            status="running",
            detail=detail,
        )
        steps.append(current)
        await emit(current)
        return current

    if is_general_intake(question) and not in_progress(deps.session):
        gen_step = await step("advise", "Keşif")
        gen_step.status = "success"
        gen_step.detail = "Cevap hazır. Veri araması yok."
        await emit(gen_step)
        return INTAKE_REPLY, [], steps

    embed_step = await step("embed", "BGE-M3 vektörleştirme")
    query = retrieval_query(deps.session, question, role="match")
    embedding = await deps.embed(deps.http, query)
    embed_step.status = "success"
    embed_step.detail = "Ürün eşleştirme vektörü hazır."
    await emit(embed_step)

    match_step = await step("retrieve", "Semantik ürün eşleştirme")
    product = active_retrieval_product(deps.session, question)
    rows = await gather_match_rows(
        match=deps.match,
        supabase=deps.supabase,
        embedding=embedding,
        product=product or question,
        match_count=max(deps.match_count, 16),
        match_threshold=max(deps.match_threshold, 0.42),
        organization_id=deps.organization_id,
    )
    filtered, _notes = enrich_matches_for_product(rows, product or question, role="match")
    matches = [deps.match_model.model_validate(row) for row in filtered]
    match_step.status = "success"
    match_step.detail = f"{len(matches)} aday ürün sıralandı."
    await emit(match_step)

    gen_step = await step("advise", "Ticari uygunluk")
    facts = "\n".join(
        p
        for p in (
            session_notes(deps.session, question=question),
            deps.format_context(matches),
        )
        if p
    ) or "Eşleşen ürün kaydı yok. Uydurma HS veya ürün yok."
    pack = await build_context_pack(
        supabase=deps.supabase,
        session=deps.session,
        account_slug=getattr(deps, "account_slug", "demo"),
        question=question,
        extra_tools=[matches_as_tool(matches, name="product_matching")],
    )
    advice = await compose_consultant_reply(
        generate=deps.generate,
        http=deps.http,
        question=question,
        facts=facts,
        pack=pack,
        task=(
            "Ürün + HS + pazar + müşteri tipini birlikte değerlendir. "
            "Ticari uygunluk söyle. Kartta yoksa uydurma."
        ),
        system_prompt=MATCHING_PROMPT,
        session=deps.session,
        matches=matches,
    )
    if not (advice or "").strip():
        prompt = build_user_turn(question, facts)
        advice = await deps.generate(deps.http, prompt, system_prompt=MATCHING_PROMPT)
    gen_step.status = "success"
    gen_step.detail = "Cevap hazır."
    await emit(gen_step)

    context = [item.model_dump(mode="json") for item in matches]
    return advice, context, steps
