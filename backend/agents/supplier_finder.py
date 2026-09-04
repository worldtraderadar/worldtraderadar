from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from prompts import INTAKE_REPLY, is_general_intake

from .context_pack import build_context_pack
from .response_engine import compose_consultant_reply, should_reason
from .retrieve import (
    enrich_matches_for_product,
    gather_match_rows,
    retrieval_query,
    with_domain_notes,
)
from .session import apply_utterance_slots, extract_product_slot, in_progress, party_action_reply
from .tools_registry import matches_as_tool, run_needed_tools
from .types import AgentDeps, AgentStep

_POOL = 24
_THRESHOLD = 0.45


async def run_buyer_finder(
    question: str,
    deps: AgentDeps,
    emit,
) -> tuple[str, list[dict[str, Any]], list[AgentStep]]:
    return await _run_party_finder(question, deps, emit, role="buyer")


async def run_supplier_finder(
    question: str,
    deps: AgentDeps,
    emit,
) -> tuple[str, list[dict[str, Any]], list[AgentStep]]:
    return await _run_party_finder(question, deps, emit, role="supplier")


async def _run_party_finder(
    question: str,
    deps: AgentDeps,
    emit,
    *,
    role: Literal["buyer", "supplier"],
) -> tuple[str, list[dict[str, Any]], list[AgentStep]]:
    agent = "buyer_finder" if role == "buyer" else "supplier_finder"
    label = "Alıcı tarama" if role == "buyer" else "Tedarikçi tarama"
    steps: list[AgentStep] = []

    async def step(key: str, title: str, detail: str = "") -> AgentStep:
        current = AgentStep(
            key=key,
            agent=agent,
            label=title,
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

    if deps.session is not None:
        deps.session.last_ask = None
        apply_utterance_slots(deps.session, question)

    product = deps.session.product if deps.session is not None else None
    if not extract_product_slot(product):
        product = extract_product_slot(question)
        if product and deps.session is not None and not deps.session.product:
            deps.session.product = product
    query = retrieval_query(deps.session, question, role=role)

    embed_step = await step("embed", "BGE-M3 vektörleştirme")
    embedding = await deps.embed(deps.http, query)
    embed_step.status = "success"
    embed_step.detail = f"Sorgu: {query[:80]}"
    await emit(embed_step)

    match_step = await step("retrieve", "Firma ve ticaret kalemi taraması")
    merged = await gather_match_rows(
        match=deps.match,
        supabase=deps.supabase,
        embedding=embedding,
        product=product or query,
        match_count=max(deps.match_count, _POOL),
        match_threshold=max(deps.match_threshold, _THRESHOLD),
        organization_id=deps.organization_id,
    )
    filtered, notes = enrich_matches_for_product(
        merged, product or query, role=role
    )
    matches = [deps.match_model.model_validate(row) for row in filtered]
    match_step.status = "success"
    match_step.detail = f"{len(matches)} ürünle uyumlu kayıt."
    await emit(match_step)

    cluster_step = await step("cluster", "Pazarlara göre kümeleme")
    origins: dict[str, int] = defaultdict(int)
    for item in matches:
        origin = (item.destination_country or item.origin_country or "?").strip() or "?"
        origins[origin] += 1
    cluster_step.status = "success"
    cluster_step.detail = ", ".join(
        f"{code}:{count}" for code, count in sorted(origins.items())
    ) or "küme yok"
    await emit(cluster_step)

    gen_step = await step("advise", "Ticari değerlendirme")
    facts = with_domain_notes(
        party_action_reply(
            role=role, product=product, matches=matches, question=question
        ),
        notes,
        product=product,
    )
    tools = await run_needed_tools(question=question, deps=deps)
    tools.append(matches_as_tool(matches, name="buyer_search" if role == "buyer" else "supplier_search"))
    pack = await build_context_pack(
        supabase=deps.supabase,
        session=deps.session,
        account_slug=getattr(deps, "account_slug", "demo"),
        question=question,
        extra_tools=tools,
    )
    if should_reason(question) and matches is not None:
        no_hits = (
            "Veritabanında eşleşen firma yok. İsim uydurma. "
            "Bulunamadığını açıkça söyle, sonra hedef segment öner."
            if not matches
            else facts
        )
        advice = await compose_consultant_reply(
            generate=deps.generate,
            http=deps.http,
            question=question,
            facts=no_hits,
            pack=pack,
            tools=tools,
            task="Alıcı/tedarikçi listesini ticari öncelikle yorumla. "
            "Hepsini aynı öncelikte görme. Kartta olmayan firma uydurma. "
            "Eşleşme skoru ticari uygunluk değildir. A/B/C öncelik ver.",
            session=deps.session,
            matches=matches,
        )
    else:
        advice = facts
    gen_step.status = "success"
    gen_step.detail = (
        f"{len(matches)} kayıt değerlendirildi."
        if matches
        else "Eşleşme yok; firma uydurulmadı."
    )
    await emit(gen_step)

    context = [item.model_dump(mode="json") for item in matches]
    return advice, context, steps
