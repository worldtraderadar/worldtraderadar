from __future__ import annotations

from typing import Any

from prompts import (
    GREETING_REPLY,
    INTAKE_REPLY,
    is_company_data_ask,
    is_current_information,
    is_decision_question,
    is_diagnostic_request,
    is_document_ask,
    is_draft_request,
    is_general_intake,
    is_language_barrier,
    is_memory_recall,
    is_method_question,
    is_sell_request,
    is_small_talk,
    is_stat_challenge,
    is_strategy_request,
    wants_web_research,
)

from .commercial import apply_commercial_guidance, build_commercial_situation
from .contracts import company_miss_facts, current_data_facts, memory_miss_facts
from .context_pack import build_context_pack
from .documents import diagnostic_outline
from .response_engine import compose_consultant_reply, fallback_web_unavailable, should_reason
from .retrieve import (
    active_retrieval_product,
    enrich_matches_for_product,
    gather_match_rows,
    retrieval_query,
    with_domain_notes,
)
from .session import advisor_answer, advisor_focus, apply_utterance_slots, in_progress, session_notes
from .tools_registry import matches_as_tool, run_needed_tools
from .types import AgentDeps, AgentStep


async def run_trade_advisor(
    question: str,
    deps: AgentDeps,
    emit,
) -> tuple[str, list[dict[str, Any]], list[AgentStep]]:
    steps: list[AgentStep] = []

    async def step(key: str, label: str, detail: str = "") -> AgentStep:
        current = AgentStep(
            key=key,
            agent="trade_advisor",
            label=label,
            status="running",
            detail=detail,
        )
        steps.append(current)
        await emit(current)
        return current

    strategy = is_strategy_request(question)
    method = (
        is_method_question(question)
        or is_language_barrier(question)
        or is_draft_request(question)
        or is_sell_request(question)
    )
    if not in_progress(deps.session) and not strategy and not method:
        if is_small_talk(question) and not is_memory_recall(question):
            gen_step = await step("advise", "Sohbet cevabı")
            gen_step.status = "success"
            gen_step.detail = "Cevap hazır."
            await emit(gen_step)
            return GREETING_REPLY, [], steps

        if is_general_intake(question) and not is_diagnostic_request(question):
            gen_step = await step("advise", "Keşif")
            gen_step.status = "success"
            gen_step.detail = "Cevap hazır. Veri araması yok."
            await emit(gen_step)
            return INTAKE_REPLY, [], steps

    if deps.session is not None:
        deps.session.last_ask = None
        apply_utterance_slots(deps.session, question)

    product = (
        active_retrieval_product(deps.session, question)
        if deps.session is not None
        else None
    )
    matches: list[Any] = []
    notes: list[str] = []
    skip_rag = is_memory_recall(question) or is_current_information(question) or is_stat_challenge(question) or is_company_data_ask(question) or (
        is_document_ask(question) and not product
    ) or (is_diagnostic_request(question) and not product) or (
        is_decision_question(question) and not product
    )

    if skip_rag:
        embed_skip = await step("retrieve", "Bağlam kontrolü")
        embed_skip.status = "success"
        embed_skip.detail = "Şirket bilgileri ve bellek kontrol ediliyor."
        await emit(embed_skip)
    else:
        embed_step = await step("embed", "BGE-M3 vektörleştirme")
        query = retrieval_query(deps.session, question, role="advisor")
        embedding = await deps.embed(deps.http, query)
        embed_step.status = "success"
        embed_step.detail = "1024 boyutlu sorgu vektörü hazır."
        await emit(embed_step)

        match_step = await step("retrieve", "Ticaret kalemi araması")
        rows = await gather_match_rows(
            match=deps.match,
            supabase=deps.supabase,
            embedding=embedding,
            product=product or question,
            match_count=deps.match_count,
            match_threshold=deps.match_threshold,
            organization_id=deps.organization_id,
        )
        filtered, notes = enrich_matches_for_product(rows, product or question, role="advisor")
        matches = [deps.match_model.model_validate(row) for row in filtered]
        match_step.status = "success"
        match_step.detail = (
            f"{len(matches)} benzer kalem bulundu."
            if matches
            else "Eşleşen kayıt yok; firma uydurulmayacak."
        )
        await emit(match_step)

    tool_step = await step("tools", "Bağlam ve araçlar")
    tools = await run_needed_tools(
        question=question,
        deps=deps,
        file_names=getattr(deps, "file_names", None),
    )
    if matches:
        tools.append(matches_as_tool(matches, name="trade_item_search"))
    tool_step.status = "success"
    tool_step.detail = _tool_step_detail(question, tools)
    await emit(tool_step)

    facts = with_domain_notes(
        advisor_answer(question, deps.session, matches),
        notes,
        product=product or (
            deps.session.product if deps.session is not None else None
        ),
    )
    if is_diagnostic_request(question) and not matches:
        facts = diagnostic_outline(question)
    if is_memory_recall(question):
        mem = next((item for item in tools if item.name == "memory_search"), None)
        if mem and mem.ok:
            facts = (
                "SOURCE_MEMORY:\n"
                + mem.summary
                + "\nBu belleğe dayan. Uydurma karar ekleme."
            )
        else:
            facts = memory_miss_facts()
    if is_company_data_ask(question):
        company = next((item for item in tools if item.name == "company_context"), None)
        if company and company.ok:
            facts = "SOURCE_MEMORY / SOURCE_USER:\n" + company.summary
        else:
            facts = company_miss_facts()
    if is_current_information(question) or is_stat_challenge(question):
        web = next((item for item in tools if item.name == "web_search"), None)
        if web is not None and web.ok and (web.data or {}).get("findings"):
            facts = "SOURCE_WEB:\n" + web.summary + "\n" + "\n".join(
                str(x) for x in (web.data or {}).get("findings")[:5]
            )
        else:
            facts = current_data_facts()
    if is_document_ask(question):
        doc = next((item for item in tools if item.name == "document_analysis"), None)
        if doc and not doc.ok:
            facts = (
                "Yüklü bir Excel, CSV veya PDF yok. Tablo uyduramam. "
                "Dosyayı gönderirseniz trend, düşüş, kanal ve ülke performansına bakarım."
            )
    if is_decision_question(question) and not is_draft_request(question):
        bits = [
            "Karar sorusu. Mail taslağı, fuar listesi veya kopyala-yapıştır teklif yazma. "
            "Firma ve fiyat uydurma."
        ]
        notes_txt = session_notes(deps.session, question=question)
        if notes_txt:
            bits.append(notes_txt)
        if matches:
            bits.append(deps.format_context(matches))
        else:
            bits.append("Eşleşen kayıt yok.")
        facts = "\n".join(bits)

    pack = await build_context_pack(
        supabase=deps.supabase,
        session=deps.session,
        account_slug=getattr(deps, "account_slug", "demo"),
        question=question,
        extra_tools=tools,
    )

    gen_step = await step("advise", "Ticari değerlendirme")
    focus = advisor_focus(question, deps.session)
    # V5.18.5 Phase 1–2: derived commercial context + controlled user-facing guidance.
    situation = build_commercial_situation(
        question,
        session=deps.session,
        focus=focus,
    )
    setattr(deps, "commercial_situation", situation)
    if should_reason(question, focus):
        advice = await compose_consultant_reply(
            generate=deps.generate,
            http=deps.http,
            question=question,
            facts=facts,
            pack=pack,
            tools=tools,
            task=getattr(deps, "goal", "") or focus,
            session=deps.session,
            matches=matches,
            situation=situation,
            focus=focus,
        )
    else:
        advice = facts
    advice = apply_commercial_guidance(
        advice,
        situation,
        question,
        session=deps.session,
        focus=focus,
    )
    web = next((item for item in tools if item.name == "web_search"), None)
    if web is not None and not web.ok and wants_web_research(question):
        if "uydur" not in advice.casefold() and "tamamlayamadım" not in advice.casefold():
            advice = fallback_web_unavailable(advice)
    gen_step.status = "success"
    gen_step.detail = "Danışman notu hazır."
    await emit(gen_step)

    context = [item.model_dump(mode="json") for item in matches]
    return advice, context, steps


def _tool_step_detail(question: str, tools: list[Any]) -> str:
    if is_memory_recall(question):
        return "Geçmiş kararlar ve bellek taranıyor."
    if wants_web_research(question):
        web = next((item for item in tools if item.name == "web_search"), None)
        if web is not None and not web.ok:
            return "Web araştırması şu anda yok; eldeki veriyle sınırlıyım."
        return "Pazar ve rakip kaynakları taranıyor."
    if is_document_ask(question) or is_diagnostic_request(question):
        return "Satış kırılımları ve dosya altyapısı kontrol ediliyor."
    return "Şirket bilgileri ve bellek kontrol ediliyor."
