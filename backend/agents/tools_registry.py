"""Açık sınırlı araç kaydı. LLM kod çalıştırmaz."""

from __future__ import annotations

from typing import Any

from prompts import is_current_information, is_document_ask, is_diagnostic_request, wants_web_research

from .company import format_profile, load_company_profile
from .documents import analyze_documents
from .memory_store import search_memories
from .tools_base import ToolResult, ToolSource
from .web_search import web_search


async def run_needed_tools(
    *,
    question: str,
    deps: Any,
    file_names: list[str] | None = None,
) -> list[ToolResult]:
    results: list[ToolResult] = []
    slug = getattr(deps, "account_slug", None) or "demo"

    profile = await load_company_profile(
        deps.supabase, account_slug=slug, session=deps.session
    )
    company_text = format_profile(profile)
    results.append(
        ToolResult(
            name="company_context",
            ok=bool(company_text),
            summary=company_text or "Kayıtlı şirket profili yok.",
            data=profile,
            provenance="SOURCE_MEMORY" if company_text else "TOOL_FAILED",
        )
    )

    memories = await search_memories(
        deps.supabase,
        account_slug=slug,
        query=question,
        session=deps.session,
        limit=6,
    )
    results.append(
        ToolResult(
            name="memory_search",
            ok=bool(memories),
            summary="\n".join(f"- {item}" for item in memories) or "Eşleşen bellek yok.",
            data={"items": memories},
            provenance="SOURCE_MEMORY" if memories else "TOOL_FAILED",
        )
    )

    if is_document_ask(question) or is_diagnostic_request(question):
        doc = analyze_documents(file_names=file_names, question=question)
        if not doc.provenance:
            doc.provenance = "SOURCE_DATABASE" if doc.ok else "TOOL_FAILED"
        results.append(doc)

    if wants_web_research(question) or is_current_information(question):
        web = await web_search(deps.http, question)
        results.append(web)

    return results


def matches_as_tool(matches: list[Any], *, name: str = "trade_item_search") -> ToolResult:
    names: list[str] = []
    for item in matches or []:
        label = (
            getattr(item, "organization_name", None)
            or getattr(item, "name", None)
            or (item.get("organization_name") if isinstance(item, dict) else None)
            or (item.get("product_name") if isinstance(item, dict) else None)
        )
        if label:
            names.append(str(label))
    if not matches:
        return ToolResult(
            name=name,
            ok=True,
            summary="Veritabanında eşleşen kayıt bulunamadı. Firma uydurulmayacak.",
            data={"count": 0, "names": []},
            provenance="SOURCE_DATABASE",
        )
    return ToolResult(
        name=name,
        ok=True,
        summary=f"{len(matches)} kayıt bulundu. Kartlardaki isimler: {', '.join(names[:8]) or 'isim yok'}.",
        data={"count": len(matches), "names": names[:12]},
        sources=[
            ToolSource(title=label, url="", confidence="high") for label in names[:8]
        ],
        provenance="SOURCE_RAG",
    )
