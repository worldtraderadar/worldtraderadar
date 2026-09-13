"""Ticari hedef (intent runner'ından ayrı). Regex; basit soruda LLM yok."""

from __future__ import annotations

from typing import Literal

from prompts import (
    is_buyer_search,
    is_competitor_research,
    is_decision_question,
    is_diagnostic_request,
    is_document_ask,
    is_draft_request,
    is_memory_recall,
    is_planning_ask,
    is_pricing_ask,
    is_small_talk,
    is_strategy_request,
    is_supplier_search,
    is_trade_docs_ask,
    wants_technical_detail,
)

from .session import SessionState, looks_like_product

CommercialGoal = Literal[
    "casual_conversation",
    "company_question",
    "product_question",
    "market_research",
    "competitor_research",
    "buyer_search",
    "supplier_search",
    "product_matching",
    "hs_code",
    "export_strategy",
    "import_strategy",
    "pricing",
    "sales_strategy",
    "marketing_strategy",
    "business_development",
    "data_analysis",
    "document_analysis",
    "strategic_decision",
    "planning",
    "follow_up",
    "memory_recall",
]


def classify_goal(question: str, session: SessionState | None = None) -> CommercialGoal:
    text = question or ""
    if is_small_talk(text, has_history=bool(session and session.in_progress())):
        return "casual_conversation"
    if is_memory_recall(text):
        return "memory_recall"
    if is_trade_docs_ask(text):
        return "export_strategy"
    if is_document_ask(text) or is_diagnostic_request(text):
        if is_document_ask(text):
            return "document_analysis"
        return "data_analysis"
    if is_buyer_search(text) and not is_decision_question(text):
        return "buyer_search"
    if is_supplier_search(text) and not is_decision_question(text):
        return "supplier_search"
    if wants_technical_detail(text) or "hs" in text.casefold():
        return "hs_code" if "hs" in text.casefold() or any(ch.isdigit() for ch in text) else "product_matching"
    if is_competitor_research(text):
        return "competitor_research"
    if is_pricing_ask(text):
        return "pricing"
    if is_planning_ask(text):
        return "planning"
    if is_draft_request(text):
        return "sales_strategy"
    low = text.casefold()
    if "ithalat" in low:
        return "import_strategy"
    if "ihracat" in low or "satmak" in low:
        return "export_strategy"
    if "pazarlama" in low or "reklam" in low or "meta" in low:
        return "marketing_strategy"
    if "iş geliştir" in low or "is gelistir" in low:
        return "business_development"
    if is_strategy_request(text):
        return "market_research"
    if is_decision_question(text):
        return "strategic_decision"
    if session and session.product and looks_like_product(text):
        return "product_question"
    if session and (session.product or session.name):
        return "follow_up"
    if "şirket" in low or "firmamız" in low or "firmamiz" in low:
        return "company_question"
    return "strategic_decision"


def plan_summary(
    goal: CommercialGoal,
    question: str,
    session: SessionState | None,
) -> str:
    from .commercial import build_commercial_brief

    brief = build_commercial_brief(question, session=session)
    plan = (brief.human_plan or "").strip()
    if plan:
        return plan
    product = (session.product if session else None) or "ürün henüz net değil"
    market = (session.market if session else None) or "pazar henüz net değil"
    return f"Durumu {product} / {market} üzerinden ticari olarak okuyup net bir sonraki adım vereceğim."


def _tools_for_goal(goal: CommercialGoal) -> list[str]:
    mapping: dict[str, list[str]] = {
        "buyer_search": ["buyer_search", "trade_item_search", "company_context"],
        "supplier_search": ["supplier_search", "trade_item_search", "company_context"],
        "product_matching": ["product_matching", "hs_lookup", "trade_item_search"],
        "hs_code": ["hs_lookup", "product_matching", "trade_item_search"],
        "competitor_research": ["web_search", "company_context", "memory_search"],
        "memory_recall": ["memory_search", "company_context"],
        "document_analysis": ["document_analysis", "data_analysis"],
        "data_analysis": ["data_analysis", "company_context"],
        "market_research": ["trade_item_search", "web_search", "company_context"],
        "export_strategy": ["trade_item_search", "company_context", "memory_search"],
        "pricing": ["company_context", "trade_item_search"],
    }
    return mapping.get(goal, ["company_context", "memory_search", "trade_item_search"])
