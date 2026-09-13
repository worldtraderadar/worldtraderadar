from __future__ import annotations

import re

from prompts import (
    is_buyer_firm_hunt,
    is_buyer_search,
    is_commercial_start,
    is_company_data_ask,
    is_current_information,
    is_customer_find_ask,
    is_decision_question,
    is_diagnostic_request,
    is_document_ask,
    is_draft_request,
    is_general_intake,
    is_incoterm_ask,
    is_language_barrier,
    is_memory_recall,
    is_method_question,
    is_mixed_commercial_start,
    is_payment_ask,
    is_planning_ask,
    is_reach_language_mix,
    is_ratio_ask,
    is_sample_ask,
    is_small_talk,
    is_sell_request,
    is_stage_followup,
    is_stat_challenge,
    is_strategy_request,
    is_supplier_search,
    is_trade_docs_ask,
    wants_data_search,
)

from .session import (
    SessionState,
    classify_option_pick,
    extract_product_slot,
    looks_like_capacity,
    looks_like_product,
)
from .types import Intent

_TOKEN = re.compile(r"[a-z0-9ğüşöçıİâîû]+", re.IGNORECASE)

BUYER_TERMS = {
    "alici",
    "alıcı",
    "musteri",
    "müşteri",
    "buyer",
    "buyers",
    "purchaser",
    "importer",
    "ithalatci",
    "ithalatçı",
}

SUPPLIER_TERMS = {
    "tedarikci",
    "tedarikçi",
    "uretici",
    "üretici",
    "supplier",
    "suppliers",
    "manufacturer",
    "exporters",
    "exporter",
    "ithalatci",
    "ithalatçı",
    "ihracatci",
    "ihracatçı",
    "firma",
    "firmalar",
    "satici",
    "satıcı",
    "kim",
    "nereden",
    "kaynak",
}

MATCHING_TERMS = {
    "benzer",
    "esdeger",
    "eşdeğer",
    "muadil",
    "alternatif",
    "eslestir",
    "eşleştir",
    "matching",
    "similar",
    "hs",
    "sinif",
    "sınıf",
    "kod",
    "substitute",
    "equivalent",
}

ADVISOR_TERMS = {
    "strateji",
    "risk",
    "tarife",
    "gumruk",
    "gümrük",
    "ihracat",
    "ithalat",
    "pazar",
    "danismanlik",
    "danışmanlık",
    "advisor",
    "strategy",
    "market",
    "lojistik",
    "hedef",
}


def classify_intent(
    question: str,
    session: SessionState | None = None,
) -> tuple[Intent, dict[str, int]]:
    empty = {
        "chat": 0,
        "intake": 0,
        "sector_chat": 0,
        "trade_advisor": 0,
        "buyer_finder": 0,
        "supplier_finder": 0,
        "product_matching": 0,
    }
    awaiting = bool(session and session.awaiting())
    has_history = bool(session and session.in_progress())
    slots_open = bool(session and session.slots_open() and session.product)

    if is_buyer_search(question) and not is_decision_question(question):
        if (
            is_draft_request(question)
            or is_sample_ask(question)
            or is_language_barrier(question)
        ):
            if session is not None:
                session.last_ask = None
            return "trade_advisor", {**empty, "trade_advisor": 1}
        if (
            is_commercial_start(question)
            or is_mixed_commercial_start(question)
            or is_reach_language_mix(question)
        ):
            if session is not None:
                session.last_ask = None
            return "trade_advisor", {**empty, "trade_advisor": 1}
        if is_customer_find_ask(question) and session is not None and session.product:
            session.last_ask = None
            return "trade_advisor", {**empty, "trade_advisor": 1}
        if session is not None:
            session.last_ask = None
        return "buyer_finder", {**empty, "buyer_finder": 1}
    if is_buyer_firm_hunt(question) and not is_decision_question(question):
        if session is not None:
            session.last_ask = None
        return "buyer_finder", {**empty, "buyer_finder": 1}
    if is_supplier_search(question) and not is_decision_question(question):
        if session is not None:
            session.last_ask = None
        return "supplier_finder", {**empty, "supplier_finder": 1}
    if (
        is_memory_recall(question)
        or is_diagnostic_request(question)
        or is_document_ask(question)
        or is_trade_docs_ask(question)
        or is_decision_question(question)
        or is_company_data_ask(question)
        or is_stat_challenge(question)
        or is_current_information(question)
        or (is_planning_ask(question) and session and session.product)
    ):
        if session is not None:
            session.last_ask = None
        return "trade_advisor", {**empty, "trade_advisor": 1}
    if (
        is_strategy_request(question)
        or is_method_question(question)
        or is_language_barrier(question)
        or is_draft_request(question)
        or is_sample_ask(question)
        or is_sell_request(question)
        or is_payment_ask(question)
        or is_ratio_ask(question)
        or is_incoterm_ask(question)
    ):
        if session is not None:
            session.last_ask = None
        return "trade_advisor", {**empty, "trade_advisor": 1}
    if session and session.product and looks_like_capacity(question):
        return "intake", {**empty, "intake": 1}
    if session and session.last_ask == "choice":
        picked = classify_option_pick(question)
        if picked:
            session.last_ask = None
            return picked, {**empty, picked: 1}
        session.last_ask = None
        return "trade_advisor", {**empty, "trade_advisor": 1}
    if (
        session
        and session.product
        and session.last_advisor_kind
        and is_stage_followup(question)
        and not is_buyer_search(question)
        and not is_supplier_search(question)
        and not is_buyer_firm_hunt(question)
    ):
        return "trade_advisor", {**empty, "trade_advisor": 1}
    if awaiting:
        return "intake", {**empty, "intake": 1}
    if slots_open and not wants_data_search(question):
        return "intake", {**empty, "intake": 1}
    if (
        session
        and session.product
        and session.last_advisor_kind
        and not is_buyer_search(question)
        and not is_supplier_search(question)
        and not is_buyer_firm_hunt(question)
        and not wants_data_search(question)
    ):
        return "trade_advisor", {**empty, "trade_advisor": 1}
    if is_small_talk(question, has_history=has_history, awaiting=awaiting):
        return "chat", {**empty, "chat": 1}
    if is_general_intake(question) and not (session and session.product):
        return "intake", {**empty, "intake": 1}
    if session and session.product and is_general_intake(question):
        return "trade_advisor", {**empty, "trade_advisor": 1}
    if looks_like_product(question) and not wants_data_search(question):
        if len(question.split()) <= 6 or extract_product_slot(question):
            return "intake", {**empty, "intake": 1}
    if not wants_data_search(question):
        return "sector_chat", {**empty, "sector_chat": 1}

    tokens = set(_TOKEN.findall(question.casefold()))
    scores = {
        **empty,
        "buyer_finder": _score(tokens, BUYER_TERMS, question),
        "supplier_finder": _score(tokens, SUPPLIER_TERMS, question),
        "product_matching": _score(tokens, MATCHING_TERMS, question),
        "trade_advisor": _score(tokens, ADVISOR_TERMS, question),
    }
    if "hs" in tokens or re.search(r"\b\d{4}(\.\d{2})?\b", question):
        scores["product_matching"] += 2
    if any(term in question.casefold() for term in ("kim üret", "kim sat", "tedarik")):
        scores["supplier_finder"] += 3
    if any(term in question.casefold() for term in ("alıcı", "alici", "müşteri", "musteri", "buyer")):
        scores["buyer_finder"] += 3

    winner = max(
        ("buyer_finder", "supplier_finder", "product_matching", "trade_advisor"),
        key=lambda key: scores[key],
    )
    if scores[winner] == 0:
        return "trade_advisor", scores
    return winner, scores  # type: ignore[return-value]


def _score(tokens: set[str], lexicon: set[str], question: str) -> int:
    lowered = question.casefold()
    hits = sum(1 for term in lexicon if term in tokens or term in lowered)
    return hits
