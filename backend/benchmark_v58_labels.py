"""V5.8 decision anatomy labels. Production evaluator'a bağlanmaz. Canlı inference yok."""

from __future__ import annotations

# V5.7.1 B (current CMO_CONTRACT stance line). Labels from full-text review.
# score_cmo_v51._DECISION is not modified.

CAT = {
    "raise_price": "A",
    "discount": "A",
    "margin_drop": "A",
    "price_pressure": "A",
    "competitor_pressure": "A",
    "volume_vs_margin": "A",
    "lost_quote": "A",
    "new_buyer_priority": "B",
    "buyer_abc": "B",
    "sales_drop": "B",
    "keep_customer": "B",
    "churn_risk": "B",
    "repeat_orders": "B",
    "quote_conversion": "B",
    "strategic_small": "B",
    "mix_priority": "B",
    "fair_leads": "B",
    "market_choice": "C",
    "germany_france": "C",
    "new_country": "C",
    "uncertain_data": "C",
    "cheap_price_market": "C",
    "distributor_pick": "C",
    "collection_risk": "D",
    "say_no": "D",
    "accept_order": "D",
    "moq": "E",
    "incoterms": "E",
    "supplier_choice": "E",
    "capacity": "F",
    "capacity_split": "F",
    "today_plan": "G",
}

CAT_NAME = {
    "A": "PRICE",
    "B": "CUSTOMER",
    "C": "MARKET",
    "D": "RISK",
    "E": "NEGOTIATION",
    "F": "CAPACITY",
    "G": "TIMING",
    "H": "OTHER",
}

# verdict: REAL_DECISION_MISS | SCORER_MISS | REAL_OK | SCORER_FALSE_POS
# lang: HEDGE DEFER QUESTION_ONLY MULTIPLE_OPTIONS CONDITIONAL_STANCE DIRECT_STANCE ACTION_WITHOUT_STANCE
LABELS = {
    "market_choice": {"verdict": "REAL_OK", "lang": "CONDITIONAL_STANCE", "regex": "ben olsam"},
    "germany_france": {"verdict": "SCORER_MISS", "lang": "CONDITIONAL_STANCE", "regex": "ikinci bir test ≠ ikinci test"},
    "price_pressure": {"verdict": "SCORER_MISS", "lang": "DIRECT_STANCE", "regex": "önceliğinizi k→ğ"},
    "raise_price": {"verdict": "REAL_DECISION_MISS", "lang": "DEFER", "regex": ""},
    "discount": {"verdict": "REAL_OK", "lang": "CONDITIONAL_STANCE", "regex": "ben olsam"},
    "margin_drop": {"verdict": "REAL_OK", "lang": "DIRECT_STANCE", "regex": "kalite contains kal"},
    "sales_drop": {"verdict": "REAL_DECISION_MISS", "lang": "QUESTION_ONLY", "regex": ""},
    "new_buyer_priority": {"verdict": "REAL_DECISION_MISS", "lang": "QUESTION_ONLY", "regex": ""},
    "buyer_abc": {"verdict": "REAL_OK", "lang": "DIRECT_STANCE", "regex": "öncelik"},
    "supplier_choice": {"verdict": "REAL_DECISION_MISS", "lang": "DEFER", "regex": ""},
    "capacity": {"verdict": "REAL_OK", "lang": "CONDITIONAL_STANCE", "regex": "öneririm"},
    "volume_vs_margin": {"verdict": "SCORER_MISS", "lang": "CONDITIONAL_STANCE", "regex": "önermem ≠ öneririm"},
    "competitor_pressure": {"verdict": "REAL_OK", "lang": "DIRECT_STANCE", "regex": "öncelikle + kalite"},
    "new_country": {"verdict": "REAL_DECISION_MISS", "lang": "DEFER", "regex": ""},
    "keep_customer": {"verdict": "SCORER_FALSE_POS", "lang": "HEDGE", "regex": "kalır"},
    "collection_risk": {"verdict": "REAL_OK", "lang": "DIRECT_STANCE", "regex": "öncelikle; kabul etmem unscored"},
    "moq": {"verdict": "SCORER_MISS", "lang": "CONDITIONAL_STANCE", "regex": ""},
    "incoterms": {"verdict": "SCORER_MISS", "lang": "DIRECT_STANCE", "regex": "FOB/kabul yok"},
    "today_plan": {"verdict": "REAL_OK", "lang": "DIRECT_STANCE", "regex": "öncelikli"},
    "uncertain_data": {"verdict": "REAL_DECISION_MISS", "lang": "QUESTION_ONLY", "regex": ""},
    "quote_conversion": {"verdict": "SCORER_MISS", "lang": "DIRECT_STANCE", "regex": "önceliğiniz k→ğ"},
    "churn_risk": {"verdict": "SCORER_FALSE_POS", "lang": "DEFER", "regex": "kalıp"},
    "repeat_orders": {"verdict": "SCORER_MISS", "lang": "DIRECT_STANCE", "regex": "aksiyon var fiil yok"},
    "distributor_pick": {"verdict": "REAL_DECISION_MISS", "lang": "QUESTION_ONLY", "regex": ""},
    "cheap_price_market": {"verdict": "REAL_OK", "lang": "DIRECT_STANCE", "regex": "öncelikle"},
    "strategic_small": {"verdict": "REAL_OK", "lang": "CONDITIONAL_STANCE", "regex": "öncelik"},
    "mix_priority": {"verdict": "REAL_DECISION_MISS", "lang": "DEFER", "regex": ""},
    "capacity_split": {"verdict": "REAL_OK", "lang": "DIRECT_STANCE", "regex": "kalırsınız"},
    "accept_order": {"verdict": "SCORER_MISS", "lang": "DIRECT_STANCE", "regex": "kabul etmeyin unscored"},
    "fair_leads": {"verdict": "SCORER_MISS", "lang": "DIRECT_STANCE", "regex": "önceliğiniz k→ğ"},
    "lost_quote": {"verdict": "SCORER_MISS", "lang": "DIRECT_STANCE", "regex": "düşürmeyin ≠ düşürmez"},
    "say_no": {"verdict": "REAL_OK", "lang": "CONDITIONAL_STANCE", "regex": "kalmak"},
}
