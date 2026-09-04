"""V5.7 diagnostic labels only. Production evaluator/validator import etmez / değiştirmez.

Resmi skor: agents.evaluator.score_cmo_v51
Bu modül yalnızca sınıf etiketleri ve tavan analizi üretir.
"""

from __future__ import annotations

import re

from agents.commercial import is_weak_decision
from agents.evaluator import score_cmo_v51
from agents.quality_flags import is_false_certainty
from agents.validator import make_speakable

# score_cmo_reply DATA_USE token listesi — kopya; evaluator değişmez.
DATA_TOKENS = ("almanya", "fransa", "marj", "hacim", "kapasite", "zeytinyağı", "3 ")

CLASS_LABELS = {
    "A": "DATA UNDERUSE",
    "B": "SHALLOW DIAGNOSIS",
    "C": "WEAK REASONING",
    "D": "WEAK DECISION",
    "E": "WEAK UNCERTAINTY",
    "F": "WEAK ACTION",
    "G": "ROBOTIC / NON-NATURAL TURKISH",
    "H": "OTHER",
}


def data_use_ceiling(question: str, facts: str) -> int:
    pool = f"{facts or ''} {question or ''}".casefold()
    hits = sum(1 for tok in DATA_TOKENS if tok in pool)
    if hits >= 3:
        return 2
    if hits:
        return 1
    return 0


def fallback_overlap(llm: str, fallback: str) -> float:
    a = set(make_speakable(llm or "").casefold().split())
    b = set(make_speakable(fallback or "").casefold().split())
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a), 3)


def classify_anatomy(
    score: dict,
    *,
    text: str,
    question: str,
    facts: str,
    fallback: str = "",
) -> dict:
    """A–H. Resmi skoru değiştirmez. Zayıf = total < 10 veya boyut == 0."""
    dims = {
        "DATA_USE": int(score.get("DATA_USE") or 0),
        "DIAGNOSIS": int(score.get("DIAGNOSIS") or 0),
        "REASONING": int(score.get("REASONING") or 0),
        "DECISION": int(score.get("DECISION") or 0),
        "UNCERTAINTY": int(score.get("UNCERTAINTY") or 0),
        "ACTION": int(score.get("ACTION") or 0),
        "NATURAL_TURKISH": int(score.get("NATURAL_TURKISH") or 0),
    }
    total = int(score.get("total") or 0)
    ceil = data_use_ceiling(question, facts)
    weak = total < 10 or 0 in dims.values()
    classes: list[str] = []
    if dims["DATA_USE"] < ceil:
        classes.append("A")
    if dims["DIAGNOSIS"] == 0 or (weak and dims["DIAGNOSIS"] <= 1):
        classes.append("B")
    if dims["REASONING"] <= 1:
        classes.append("C")
    if dims["DECISION"] <= 1 or is_weak_decision(text or ""):
        classes.append("D")
    if dims["UNCERTAINTY"] <= 1:
        classes.append("E")
    if dims["ACTION"] <= 1:
        classes.append("F")
    if dims["NATURAL_TURKISH"] <= 1:
        classes.append("G")
    if weak and not classes:
        classes.append("H")
    overlap = fallback_overlap(text, fallback)
    return {
        "classes": classes,
        "labels": [CLASS_LABELS[c] for c in classes],
        "weak": weak,
        "data_ceiling": ceil,
        "data_gap": ceil - dims["DATA_USE"],
        "fallback_overlap": overlap,
        "fallback_like": overlap >= 0.55,
        "hedge": is_weak_decision(text or ""),
    }


def official_score(text: str, question: str, facts: str) -> dict:
    return score_cmo_v51(text, question=question, facts=facts).as_dict()


_STANCE = re.compile(
    r"(?i)("
    r"artırmaz|düşürmez|kabul etme|kabul etmem|etmemeniz|"
    r"bırakmaz|girmez|girmem|önermem|"
    r"şimdilik|şu aşamada|öncelik|"
    r"sabit tut|erteley|fob|seçerdim|tercihim|"
    r"öneririm|korur|koruma"
    r")"
)
_DEFER = re.compile(
    r"(?i)("
    r"daha sonra belirle|"
    r"stratejisini belirleriz|"
    r"dikte edemem|"
    r"seçemem|"
    r"kilitleyemem|"
    r"doğru (hamle|distribütör)ü? seçemem|"
    r"bilgiler(le birlikte| gelince).{0,40}(belirle|sun|öner)"
    r")"
)
_FLIP = re.compile(
    r"(?i)("
    r"karar[ıi]mı? değiş|"
    r"değiştirecek|"
    r"değiştirebiliriz|"
    r"eğer .{0,48}(ise|olursa|çıkarsa)|"
    r"netleş(ince|irse|tikten)|"
    r"doğrulanırsa|"
    r"altındaysa"
    r")"
)
_THIN_CTX = re.compile(
    r"(?i)(yok\.|yok |kayıtlı değil|doğrulanmad|crm yok|web yok|web bu turda yok)"
)


def is_thin_context(facts: str, question: str = "") -> bool:
    return bool(_THIN_CTX.search(f"{facts or ''} {question or ''}"))


def provisional_stance(text: str) -> dict:
    """Diagnostic only. Production evaluator'a girmez."""
    body = make_speakable(text or "")
    stance = bool(_STANCE.search(body))
    flip = bool(_FLIP.search(body))
    defer = bool(_DEFER.search(body))
    fc = is_false_certainty(body)
    ok = stance and flip and not fc
    return {
        "stance": stance,
        "flip_condition": flip,
        "defer": defer,
        "false_certainty": fc,
        "ok": ok,
        "defer_without_stance": defer and not stance,
    }


def provisional_stance_rate(rows: list[dict]) -> dict:
    thin = [
        r
        for r in rows
        if is_thin_context(r.get("context") or "", r.get("input") or "")
    ]
    n = len(thin) or 1
    oks = [r for r in thin if (r.get("provisional") or {}).get("ok")]
    stances = [r for r in thin if (r.get("provisional") or {}).get("stance")]
    flips = [r for r in thin if (r.get("provisional") or {}).get("flip_condition")]
    defers = [r for r in thin if (r.get("provisional") or {}).get("defer_without_stance")]
    return {
        "thin_n": len(thin),
        "ok_n": len(oks),
        "rate": round(len(oks) / n, 3),
        "stance_rate": round(len(stances) / n, 3),
        "flip_rate": round(len(flips) / n, 3),
        "defer_without_stance_n": len(defers),
    }
