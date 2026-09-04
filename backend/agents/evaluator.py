"""Internal CMO puanı (0–2). Kullanıcıya gösterilmez."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .commercial import commercial_fallback_reply, is_weak_decision, treats_score_as_win
from .validator import english_leak_score, make_speakable

_REASON = re.compile(
    r"(?i)(çünkü|zira|bu yüzden|şu nedenle|marj|hacim|kapasite|risk|fırsat|trade-?off)"
)
# Geçici ticari duruş. Bare "kal" yok: kalite/kalır/kalıp false-positive üretiyordu.
# "daha sonra bakalım" / "bilmiyorum" / "belirleriz" bu listeye girmez.
_DECISION = re.compile(
    r"(?i)("
    r"bırakmaz|"
    r"düşürmez|düşürmeyin|düşürmemen|"
    r"öncelik|önceliğ|"
    r"ben olsam|tercihim|öneririm|önermem|"
    r"gitme|hemen temas|"
    r"ikinci (bir )?test|"
    r"dengede kal|"
    r"fiyat[ıi]?\s*sabit tut|sabit tutmak|sabitleme karar|"
    r"durumunu koru|"
    r"mevcut fiyat[ıi]\s*koru|"
    r"mevcut.{0,28}koru|"
    r"marj[ıi]n?[ıi]?\s*koru|"
    r"hemen düşürme|"
    r"(gelene|netle[sş]ene) kadar.{0,40}(koru|tut|beklet)|"
    r"geçici olarak ertele|erteleme duruş|geçici duruş|"
    r"mevcut m[uü][sş]teri.{0,32}önceli|"
    r"kabul etmem(?!iz)|kabul etmeyin|kabul etmemek|"
    r"yönelmeyin|önceliklendirmeyin|"
    r"bırakmak risk|"
    r"beklemek yerine|"
    r"ayırmak daha güvenli|"
    r"\bfob\b.{0,32}(seç|sun|gidin)|(seçin|sunun).{0,24}\bfob\b"
    r")"
)
_UNCERTAIN_OK = re.compile(
    r"(?i)(mevcut veri|sinyal|gözlem|iddia|doğrulan|değiştirecek|henüz ölçül|"
    r"kayıtlı değil|uydurm)"
)
_FALSE_SURE = re.compile(
    r"(?i)(kesinlikle daha|kesin daha|kesin alır|fransa kesin|almanya kesin)"
)
_ACTION = re.compile(
    r"(?i)(önce |sonraki adım|karşılaştır|ölçelim|teklif|doğrula|sor|"
    r"önceliklendir|test pazar)"
)
_HEADING = re.compile(r"(?m)^(VERİ|DEĞERLENDİRME|ÖNERİ|FIRSAT|RİSK|KARAR|İLK AKSİYON)\s*:")
_TEMPLATE = "ben şu aşamada almanya'yı bırakmazdım"


@dataclass
class CmoScore:
    data_use: int
    reasoning: int
    decision: int
    uncertainty: int
    action: int
    natural_language: int

    @property
    def total(self) -> int:
        return (
            self.data_use
            + self.reasoning
            + self.decision
            + self.uncertainty
            + self.action
            + self.natural_language
        )

    @property
    def average(self) -> float:
        return self.total / 6

    def as_dict(self) -> dict[str, int | float]:
        return {
            "DATA_USE": self.data_use,
            "REASONING": self.reasoning,
            "DECISION": self.decision,
            "UNCERTAINTY": self.uncertainty,
            "ACTION": self.action,
            "NATURAL_LANGUAGE": self.natural_language,
            "total": self.total,
            "average": round(self.average, 3),
        }


def _clamp(value: int) -> int:
    return 0 if value < 0 else 2 if value > 2 else value


def has_decision_stance(text: str) -> bool:
    """Geçici/net ticari duruş var mı? Zayıf kaçış (is_weak_decision) ayrı."""
    return bool(_DECISION.search(make_speakable(text or "")))


def score_cmo_reply(
    text: str,
    *,
    question: str = "",
    facts: str = "",
    company: str = "",
    memories: list[str] | None = None,
    matches: list | None = None,
) -> CmoScore:
    body = make_speakable(text or "")
    low = body.casefold()
    pool = " ".join(
        [
            facts or "",
            company or "",
            " ".join(memories or []),
            question or "",
        ]
    ).casefold()

    data = 0
    hits = 0
    for token in ("almanya", "fransa", "marj", "hacim", "kapasite", "zeytinyağı", "3 "):
        if token in low and (token in pool or token in (question or "").casefold()):
            hits += 1
    if matches:
        hits += 1
    if hits >= 3:
        data = 2
    elif hits:
        data = 1

    if _REASON.search(body) and ("marj" in low or "hacim" in low or "lead" in low or "müşteri" in low):
        reason = 2
    elif _REASON.search(body):
        reason = 1
    else:
        reason = 0

    if is_weak_decision(body):
        decision = 0
    elif has_decision_stance(body) and _REASON.search(body):
        decision = 2
    elif has_decision_stance(body):
        decision = 1
    else:
        decision = 0

    if _FALSE_SURE.search(body) or treats_score_as_win(body):
        uncertainty = 0
    elif _UNCERTAIN_OK.search(body):
        uncertainty = 2
    else:
        uncertainty = 1

    qn = body.count("?")
    if _ACTION.search(body) and qn <= 3:
        action = 2
    elif _ACTION.search(body) or (qn == 1):
        action = 1
    else:
        action = 0

    leak, _ = english_leak_score(body)
    headings = len(_HEADING.findall(body))
    template = _TEMPLATE in low
    if leak:
        natural = 0
    elif headings >= 3:
        natural = 0
    elif headings or template or body.count("\n") > 6:
        natural = 1
    else:
        natural = 2

    return CmoScore(
        data_use=_clamp(data),
        reasoning=_clamp(reason),
        decision=_clamp(decision),
        uncertainty=_clamp(uncertainty),
        action=_clamp(action),
        natural_language=_clamp(natural),
    )


def compare_llm_and_fallback(
    llm_text: str,
    fallback_text: str,
    **kwargs,
) -> dict:
    llm = score_cmo_reply(llm_text, **kwargs)
    fb = score_cmo_reply(fallback_text, **kwargs)
    winner = "llm" if llm.total > fb.total else "fallback" if fb.total > llm.total else "tie"
    same_template = make_speakable(llm_text).casefold() == make_speakable(fallback_text).casefold()
    return {
        "llm": llm.as_dict(),
        "fallback": fb.as_dict(),
        "winner": winner,
        "llm_copies_fallback": same_template,
        "gap": llm.total - fb.total,
    }


def fallback_for(question: str, **brief_kw) -> str:
    from .commercial import build_commercial_brief

    brief = build_commercial_brief(question, **brief_kw)
    return commercial_fallback_reply(brief, question) or ""


_DIAGNOSIS = re.compile(
    r"(?i)(fiyat baskısı.{0,24}(gözlem|iddia)|mevcut vs yeni|kazan[ıi]m|"
    r"üç.{0,12}(lead|m[uü][sş]teri)|marj.{0,16}(ölç|kabul)|ikinci test)"
)


@dataclass
class CmoScoreV51:
    data_use: int
    diagnosis: int
    reasoning: int
    decision: int
    uncertainty: int
    action: int
    natural_turkish: int
    brief_echo: bool = False
    english_leakage: bool = False
    unsupported_fact: bool = False
    false_certainty: bool = False
    prompt_leak: bool = False
    memory_dishonesty: bool = False

    @property
    def total(self) -> int:
        return (
            self.data_use
            + self.diagnosis
            + self.reasoning
            + self.decision
            + self.uncertainty
            + self.action
            + self.natural_turkish
        )

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "DATA_USE": self.data_use,
            "DIAGNOSIS": self.diagnosis,
            "REASONING": self.reasoning,
            "DECISION": self.decision,
            "UNCERTAINTY": self.uncertainty,
            "ACTION": self.action,
            "NATURAL_TURKISH": self.natural_turkish,
            "total": self.total,
            "BRIEF_ECHO": self.brief_echo,
            "ENGLISH_LEAKAGE": self.english_leakage,
            "UNSUPPORTED_FACT": self.unsupported_fact,
            "FALSE_CERTAINTY": self.false_certainty,
            "PROMPT_LEAK": self.prompt_leak,
            "MEMORY_DISHONESTY": self.memory_dishonesty,
        }


def score_cmo_v51(
    text: str,
    *,
    question: str = "",
    facts: str = "",
    memories: list[str] | None = None,
) -> CmoScoreV51:
    from .quality_flags import (
        is_brief_echo,
        is_false_certainty,
        is_memory_dishonest,
        is_prompt_leak,
        memory_present_from,
    )
    from .validator import evaluator_unsourced_stats

    base = score_cmo_reply(text, question=question, facts=facts, memories=memories)
    body = make_speakable(text or "")
    low = body.casefold()
    echo = is_brief_echo(text or "")
    leak, _ = english_leak_score(body)
    diag = 2 if _DIAGNOSIS.search(body) else 1 if any(
        tok in low for tok in ("marj", "müşteri", "pazar", "kapasite", "lead")
    ) else 0
    natural = 0 if echo or leak else base.natural_language
    return CmoScoreV51(
        data_use=base.data_use,
        diagnosis=_clamp(diag),
        reasoning=0 if echo else base.reasoning,
        decision=base.decision,
        uncertainty=0 if is_false_certainty(body) else base.uncertainty,
        action=base.action,
        natural_turkish=_clamp(natural),
        brief_echo=echo,
        english_leakage=leak,
        unsupported_fact=bool(
            evaluator_unsourced_stats(body, question=question, facts=facts)
        ),
        false_certainty=is_false_certainty(body) or treats_score_as_win(body),
        prompt_leak=is_prompt_leak(text or ""),
        memory_dishonesty=is_memory_dishonest(
            body, memory_present=memory_present_from(facts=facts, memories=memories)
        ),
    )
