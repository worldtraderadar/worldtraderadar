"""Kalite bayrakları. Kullanıcıya gösterilmez. Validator'dan bağımsız tespit."""

from __future__ import annotations

import re

_BRIEF_ECHO = re.compile(
    r"(?is)("
    r"t[ií]cari\s*zek[aâ]\s*br[ií]f|"
    r"cmo\s*brief|"
    r"do[gğ]al\s*konu[sş]|"
    r"bellek\s*bo[sş]\s*:|"
    r"kullan[iı]c[iı]\s*bilgisi\s*\(fact|"
    r"\bprovenance\s*:|"
    r"tool\s*result|"
    r"\bvalidator\s*:|"
    r"system\s*says|"
    r"\bmod:\s*decision\b|"
    r"buyers?\s*a\s*/\s*b\s*/\s*c|"
    r"(?:^|\n)(?:data|decision|diagnosis|tool output)\s*:|"
    r"kopyalama,\s*yorumla"
    r")"
)
_PROMPT_LEAK = re.compile(
    r"(?i)("
    r"system prompt|"
    r"gizli zincir|"
    r"retry_contract|"
    r"evaluator|"
    r"uydurma yasak"
    r")"
)
_FAKE_MEMORY = re.compile(
    r"(?i)("
    r"\bhatırl[ıi]yorum\b|"
    r"ge[cç]en\s+(konu[sş]mada|sefer|turda)|"
    r"sen\s+daha\s+[oö]nce"
    r")"
)
_FALSE_CERT = re.compile(
    r"(?i)("
    r"fransa\s+kesin|"
    r"almanya\s+kesin|"
    r"kesin\s+daha\s+(k[aâ]rl[ıi]|iyi)|"
    r"kesin\s+al[ıi]r|"
    r"kesin\s+iyi\s+m[uü][sş]teri|"
    r"sat[iı]n\s+alma\s+ihtimali\s+y[uü]ksek"
    r")"
)


def is_brief_echo(text: str) -> bool:
    return bool(_BRIEF_ECHO.search(text or ""))


def is_prompt_leak(text: str) -> bool:
    return bool(_PROMPT_LEAK.search(text or ""))


def is_memory_dishonest(text: str, *, memory_present: bool) -> bool:
    if memory_present:
        return False
    return bool(_FAKE_MEMORY.search(text or ""))


def is_false_certainty(text: str) -> bool:
    return bool(_FALSE_CERT.search(text or ""))


def memory_present_from(*, facts: str = "", memories: list[str] | None = None) -> bool:
    if memories:
        return any(str(item).strip() for item in memories)
    body = (facts or "").casefold()
    if re.search(r"bellek\s*bo[sş]", body):
        return False
    if re.search(r"bellek\s*\(yalnızca kayıtlı", body):
        return True
    return False
