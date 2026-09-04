"""Cevap doğrulayıcı: PASS / WARN / REJECT. LLM son kapısı."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

from prompts import is_draft_request, looks_english

from .commercial import is_false_confidence, is_weak_decision, treats_score_as_win
from .quality_flags import (
    is_brief_echo,
    is_memory_dishonest,
    is_prompt_leak,
    memory_present_from,
)
from .tools_base import ToolResult

Verdict = Literal["PASS", "WARN", "REJECT"]

_EN_LEAD = re.compile(
    r"(?i)^(i\s+|as the\s+|sure\b|of course\b|let'?s\b|hello\b|i understand\b|dear\b|best regards\b)"
)
_TRADE_TERMS = {
    "b2b", "b2c", "b2b2c", "hs", "moq", "fob", "cif", "exw", "dap", "ddp",
    "oem", "sku", "crm", "kpi", "cac", "roi", "incoterms", "linkedin",
    "pdf", "csv", "excel", "http", "https", "www", "url", "api", "eori",
    "nace", "cn", "gtip", "pla", "petg", "fdm", "iso", "ce",
    "saas", "rag", "llm",
}
_MIXED_EN = re.compile(
    r"(?i)(?<![a-z])("
    r"consider|analyze|analysing|analyzing|"
    r"customer|customers|"
    r"switch(?:ing)?|focusing|"
    r"let'?s\s+consider"
    r")(?![a-z])"
)
_EN_FUNC = {
    "the", "and", "you", "your", "this", "that", "with", "from", "have",
    "what", "would", "could", "should", "about", "lets", "however",
    "because", "through", "into", "also", "been", "will", "can", "for",
    "are", "not", "but", "which", "their", "please", "welcome",
}
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STAT = re.compile(
    r"(?i)("
    r"%\s*\d+(?:[.,]\d+)?|"
    r"\d+(?:[.,]\d+)?\s*%|"
    r"yüzde\s*\d+(?:[.,]\d+)?|"
    r"\d+(?:[.,]\d+)?\s*(milyon|milyar)\s*(euro|eur|dolar|usd|\$|€)?|"
    r"\d+(?:[.,]\d+)?\s*(euro|eur|dolar|usd)\b|"
    r"\b\d{2,}\s*ton\b"
    r")"
)
_FIRM_CLAIM = re.compile(
    r"(?i)(\d+\s*(firma|şirket|alıcı|tedarikçi)\s*(buldum|tespit|var\b))"
)
_CURRENT_CLAIM = re.compile(
    r"(?i)((bugün|şu anda|2026|bu yıl).{0,40}(büyüd|artt[ıi]|düşt|ithalat|ihracat|fiyat))"
)
_SUCCESS_CLAIM = re.compile(
    r"(?i)(\d+\s*(firma|kayıt|kaynak)\s*(buldum|taradım)|internette\s+\d+)"
)
_META_LEAK = re.compile(
    r"(?i)("
    r"güvenilmeyen|"
    r"uydurma yasak|"
    r"kural motoru|"
    r"system prompt|"
    r"rag [cç][ıi]kt[ıi]s|"
    r"kaynakta yoksa|"
    r"istatistik uydurma|"
    r"firma, fiyat, hs|"
    r"t[ií]cari zek[aâ] br[ií]f|"
    r"cmo brief|"
    r"do[gğ]al konu[sş]"
    r")"
)
_MARKDOWN = re.compile(r"[#*_`]{2,}|https?://\S+")
_TABLE = re.compile(r"(?m)^\s*\|.+\|\s*$")
_MAIL_LEAK = re.compile(
    r"(?is)("
    r"^dear\s|"
    r"best regards|"
    r"\[your company name\]|"
    r"mail tasla[gğ]|"
    r"kopyala-yap[ıi][sş]t[ıi]r"
    r")"
)
_ROBOT_OPEN = re.compile(
    r"(?i)("
    r"size yardımcı olmaktan (mutluluk|memnuniyet)|"
    r"tabii ki[,.]?\s*size|"
    r"elbette[,.]?\s*(size )?yardımcı"
    r")"
)
_EN_PASSAGE = re.compile(r"(?<![ğüşıöçĞÜŞİÖÇ])[A-Za-z][A-Za-z,'’ ]{72,}(?![ğüşıöçĞÜŞİÖÇ])")


@dataclass
class ValidationResult:
    status: Verdict
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE.split((text or "").strip()) if p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def _content_words(sentence: str) -> list[str]:
    words = re.findall(r"[A-Za-z']+", sentence.lower())
    return [w for w in words if w not in _TRADE_TERMS and len(w) > 1]


def sentence_is_english(sentence: str) -> bool:
    if re.search(r"[ğüşıöçĞÜŞİÖÇ]", sentence):
        return False
    words = _content_words(sentence)
    if len(words) < 6:
        return False
    hits = sum(1 for w in words if w in _EN_FUNC)
    return hits >= 3 or hits / max(len(words), 1) >= 0.22


def english_leak_score(text: str) -> tuple[bool, str]:
    """Uzun İngilizce pasaj / cümle oranı. Kısa teknik terim serbest."""
    body = (text or "").strip()
    if not body:
        return True, "boş cevap"
    if _EN_LEAD.match(body):
        return True, "İngilizce açılış"
    if looks_english(body) and not re.search(r"[ğüşıöçĞÜŞİÖÇ]", body[:80]):
        return True, "İngilizce paragraf"
    sents = _sentences(body)
    en = [s for s in sents if sentence_is_english(s)]
    if len(en) >= 2:
        return True, "birden fazla İngilizce cümle"
    if sents and len(en) / len(sents) >= 0.45 and len(en) >= 1 and len(sents) >= 2:
        return True, "İngilizce cümle oranı yüksek"
    if _EN_PASSAGE.search(body) and not re.search(r"[ğüşıöçĞÜŞİÖÇ]", body):
        return True, "uzun İngilizce pasaj"
    if _MIXED_EN.search(body):
        return True, "Türkçe içinde gereksiz İngilizce fiil/isim"
    return False, ""


def _haystack(parts: Iterable[str]) -> str:
    return " ".join(p for p in parts if p).casefold()


def _percent_nums(text: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(
        r"%\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*%|yüzde\s*(\d+(?:[.,]\d+)?)",
        text or "",
        re.I,
    ):
        num = match.group(1) or match.group(2) or match.group(3)
        if num:
            found.add(num.replace(",", "."))
    return found


def unsourced_stats(text: str, allowed: str) -> list[str]:
    found: list[str] = []
    pool = (allowed or "").casefold()
    pool_c = re.sub(r"\s+", "", pool)
    allowed_pct = _percent_nums(allowed or "")
    for match in _STAT.finditer(text or ""):
        token = match.group(0).strip()
        compact = re.sub(r"\s+", "", token.casefold())
        if compact and compact in pool_c:
            continue
        if token.casefold() in pool:
            continue
        token_pct = _percent_nums(token)
        if token_pct and token_pct <= allowed_pct:
            continue
        found.append(token)
    return found


_CLAIM_FRAME = re.compile(
    r"(?i)("
    r"iddia|"
    r"aktard|"
    r"varsaymadan|"
    r"söylediğin|"
    r"senin .{0,32}(gözlem|iddia|tablo|veri)|"
    r"henüz .{0,24}(doğrula|ölç|kanıt)|"
    r"doğrulanmış .{0,16}değil|"
    r"kullanıcı .{0,16}(iddia|söyl)"
    r")"
)
_CLAIM_TO_FACT = re.compile(
    r"(?i)("
    r"rakipler .{0,32}(daha ucuz|düşük)|"
    r"pazar .{0,28}(büyüd|küçüld|milyar)|"
    r"cirosu .{0,20}\d|"
    r"kesin .{0,12}(kârlı|karli|daha iyi)"
    r")"
)


def is_user_claim_stat(token: str, question: str, reply: str) -> bool:
    """Kullanıcının verdiği sayıyı iddia olarak tartışmak yeni fact değildir."""
    nums = _percent_nums(token)
    qnums = _percent_nums(question)
    if not nums or not nums <= qnums:
        return False
    if _CLAIM_TO_FACT.search(reply or "") and not _CLAIM_FRAME.search(reply or ""):
        return False
    return True


def evaluator_unsourced_stats(
    text: str, *, question: str = "", facts: str = ""
) -> list[str]:
    """Evaluator: kullanıcı iddiası ≠ model fact. Validator REJECT kuralları aynı kalır."""
    hits = unsourced_stats(text, facts or "")
    kept: list[str] = []
    for token in hits:
        if is_user_claim_stat(token, question, text):
            continue
        kept.append(token)
    return kept


def validate_response(
    text: str,
    *,
    question: str = "",
    facts: str = "",
    tools: list[ToolResult] | None = None,
    memories: list[str] | None = None,
) -> ValidationResult:
    reasons: list[str] = []
    status: Verdict = "PASS"
    body = (text or "").strip()
    tools = tools or []

    leak, why = english_leak_score(body)
    if leak:
        reasons.append(why)
        status = "REJECT"

    if _META_LEAK.search(body) or is_prompt_leak(body) or is_brief_echo(body):
        reasons.append("brief/prompt sızıntısı")
        status = "REJECT"

    if _MAIL_LEAK.search(body) and not is_draft_request(question):
        reasons.append("mail taslağı sızıntısı")
        status = "REJECT"

    allowed = _haystack(
        [
            facts,
            *[item.for_prompt() for item in tools],
            *[str(item.data) for item in tools],
        ]
    )
    stats = unsourced_stats(body, allowed)
    if stats:
        reasons.append("kaynaksız rakam: " + ", ".join(stats[:4]))
        status = "REJECT"

    failed = [item for item in tools if not item.ok]
    if failed and _SUCCESS_CLAIM.search(body):
        reasons.append("başarısız araca rağmen başarı iddiası")
        status = "REJECT"

    mem = next((item for item in tools if item.name == "memory_search"), None)
    mem_ok = bool(mem is not None and mem.ok)
    has_memory = mem_ok or memory_present_from(facts=facts, memories=memories)
    if is_memory_dishonest(body, memory_present=has_memory):
        reasons.append("boş bellekte hatırlıyorum iddiası")
        status = "REJECT"

    web = next((item for item in tools if item.name == "web_search"), None)
    if web is not None and not web.ok and _CURRENT_CLAIM.search(body):
        reasons.append("web yokken güncel olay iddiası")
        status = "REJECT"

    named = []
    for item in tools:
        named.extend(str(n) for n in (item.data or {}).get("names") or [])
    if _FIRM_CLAIM.search(body):
        if not named and not re.search(r"(?i)(kayıt|kart).{0,12}(yok|bulunamad)", body):
            if not re.search(r"(?i)tamamlayamadım|bulunamadı|uydurm", body):
                reasons.append("kaynaksız firma sayısı")
                status = "REJECT"

    if len(body) > 1600:
        reasons.append("gereksiz uzun")
        if status == "PASS":
            status = "WARN"

    if _MARKDOWN.search(body):
        spoken = make_speakable(body)
        if _MARKDOWN.search(spoken) or _TABLE.search(spoken):
            reasons.append("markdown/url ses için ağır")
            if status == "PASS":
                status = "WARN"
        # Hafif **bold**/liste: make_speakable sonrası ses sözleşmesi sağlanır; REJECT/WARN yok.

    if treats_score_as_win(body):
        reasons.append("eşleşme skorunu kazanma ihtimali yaptı")
        status = "REJECT"
    if is_false_confidence(body):
        reasons.append("kaynaksız mutlak kesinlik")
        status = "REJECT"
    if is_weak_decision(body) and status == "PASS":
        reasons.append("zayıf karar kaçışı")
        status = "WARN"
    if _ROBOT_OPEN.search(body) and status == "PASS":
        reasons.append("robot asistan açılışı")
        status = "WARN"
    q = (question or "").strip()
    if q and len(q) > 28 and q in body and status == "PASS":
        reasons.append("kullanıcı sorusunu tekrar etti")
        status = "WARN"

    return ValidationResult(status=status, reasons=reasons)


def retry_hint(result: ValidationResult) -> str:
    bits = "; ".join(result.reasons) or "kalite"
    return (
        f"Düzelt: {bits}. Kaynakta olmayan rakam ve firma yok. "
        "Yalnızca Türkçe ticari değerlendirme. "
        "Cümleyi yarım bırakma; 4-5 kısa cümlede bitir."
    )


def make_speakable(text: str) -> str:
    body = (text or "").strip()
    body = re.sub(r"https?://\S+", "", body)
    body = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", body)
    body = re.sub(r"\*\*([^*]+)\*\*", r"\1", body)
    body = re.sub(r"__([^_]+)__", r"\1", body)
    body = re.sub(r"`+", "", body)
    body = re.sub(r"[#*_`]{2,}", "", body)
    body = re.sub(r"(?m)^\s*[-*]\s+", "", body)
    body = re.sub(r"(?m)^\s*\d+[.)]\s+", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return re.sub(r"[ \t]{2,}", " ", body).strip()
