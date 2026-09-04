"""CMO / ticari zekâ: teşhis, karar, öncelik. Reliability'den ayrı katman."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from prompts import (
    is_competitor_research,
    is_decision_question,
    is_diagnostic_request,
    is_planning_ask,
    is_stat_challenge,
)

Mode = Literal["decision", "diagnostic", "buyer_priority", "matching", "general"]

_LEAD_COUNT = re.compile(
    r"(?i)(\d+)\s*(müşteri|musteri|alıcı|alici|firma|lead)"
)
_PRICE_PRESSURE = re.compile(
    r"(?i)(fiyat\s*bask[ıi]|marj.{0,12}(d[uü][sş]|yok|ince)|ucuz|price\s*pressure)"
)
def is_market_switch(text: str) -> bool:
    body = text or ""
    de = bool(re.search(r"(?i)almanya", body))
    fr = bool(re.search(r"(?i)fransa", body))
    if de and fr:
        return True
    return bool(
        re.search(r"(?i)(vazge[cç].{0,32}(pazar|[uü]lke)|y[oö]nelmeliyim)", body)
    )
_PRICE_CUT = re.compile(
    r"(?i)(fiyat[ıi]?\s*d[uü][sş][uü]r|indirim\s*yap|ucuz(lat|a))"
)
_NEW_VS_EXISTING = re.compile(
    r"(?i)(yeni\s*m[uü][sş]teri|mevcut\s*m[uü][sş]teri|yeni\s*mi\s*mevcut)"
)
_BEST_BUYER = re.compile(
    r"(?i)(en\s*iyi\s*(buyer|al[iı]c[iı]|m[uü][sş]teri)|hangisine\s*[oö]ncelik|"
    r"[oö]ncelik\s*ver)"
)
_SCORE_AS_WIN = re.compile(
    r"(?i)("
    r"(%\s*\d+|yüzde\s*\d+).{0,24}(ihtimal|m[uü][sş]teri\s*olur|kesin\s*al[ıi]r|kazanma)|"
    r"(e[sş]le[sş]me|matching)\s*skor.{0,32}(kazanma|ihtimal|win\s*prob)"
    r")"
)
_WEAK_BOTH = re.compile(
    r"(?i)(ikisi\s*de\s*(olabilir|de[gğ]erlendirilebilir)|hangisiyse)"
)
_HESITATION = re.compile(
    r"(?i)(fransa'?y[ıi]\s*da\s*analiz|daha\s*sonra\s*bakar[ıi]z|"
    r"ikisini\s*de\s*ara[sş]t[ıi]r)"
)


_HIGH_VOL_LOW_MARGIN = re.compile(
    r"(?i)(y[uü]ksek\s*hacim|çok\s*y[uü]ksek\s*hacim).{0,40}(marj\s*d[uü][sş]|d[uü][sş][uü]k\s*marj)"
)
_LOW_VOL_HIGH_MARGIN = re.compile(
    r"(?i)(az\s*hacim|d[uü][sş][uü]k\s*hacim).{0,40}(y[uü]ksek\s*marj)"
)
_FRANCE_EMPTY = re.compile(
    r"(?i)fransa.{0,32}(hi[cç]\s*m[uü][sş]teri|m[uü][sş]terim\s*yok)|"
    r"(hi[cç]\s*m[uü][sş]teri|m[uü][sş]terim\s*yok).{0,32}fransa"
)
_COMPETITOR_CHEAP = re.compile(
    r"(?i)raki[pb].{0,48}(d[uü][sş][uü]k|ucuz|%\s*\d+|yüzde\s*\d+)"
)
_EXISTING_STABLE = re.compile(
    r"(?i)(mevcut\s*m[uü][sş]teri.{0,16}(ayn[ıi]|sipari[sş])|"
    r"yeni\s*m[uü][sş]teri\s*(gelmiyor|yok|durdu))"
)
_CAPACITY_TIGHT = re.compile(
    r"(?i)(kapasite.{0,12}(d[uü][sş][uü]k|s[iı]n[iı]rl[iı]|az)|d[uü][sş][uü]k\s*kapasite)"
)
_PRICE_HIKE = re.compile(r"(?i)fiyat[ıi]?\s*art[ıi]r")
_MOQ_CUT = re.compile(r"(?i)(moq.{0,32}d[uü][sş]|d[uü][sş][uü]r.{0,24}moq)")
_SUPPLIER_PICK = re.compile(
    r"(?i)((tedarik[cç]i|distrib[uü]t[oö]r).{0,40}(se[cç]|hangi)|"
    r"iki\s+(tedarik[cç]i|distrib[uü]t[oö]r))"
)
_INCOTERMS = re.compile(r"(?i)\b(fob|cif)\b")
_CREDIT_ACCEPT = re.compile(
    r"(?i)("
    r"kabul\s*m[uü]|reddedeyim|sipari[sş]i kabul|"
    r"ödemesi gecik|hay[ıi]r demeli"
    r")"
)
_CONVERSION_DROP = re.compile(r"(?i)d[oö]n[uü][sş][uü]m\s*d[uü][sş]")
_REPEAT_ORDERS = re.compile(r"(?i)tekrar sipari[sş]")
_MARGIN_DROP = re.compile(r"(?i)marj.{0,20}d[uü][sş]")
_LEAVE_MARKET = re.compile(
    r"(?i)(bırak[ıi]p\s+yeni|yeni\s+([uü]lke|[uü]lkeye|pazar)|mevcut.{0,24}bırak)"
)
_DEFAULT_HOLD = (
    "Kritik metrik yok diye kararı erteleme. Geçici duruş: mevcut tahsisi ve fiyatı koru; "
    "kör kilit veya fiyat hareketi önermem."
)
_DEFAULT_NEXT = (
    "Eksik değişkeni duruştan sonra bir cümlede sor; şimdilik mevcut durumu tut."
)


@dataclass
class BuyerPriority:
    name: str
    band: str  # A | B | C
    reason: str
    country: str = ""
    match_score: float | None = None
    signals: dict[str, str] = field(default_factory=dict)


@dataclass
class CommercialBrief:
    mode: Mode
    human_plan: str
    user_claims: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    critical_questions: list[str] = field(default_factory=list)
    next_action: str = ""
    stance: str = ""
    priorities: list[BuyerPriority] = field(default_factory=list)
    decision_required: bool = False
    company_notes: str = ""
    memory_hits: list[str] = field(default_factory=list)
    available_data: str = ""
    tradeoff: str = ""
    change_mind: str = ""
    provenance_note: str = ""

    def as_prompt(self) -> str:
        lines = [
            "TİCARİ ZEKÂ BRİFİ (kopyalama, yorumla):",
            f"Mod: {self.mode}",
        ]
        if self.company_notes:
            lines.append("ŞİRKET PROFİLİ (karara kat, uydurma): " + self.company_notes)
        if self.memory_hits:
            lines.append("BELLEK (yalnızca kayıtlı; yoksa hatırlıyorum deme):")
            lines.extend(f"- {item}" for item in self.memory_hits[:4])
        else:
            lines.append("BELLEK BOŞ: hatırlıyorum / geçen sefer deme.")
        if self.user_claims:
            lines.append("KULLANICI BİLGİSİ (fact değil):")
            lines.extend(f"- {c}" for c in self.user_claims)
        if self.priorities:
            lines.append("Alıcı önceliği (eşleşme skoru ≠ ticari uygunluk):")
            for item in self.priorities[:8]:
                score = (
                    f" eşleşme={item.match_score:.2f}"
                    if item.match_score is not None
                    else ""
                )
                lines.append(f"- {item.band}: {item.name}{score} — {item.reason}")
        if self.stance:
            lines.append("ÖNCE DURUŞ (sorulardan önce, kesin rakam yok): " + self.stance)
        if self.tradeoff:
            lines.append("Trade-off: " + self.tradeoff)
        if self.change_mind:
            lines.append("Kararı değiştiren veri: " + self.change_mind)
        if self.next_action:
            lines.append("Sonraki adım: " + self.next_action)
        if self.missing:
            lines.append("Eksik kritik veri (duruştan sonra): " + "; ".join(self.missing[:3]))
        if self.critical_questions:
            lines.append("Duruştan SONRA gerekirse en fazla 1–3 soru (soru duruşun yerine geçmez):")
            lines.extend(f"- {q}" for q in self.critical_questions[:3])
        if self.provenance_note:
            lines.append("PROVENANCE: " + self.provenance_note)
        lines.append(
            "Bu iç yapıyı başlık başlık dökme. Doğal konuş. "
            "Kullanıcı iddiasını OBJECTIVE FACT yapma. "
            "Eşleşme skorunu kazanma ihtimali gibi sunma. "
            "«İkisi de olabilir» ile yetinme. Uydurma rakam yok. "
            "Cevabı soruyla açma. Önce duruş, sonra gerekirse soru. "
            "Aynı «Ben şu aşamada X'i bırakmazdım» kalıbını ezberleme."
        )
        return "\n".join(lines)


def extract_user_claims(question: str) -> list[str]:
    text = (question or "").strip()
    claims: list[str] = []
    if _PRICE_PRESSURE.search(text) and re.search(
        r"(?i)(fiyat\s*bask[ıi]|ucuz|price\s*pressure)", text
    ):
        claims.append(
            "Kullanıcı, mevcut pazardaki müşterilerin fiyat baskısı yarattığını söylüyor. "
            "Bu henüz ölçülmüş marj verisi değil."
        )
    found = _LEAD_COUNT.search(text)
    if found:
        claims.append(
            f"Kullanıcı {found.group(1)} potansiyel müşteri/kayıt yakaladığını belirtiyor. "
            "Kaynak doğrulanmadıysa SOURCE_USER."
        )
    if _COMPETITOR_CHEAP.search(text):
        pct = re.search(r"%\s*(\d+(?:[.,]\d+)?)|yüzde\s*(\d+(?:[.,]\d+)?)", text)
        amount = ""
        if pct:
            amount = f" (%{pct.group(1) or pct.group(2)})"
        claims.append(
            "Kullanıcı rakibin daha düşük fiyat verdiğini söylüyor"
            f"{amount}. Bu henüz doğrulanmış teklif değil."
        )
    if re.search(r"(?i)almanya.{0,24}k[oö]t[uü]|almanya.{0,16}vazge[cç]", text):
        claims.append("Kullanıcı Almanya'yı zayıf görüyor. Bunu otomatik gerçek sayma.")
    if _EXISTING_STABLE.search(text):
        claims.append(
            "Kullanıcı mevcut müşteri siparişi ile yeni müşteri akışını ayırarak verdi. "
            "Bu kullanıcı tablosu; CRM kırılımı değil. Aynı ayrımı tekrar sorma."
        )
    return claims


def _field(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, dict) and item.get(name) not in (None, ""):
            return item.get(name)
        value = getattr(item, name, None)
        if value not in (None, ""):
            return value
    return None


def prioritize_matches(matches: list[Any] | None) -> list[BuyerPriority]:
    """A/B/C: karttaki sinyal. similarity ≠ kazanma ihtimali. Eksik veri negatif değil."""
    ranked: list[BuyerPriority] = []
    for item in matches or []:
        name = str(
            _field(item, "organization_name", "product_name") or "kayıt"
        )
        country = str(
            _field(item, "destination_country", "origin_country") or ""
        )
        product = str(_field(item, "product_name", "hs_code") or "")
        channel = str(_field(item, "channel", "buyer_type", "customer_type") or "")
        sim = _field(item, "similarity")
        try:
            score = float(sim) if sim is not None else None
        except (TypeError, ValueError):
            score = None
        has_contact = bool(
            _field(item, "has_contact", "contact_email", "website", "email")
        )
        match_fit = (
            "high" if score is not None and score >= 0.72 else
            "mid" if score is not None and score >= 0.45 else
            "low" if score is not None else "unknown"
        )
        country_fit = "present" if country else "unknown"
        product_fit = "present" if product else ("high" if match_fit == "high" else "unknown")
        channel_fit = "present" if channel else "unknown"
        contact = "present" if has_contact else "verify"
        filled = sum(
            1
            for value in (match_fit != "unknown", bool(country), bool(product), has_contact)
            if value
        )
        completeness = "high" if filled >= 3 else "mid" if filled >= 2 else "low"
        signals = {
            "MATCH_SCORE": match_fit,
            "COUNTRY_FIT": country_fit,
            "PRODUCT_FIT": product_fit,
            "CHANNEL_FIT": channel_fit,
            "CONTACT_AVAILABILITY": contact,
            "DATA_COMPLETENESS": completeness,
        }
        if match_fit == "high" and (country or product or has_contact):
            band = "A"
            reason = "Ürün eşleşmesi yüksek ve pazar/ürün sinyali var."
            if not has_contact:
                reason += " Temas bilgisi doğrulanmalı; eksik iletişim olumsuz müşteri hükmü değildir."
            else:
                reason += " Hacim hâlâ doğrulanmadı."
        elif match_fit in ("high", "mid"):
            band = "B"
            reason = "Ürün eşleşmesi var; şirket, kanal veya hacim doğrulaması eksik. Araştır."
            if not has_contact:
                reason += " Temas bilgisi doğrulanmalı."
        else:
            band = "C"
            reason = "Yüzeysel eşleşme. Hedef profile zayıf uyum; düşük öncelik."
        ranked.append(
            BuyerPriority(
                name=name,
                band=band,
                reason=reason,
                country=country,
                match_score=score,
                signals=signals,
            )
        )
    order = {"A": 0, "B": 1, "C": 2}
    ranked.sort(key=lambda row: (order.get(row.band, 9), -(row.match_score or 0)))
    return ranked


def cluster_matches(matches: list[Any] | None) -> str:
    if not matches:
        return ""
    countries: Counter[str] = Counter()
    for item in matches:
        country = (
            getattr(item, "destination_country", None)
            or getattr(item, "origin_country", None)
            or (item.get("destination_country") if isinstance(item, dict) else None)
            or "?"
        )
        countries[str(country).upper()] += 1
    bits = [f"{code}:{n}" for code, n in countries.most_common(5)]
    return "Kayıtlar ülke kırılımı: " + ", ".join(bits) if bits else ""


def classify_commercial_mode(question: str) -> Mode:
    text = question or ""
    if _HIGH_VOL_LOW_MARGIN.search(text) or _LOW_VOL_HIGH_MARGIN.search(text):
        return "decision"
    if _BEST_BUYER.search(text) or (
        is_decision_question(text) and re.search(r"(?i)(al[iı]c[iı]|buyer|m[uü][sş]teri)", text)
        and re.search(r"(?i)(hangisi|[oö]ncelik|en\s*iyi)", text)
    ) or re.search(
        r"(?i)(matching\s*score|e[sş]le[sş]me\s*skor).{0,48}(gideyim|mi\s*gide|hangisindeyse)",
        text,
    ):
        return "buyer_priority"
    if is_diagnostic_request(text) or _NEW_VS_EXISTING.search(text) or _EXISTING_STABLE.search(text):
        return "diagnostic"
    if _COMPETITOR_CHEAP.search(text) or (
        is_competitor_research(text) and not is_stat_challenge(text)
    ):
        return "decision"
    if (
        is_decision_question(text)
        or is_market_switch(text)
        or _PRICE_CUT.search(text)
        or _PRICE_HIKE.search(text)
        or _HIGH_VOL_LOW_MARGIN.search(text)
        or _LOW_VOL_HIGH_MARGIN.search(text)
        or _FRANCE_EMPTY.search(text)
        or _MOQ_CUT.search(text)
        or _SUPPLIER_PICK.search(text)
        or _INCOTERMS.search(text)
        or _CREDIT_ACCEPT.search(text)
        or _CONVERSION_DROP.search(text)
        or _REPEAT_ORDERS.search(text)
        or _MARGIN_DROP.search(text)
        or _LEAVE_MARKET.search(text)
        or _PRICE_PRESSURE.search(text)
        or is_planning_ask(text)
    ):
        return "decision"
    if re.search(r"(?i)(e[sş]le[sş]me\s*skor|matching\s*score|%\s*9\d)", text):
        return "matching"
    return "general"


_FALSE_CONF = re.compile(
    r"(?i)("
    r"\bkesinlikle\s+(daha\s+)?(k[aâ]rl[ıi]|iyi)\b|"
    r"\bkesin\s+daha\s+(k[aâ]rl[ıi]|iyi)\b|"
    r"\b(fransa|almanya)\s+kesin\b|"
    r"\bbu\s+m[uü][sş]teri\s+kesin\s+al"
    r")"
)


def is_false_confidence(text: str) -> bool:
    """Veri yokken mutlak ticari iddia."""
    return bool(_FALSE_CONF.search(text or ""))


def human_intent_detail(intent: str, question: str) -> str:
    """UI niyet satırı: skor, ajan adı, LLM yok."""
    mode = classify_commercial_mode(question)
    if mode == "decision":
        return "Bunu bir ticari karar sorusu olarak okuyorum. Skor tablosu yok; duruş ve sonraki adım vereceğim."
    if mode == "diagnostic":
        return "Önce düşüşün kaynağını ayıracağım, sonra tek bir net adım önereceğim."
    if mode == "buyer_priority":
        return "Alıcı kayıtlarını A / B / C ticari önceliğe çevireceğim. Eşleşme skoru kazanma ihtimali değil."
    if mode == "matching":
        return "Ürün eşleşmesini ticari uygunluktan ayırarak yorumlayacağım."
    mapping = {
        "chat": "Kısa sohbet olarak yanıtlıyorum.",
        "intake": "Önce ürün ve hedefi netleştiriyorum.",
        "sector_chat": "Sektör bağlamında bakıyorum.",
        "trade_advisor": "Durumu ticari olarak değerlendirip somut bir sonraki adım vereceğim.",
        "buyer_finder": "Alıcı kayıtlarını tarayıp ticari öncelik sırasına koyacağım.",
        "supplier_finder": "Tedarikçi kayıtlarını tarayıp önceliklendireceğim.",
        "product_matching": "Ürün eşleşmesini ticari uygunlukla karıştırmadan yorumlayacağım.",
    }
    return mapping.get(intent, "Ticari olarak okuyup net bir sonraki adım vereceğim.")


def _questions_for(
    mode: Mode,
    question: str,
    session: Any,
    *,
    has_stance: bool = False,
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    questions: list[str] = []
    text = question or ""
    if mode == "decision" and _MARGIN_DROP.search(text) and not re.search(
        r"(?i)fiyat\s*bask", text
    ):
        missing.append("hacim vs birim fiyat kırılımı")
    elif mode == "decision" and _PRICE_PRESSURE.search(text):
        missing.append("teklif fiyatı vs minimum kabul marjı")
        questions.append(
            "Almanya'daki bu müşterilerin istediği fiyat, senin minimum kabul edeceğin marjın altında mı?"
        )
        questions.append("Önceliğin yüksek hacim mi, yoksa daha yüksek marj mı?")
    elif mode == "decision" and _PRICE_CUT.search(text):
        missing.append("mevcut marj ve rakip fiyatı")
        questions.append("Fiyatı düşürmeden önce mevcut brüt marjın yüzde kaç?")
    elif mode == "decision" and _PRICE_HIKE.search(text):
        missing.append("mevcut marj eşiği ve rekabet")
    elif mode == "decision" and _MOQ_CUT.search(text):
        missing.append("kayıtlı MOQ ve birim maliyet")
    elif mode == "decision" and _SUPPLIER_PICK.search(text):
        missing.append("fiyat, MOQ, teslimat, münhasırlık")
    elif mode == "decision" and _INCOTERMS.search(text):
        missing.append("lojistik maliyet ve sigorta")
    elif mode == "decision" and _CREDIT_ACCEPT.search(text):
        missing.append("kredi/tahsilat kaydı ve teslim takvimi")
    elif mode == "decision" and _CONVERSION_DROP.search(text):
        missing.append("kayıp nedeni: fiyat vs hedef vs takip")
    elif mode == "diagnostic":
        if _EXISTING_STABLE.search(text):
            pass
        else:
            missing.append("düşüşün kaynağı: mevcut vs yeni müşteri")
            questions.append(
                "Son iki aydaki düşüş mevcut müşterilerden mi geliyor, yoksa yeni müşteri girişinin durmasından mı?"
            )
    elif mode == "buyer_priority" and not re.search(r"\d", text):
        missing.append("hangi kayıtların kartta olduğu")
    elif mode == "decision" and not (session and getattr(session, "product", None)):
        missing.append("ürün ve hedef kanal")
        # Duruş varken ürün sorusu duruşun yerini almasın.
        if not has_stance:
            questions.append("Hangi ürün ve müşteri segmenti için karar veriyoruz?")
    return missing[:3], questions[:3]


def _company_line(session: Any, pack: Any = None) -> str:
    if pack is not None and getattr(pack, "company", None):
        return str(pack.company).strip()
    if session is None:
        return ""
    bits: list[str] = []
    product = getattr(session, "product", None)
    capacity = getattr(session, "capacity", None)
    stock = getattr(session, "stock", None)
    market = getattr(session, "market", None)
    if product:
        bits.append(f"Ürün: {product}")
    if capacity:
        bits.append(f"Kapasite: {capacity}")
    if stock:
        bits.append(f"MOQ/stok: {stock}")
    if market:
        bits.append(f"Hedef pazar: {market}")
    return "; ".join(bits)


def _memory_hits(session: Any, pack: Any = None) -> list[str]:
    if pack is not None and getattr(pack, "memories", None):
        return [str(item) for item in pack.memories if item][:6]
    if session is not None and getattr(session, "memories", None):
        return [str(item) for item in session.memories if item][:6]
    return []


def _stance_and_action(
    question: str,
    session: Any,
    leads: int | None,
    *,
    mode: Mode = "general",
) -> tuple[str, str]:
    text = question or ""
    product = getattr(session, "product", None) if session is not None else None
    capacity = getattr(session, "capacity", None) if session is not None else None
    tight = bool(
        _CAPACITY_TIGHT.search(text)
        or _CAPACITY_TIGHT.search(str(capacity or ""))
        or _CAPACITY_TIGHT.search(_company_line(session))
    )
    if _HIGH_VOL_LOW_MARGIN.search(text):
        return (
            "Hacmi tek başına öncelik yapma. Düşük marjlı yüksek hacim operasyonu ve nakit bağlar.",
            "Önce birim marjı ve kapasiteyi yan yana koy; hacmi ancak marj nefes alıyorsa konuş.",
        )
    if _LOW_VOL_HIGH_MARGIN.search(text):
        extra = " Kapasite kısıtlıysa bu segment daha uyumlu." if tight else ""
        return (
            "Az hacim + yüksek marj, özellikle kapasite sınırlıysa daha sağlıklı bir test." + extra,
            "Bu iki müşteriyi marj ve tekrar sipariş potansiyeline göre sırala, yüksek hacme acele etme.",
        )
    if is_market_switch(text) and (leads or 0) >= 2:
        cap = (
            " Kapasite dar ise Almanya'daki mevcut lead'i yüksek hacimli tek hesaba kilitleme."
            if tight
            else ""
        )
        return (
            "Mevcut pazarı (Almanya) henüz bırakma. Elde lead varsa önce fiyat/marjı ölç; "
            "Fransa ikinci test pazarı olabilir." + cap,
            "Önce mevcut müşterileri fiyat, hacim ve marj açısından karşılaştır; "
            "üçünde de marj kabul edilemezse Fransa'yı ikinci test pazarı aç.",
        )
    if _FRANCE_EMPTY.search(text):
        return (
            "Pazar büyüklüğü tek başına giriş gerekçesi değil. Fransa'ya kör giriş önermem; "
            "mevcut fırsatı koru, lead yokken maliyet riski var.",
            "Fransa iddiasını kaynakla doğrulamadan mevcut fırsatı bırakma.",
        )
    if re.search(r"(?i)(fiyat\s*bask[ıi]|ucuz|price\s*pressure)", text) and not _PRICE_CUT.search(
        text
    ) and not _PRICE_HIKE.search(text):
        return (
            "Fiyat baskısı senin gözlemin; henüz ölçülmüş marj değil. "
            "Mevcut lead'i koru, fiyatı hemen düşürme.",
            "Önce teklifin minimum kabul marjının altında olup olmadığını ölç.",
        )
    if _PRICE_HIKE.search(text):
        return (
            "Marj yokken fiyat artırmayı önermem; kritik veri gelene kadar fiyatı sabit tut.",
            "Önce minimum kabul marjını netleştir, sonra artışı test et.",
        )
    if _PRICE_CUT.search(text):
        return (
            "Fiyatı hemen düşürme. Önce marj ve segment uyumunu ölç.",
            "Mevcut teklif marjını ve rakibin gerçek fiyatını netleştir, sonra indirim kararı ver.",
        )
    if _MOQ_CUT.search(text):
        return (
            "MOQ'u hemen düşürme. Maliyet netleşene kadar mevcut şartı koru.",
            "Birim maliyet ve kapasite eşiğini netleştirmeden MOQ indirme.",
        )
    if _SUPPLIER_PICK.search(text):
        return (
            "Doğrulanmış fiyat, MOQ veya teslimat yokken tekine kilitlenmeyi önermem; "
            "mevcut ilişkiyi koru, adayları paralel tut.",
            "Kapsama, MOQ ve teslimatı yan yana koy; kör seçim yok.",
        )
    if _INCOTERMS.search(text):
        return (
            "Lojistik maliyet doğrulanmadan CIF'e geçmeyi önermem; şimdilik FOB şartını seç.",
            "Navlun ve sigorta netleşince CIF'i ikinci test olarak konuş.",
        )
    if _CREDIT_ACCEPT.search(text):
        if re.search(r"(?i)(ödeme|kredi|gecik|hay[ıi]r)", text):
            return (
                "Kredi kaydı yokken büyük vadeli siparişi kabul etmem. "
                "Geçici duruş: peşin veya teminat netleşene kadar beklet.",
                "Tahsilat eşiğini netleştir; kör evet yok.",
            )
        return (
            "Kapasite ve teslim takvimi yokken büyük siparişi kabul etmem. "
            "Geçici duruş: mevcut tahsisi koru, sıkışık tarihi ikinci sıraya al.",
            "Teslim penceresi ve kapasiteyi netleştirmeden kabul etme.",
        )
    if _CONVERSION_DROP.search(text):
        return (
            "Dönüşüm düşünce fiyatı hemen düşürme. Geçici duruş: mevcut teklifi koru; "
            "kayıp nedenini fiyat varsaymadan ayır.",
            "Fiyat mı hedef mi takip mi olduğunu netleştir; funnel uydurma.",
        )
    if _REPEAT_ORDERS.search(text):
        return (
            "Reklam veya fiyatı hemen düşürme. Geçici duruş: mevcut ilk müşteriyi koru; "
            "tekrar sipariş deliğini teslimat ve takipte ara.",
            "İlk sipariş sonrası temas ve teslimat kalitesini kontrol et.",
        )
    if _MARGIN_DROP.search(text) and not _HIGH_VOL_LOW_MARGIN.search(text):
        return (
            "Marj düşünce fiyatı hemen düşürme. Geçici duruş: mevcut fiyatı koru; "
            "hacim mi birim fiyat mı eridiğini ayır.",
            "Kırılım gelmeden indirim veya hacim kovalama yok.",
        )
    if _LEAVE_MARKET.search(text):
        return (
            "Mevcut ilişkiyi bırakıp yeni ülkeye geçmeyi önermem; kaybı teşhis et, "
            "yeni pazarı ikinci test yap.",
            "Önce kesilen siparişin nedenini netleştir; pazar değişimi ikinci.",
        )
    if _COMPETITOR_CHEAP.search(text):
        return (
            "Rakip fiyatı senin aktardığın tablo; henüz doğrulanmış liste değil. "
            "Fiyatı hemen eşitlemeyi önermem.",
            "Önce kendi marjını ve rakibin gerçekten ödediği fiyatı netleştir.",
        )
    if (is_decision_question(text) or is_market_switch(text)) and re.search(
        r"(?i)almanya|fransa", text
    ):
        extra = f" Kayıtlı ürün: {product}." if product else ""
        if tight:
            extra += " Kapasite kısıtı yüksek hacimli pazara körlemesine gitmeyi sınırlar."
        return (
            "Veri sınırlı. Almanya'daki mevcut lead sinyalini koru; Fransa'yı ikinci test olarak beklet. "
            f"Ürün ve kanal netleşmeden pazar kilidi önermem.{extra}",
            "Önce ürün, kapasite ve hedef kanalı netleştir; Almanya/Fransa seçimini kilitleme.",
        )
    if mode == "buyer_priority" or _BEST_BUYER.search(text):
        return (
            "Eşleşme skoruna bakarak A/B/C öncelik ver; skoru kazanma ihtimali sayma.",
            "Önce A grubuna kısa teklif ve hacim sor.",
        )
    if is_planning_ask(text):
        return (
            "Bugün yeni pazar kilidi önermem; mevcut yüksek sinyalli kayıtlara hemen temas öneririm.",
            "A-sinyalli kartlara kısa teklif ve hacim sor; C'yi beklet.",
        )
    if is_diagnostic_request(text) or _EXISTING_STABLE.search(text):
        if _EXISTING_STABLE.search(text):
            return (
                "Mevcut müşteriyi koru; yeni kazanımı ayrı test yap. Reklam veya fiyat dayatma.",
                "Önce mevcut müşteriyi kaybetmeyecek koşulları koru; yeni girişte kanal tıkanıklığını kontrol et.",
            )
        return (
            "Satış düşüşünü tek nedene bağlama. Reklam veya fiyatı hemen düşürme; "
            "kırılım netleşene kadar mevcut müşteriyi koru.",
            "Önce mevcut vs yeni müşteri kırılımını netleştir.",
        )
    if tight:
        return (
            "Kayıtlı kapasite kısıtlı. Yüksek hacimli genel perakendeden önce yüksek marjlı dar segmenti test et.",
            "Kapasiteyi zorlamayan bir sonraki ticari adımı seç.",
        )
    if product:
        return (
            f"Kayıtlı ürün ({product}) için geçici duruş: mevcut tahsisi ve fiyatı koru; "
            "kör kilit önermem.",
            "Öneri, kayıtlı ürün ve kapasiteyle uyumlu bir sonraki adıma bağlansın.",
        )
    if mode in ("decision", "diagnostic", "buyer_priority", "matching") or is_decision_question(
        text
    ) or is_diagnostic_request(text) or is_planning_ask(text):
        return _DEFAULT_HOLD, _DEFAULT_NEXT
    return "", "Somut bir sonraki ticari adım söyle."


def build_commercial_brief(
    question: str,
    *,
    session: Any = None,
    matches: list[Any] | None = None,
    tools: list[Any] | None = None,
    pack: Any = None,
) -> CommercialBrief:
    mode = classify_commercial_mode(question)
    claims = extract_user_claims(question)
    leads = None
    hit = _LEAD_COUNT.search(question or "")
    if hit:
        leads = int(hit.group(1))
    if matches:
        leads = leads or len(matches)
    stance, action = _stance_and_action(question, session, leads, mode=mode)
    missing, questions = _questions_for(
        mode, question, session, has_stance=bool((stance or "").strip())
    )
    priorities = prioritize_matches(matches) if matches else []
    cluster = cluster_matches(matches)
    decision = mode == "decision" or is_decision_question(question)
    human = _human_plan(mode, question, session, stance)
    if cluster:
        human = f"{human} {cluster}"
    available, tradeoff, change, prov = _decision_frame(
        question, session, leads, stance, pack
    )
    return CommercialBrief(
        mode=mode,
        human_plan=human,
        user_claims=claims,
        missing=missing,
        critical_questions=questions,
        next_action=action,
        stance=stance,
        priorities=priorities,
        decision_required=decision,
        company_notes=_company_line(session, pack),
        memory_hits=_memory_hits(session, pack),
        available_data=available,
        tradeoff=tradeoff,
        change_mind=change,
        provenance_note=prov,
    )


def _decision_frame(
    question: str,
    session: Any,
    leads: int | None,
    stance: str,
    pack: Any,
) -> tuple[str, str, str, str]:
    bits: list[str] = []
    if leads:
        bits.append(f"{leads} lead/kayıt (kullanıcı veya kart)")
    company = _company_line(session, pack)
    if company:
        bits.append(company)
    mem = _memory_hits(session, pack)
    if mem:
        bits.append("kayıtlı bellek var")
    available = "; ".join(bits) if bits else "doğrulanmış pazar istatistiği yok"
    text = question or ""
    if is_market_switch(text) and (leads or 0) >= 2:
        tradeoff = "Almanya hazır fırsat; fiyat baskısı (iddia) marj riski. Fransa ikinci test, ilk terk değil."
        change = "Üç lead'in tamamında kabul edilemez marj çıkarsa Fransa'yı öne alırım."
    elif _HIGH_VOL_LOW_MARGIN.search(text):
        tradeoff = "Hacim ciro büyütür, düşük marj nakit ve kapasiteyi yer."
        change = "Birim marj hedef eşiğin üstüne çıkarsa hacmi konuşurum."
    elif _LOW_VOL_HIGH_MARGIN.search(text):
        tradeoff = "Düşük hacim ölçek vermez; yüksek marj kapasiteyi korur."
        change = "Kapasite boşa çıkarsa ve tekrar sipariş durursa hacimli hesabı test ederim."
    elif _PRICE_CUT.search(text) or _COMPETITOR_CHEAP.search(text):
        tradeoff = "İndirim siparişi kurtarabilir, marjı yakabilir."
        change = "Rakip fiyatı kaynakla doğrulanır ve marj hâlâ nefes alıyorsa sınırlı indirimi konuşurum."
    elif is_diagnostic_request(text) or _EXISTING_STABLE.search(text):
        if _EXISTING_STABLE.search(text):
            tradeoff = "Mevcutı korumak nakit ve ilişkiyi tutar; yeni kazanım ayrı test ister."
            change = "Yeni tarafta doğrulanmış hacim ve kabul edilebilir marj olursa önceliği değiştiririm."
        else:
            tradeoff = "Yanlış teşhis yanlış aksiyon doğurur (reklam vs elde tutma)."
            change = "Kırılım netleşince (mevcut vs yeni) tek nedene inerim."
    else:
        tradeoff = stance or "veri sınırlı"
        change = "Kritik eksik veri (marj, kanal, kapasite) gelirse duruşu güncellerim."
    prov = "SOURCE_USER" if extract_user_claims(question) else "INFERENCE"
    if company:
        prov += " + SOURCE_MEMORY/PROFILE"
    return available, tradeoff, change, prov


def _human_plan(mode: Mode, question: str, session: Any, stance: str) -> str:
    product = getattr(session, "product", None) if session is not None else None
    market = getattr(session, "market", None) if session is not None else None
    if mode == "decision" and is_market_switch(question or ""):
        return (
            "Önce eldeki pazardaki fırsatları fiyat baskısı açısından değerlendireceğim, "
            "ardından alternatif pazarın gerçekten ikinci test olup olmadığına bakacağım."
        )
    if mode == "diagnostic":
        return (
            "Satış düşüşünün kaynağını ayıracağım: mevcut müşteri mi, yeni müşteri mi, "
            "yoksa fiyat ve marj mı. Sonra tek bir net sonraki adım önereceğim."
        )
    if mode == "buyer_priority":
        return (
            "Kayıtları ticari önceliğe çevireceğim. Eşleşme skoru kazanma ihtimali değil; "
            "hemen temas, araştır veya düşük öncelik diyeceğim."
        )
    if mode == "matching":
        return "Eşleşme skorunu ticari uygunluktan ayırıp ürün-pazar uyumunu yorumlayacağım."
    bits = ["Durumu ticari olarak okuyup net bir öneri ve sonraki adım vereceğim."]
    if product:
        bits.append(f"Ürün: {product}.")
    if market:
        bits.append(f"Pazar: {market}.")
    if stance:
        bits.append(stance)
    return " ".join(bits)


def commercial_fallback_reply(brief: CommercialBrief, question: str) -> str | None:
    """LLM yok/reddedilince kural tabanlı CMO cevabı. Ezber cümle değil, aynı mantık."""
    text = question or ""
    if brief.mode == "decision" and is_market_switch(text) and _PRICE_PRESSURE.search(text):
        n = ""
        hit = _LEAD_COUNT.search(text)
        if hit:
            n = hit.group(1)
        who = f"{n} potansiyel müşteri" if n else "eldeki potansiyel müşteriler"
        q1 = brief.critical_questions[0] if brief.critical_questions else (
            "Bu müşterilerin istediği fiyat minimum marjının altında mı?"
        )
        return (
            f"Ben şu aşamada Almanya'yı bırakmazdım. Çünkü {who} zaten masada. "
            "Fiyat baskısı senin gözlemin; henüz ölçülmüş marj verisi değil. "
            "Önce bu müşterilerde fiyatın gerçekten kabul edilemez bir marja inip inmediğini ölçmek daha doğru. "
            "Üçünde de aynı tablo çıkarsa Fransa'yı ikinci test pazarı olarak açabiliriz. "
            f"{q1} "
            + (f"Kararı değiştiren veri: {brief.change_mind} " if brief.change_mind else "")
            + "İstersen şimdi bu müşterileri fiyat, ürün uyumu ve potansiyel sipariş açısından önceliklendirelim."
        )
    if brief.mode == "decision" and _HIGH_VOL_LOW_MARGIN.search(text):
        return (
            "Ben bu hacmi bugün kovalamazdım. Yüksek sipariş düşük marjla nakit ve kapasiteyi bağlar. "
            "Kararı değiştiren veri: birim marjın kabul eşiğinin üstünde kalması. "
            "Önce marj kırılımını masaya koyalım."
        )
    if brief.mode == "decision" and _LOW_VOL_HIGH_MARGIN.search(text):
        return (
            "Ben olsam bu yüksek marjlı, az hacimli hesapları önce test ederdim. "
            "Ölçek vermez ama marjı ve kapasiteyi korur. "
            "Sonraki adım: ikisini tekrar sipariş ve ödeme disiplinine göre sırala."
        )
    if brief.mode == "decision" and _FRANCE_EMPTY.search(text):
        return (
            "Pazar büyük iddiası henüz kaynaklı veri değil. Fransa'da lead yokken oraya yönelmek giriş maliyeti taşır. "
            "Ben olsam önce eldeki sinyali bırakmaz, Fransa büyüklüğünü kaynakla doğrulatırdım. "
            "Kararı değiştiren veri: doğrulanmış talep ve en az birkaç nitelikli lead."
        )
    if brief.mode == "decision" and _COMPETITOR_CHEAP.search(text):
        return (
            "Rakibin yüzde yirmi düşük sattığı senin aktardığın tablo; ben bunu henüz fact yapmam. "
            "Fiyatı eşitlemek marjı yakabilir. Önce kendi brüt marjın ve rakibin gerçek teklifi. "
            "Kararı değiştiren veri: doğrulanmış rakip fiyatı ve senin eşiğinin üstünde kalan marj."
        )
    if is_competitor_research(text) and not is_stat_challenge(text):
        return (
            "Rakip hareketini senin aktardığın kadarıyla görüyorum; güncel rakip fiyat veya pazar payı uydurmam. "
            "Fiyatı hemen eşitlemem. Kararı değiştiren veri: doğrulanmış rakip teklifi ve senin marj eşiğin. "
            "Önce kendi teklifini marj kırılımıyla masaya koy."
        )
    if brief.mode == "decision" and _PRICE_CUT.search(text):
        return (
            "Ben fiyatı şu anda düşürmezdim. Önce mevcut brüt marjın hâlâ nefes alıp almadığına bakardım. "
            "Marj zaten inciyse indirim ciroyu değil zararı büyütür. "
            "Kararı değiştirecek veri: bugünkü marj ve rakibin gerçekten ödediği fiyat. "
            "Sonraki adım: mevcut teklifi marj kırılımıyla masaya koy, sonra indirimi konuş."
        )
    if brief.mode == "diagnostic":
        if re.search(r"(?i)yeni\s*m[uü][sş]teri\s*(gelmiyor|yok|durdu)", text):
            return (
                "Mevcut müşteri sipariş veriyorsa asıl delik kazanım tarafı. "
                "Reklam bütçesini artırmadan önce yeni girişin neden durduğunu ayıralım. "
                "Elimde CRM yok; funnel uydurmam. Sonraki adım: kanal ve teklif tıkanıklığını netleştir."
            )
        if re.search(r"(?i)mevcut\s*m[uü][sş]teri.{0,16}ayn[ıi]", text):
            return (
                "Mevcut müşteriler aynıysa düşüş büyük olasılıkla yeni girişten geliyor. "
                "Yine de sipariş sıklığını tek cümlede kapatmam. "
                "Yeni müşteri adedi durdu mu, yoksa ortalama sipariş mi inceldi?"
            )
        q1 = brief.critical_questions[0] if brief.critical_questions else (
            "Düşüş mevcut müşterilerden mi, yeni müşteri girişinin durmasından mı?"
        )
        return (
            "Satış düşüşünü tek cümleyle kapatmam. Önce kaynağı ayıralım. "
            f"{q1} "
            "Bu netleşmeden fiyat mı, kanal mı, yoksa talep mi diye atlamam."
        )
    if brief.mode == "buyer_priority" and brief.priorities:
        top = [p for p in brief.priorities if p.band == "A"][:3]
        mid = [p for p in brief.priorities if p.band == "B"][:3]
        bits = [
            "Eşleşme skoru ticari uygunluk değildir; yüksek skor «kesin müşteri olur» demek değil. "
            "Karttaki veriye göre öncelik:"
        ]
        if top:
            bits.append("Hemen temas: " + ", ".join(p.name for p in top) + ".")
        if mid:
            bits.append("Araştır: " + ", ".join(p.name for p in mid) + ".")
        bits.append(brief.next_action or "Önce A grubuna kısa teklif ve hacim sor.")
        return " ".join(bits)
    if brief.mode == "buyer_priority":
        return (
            "Eşleşme skoru ticari öncelik değildir; en yüksek skora körlemesine gitmem. "
            "Kartta A/B/C sinyali yoksa önce ülke, kanal ve temas verisini tamamla. "
            "Kazanma ihtimali diye okuma."
        )
    if brief.mode == "matching":
        return (
            "Yüksek eşleşme skoru yalnızca ürün benzerliğini gösterir, ticari uygunluğu değil. "
            "Müşteri olur ihtimali diye okuma. Önce kanal, hacim ve fiyat uyumuna bakardım."
        )
    if brief.decision_required and brief.stance:
        ask = brief.critical_questions[0] if brief.critical_questions else ""
        return (
            f"{brief.stance} {brief.next_action} {ask}"
        ).strip()
    return None


def is_weak_decision(text: str) -> bool:
    return bool(_WEAK_BOTH.search(text or "") or _HESITATION.search(text or ""))


def is_reasoning_task(question: str) -> bool:
    """Karar / teşhis / öncelik / rakip: reasoning model. Selamlama değil."""
    mode = classify_commercial_mode(question)
    if mode in ("decision", "diagnostic", "buyer_priority", "matching"):
        return True
    return bool(
        is_decision_question(question)
        or is_diagnostic_request(question)
        or is_competitor_research(question)
    )


def treats_score_as_win(text: str) -> bool:
    body = text or ""
    if not _SCORE_AS_WIN.search(body):
        return False
    if re.search(
        r"(?i)("
        r"kazanma ihtimali değil|"
        r"ticari uygunluk değildir|"
        r"sadece (temas |ürün )?ihtimal|"
        r"ihtimalini göstermez|"
        r"müşteri olur ihtimali diye okuma|"
        r"skoru kazanma"
        r")",
        body,
    ):
        return False
    return True
