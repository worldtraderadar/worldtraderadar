"""Danışmanlık oturumu: geçmiş + keşif slotları.

İstemci history kaynak gerçeğidir (uvicorn --reload bellegi siler).
Sunucu bellegi ayni session_id icin hizli slot okumasi saglar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from prompts import (
    INTAKE_REPLY,
    advisor_focus,
    is_draft_request,
    is_general_intake,
    is_language_barrier,
    is_method_question,
    is_strategy_request,
)

AskSlot = Literal["name", "product", "choice"] | None
MAX_TURNS = 24

_CAPACITY_HINT = re.compile(
    r"(?i)(milyon|bin|metre|\bmt\b|adet|ton|kg|kapasite|ayda|ayl[ıi]k|"
    r"\d{1,3}(?:\.\d{3}){1,})"
)
_STOCK_HINT = re.compile(
    r"(?i)(\bstok\b|elimde|eldeki|depoda|\bparti\b|\blot\b|sat[iı]lacak)"
)
_PERIOD_WORD = re.compile(r"(?i)\b(ayda|ayl[ıi]k|günlük|gunluk|yıllık|yillik)\b")
_SCALE_WORD = re.compile(r"(?i)\b(milyon|million|bin|thousand)\b")
_UNIT_WORD = re.compile(r"(?i)\b(metre|mt|adet|ton|kg|m)\b")
_NUM_DOT_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")
_NUM_SPACE_THOUSANDS = re.compile(r"\d{1,3}(?: \d{3})+")
_NUM_DECIMAL = re.compile(r"\d+[.,]\d+")
_NUM_INT = re.compile(r"\d+")
_CONNECT_PICK = re.compile(
    r"(?i)(ba[gğ]la|üretici|uretici|avrupa|haz[ıi]r\s*giyim|"
    r"al[ıi]c[ıi]|ilk[iı]|birinci|tedarik)"
)
_ANALYZE_PICK = re.compile(
    r"(?i)(rakip|analiz|almanya|pazar\s*analiz|ikinci)"
)

_PRODUCTISH = re.compile(
    r"(?i)(etiket|dokuma|kuma[sş]|iplik|tekstil|makine|mobilya|"
    r"g[iı]da|deri|ayakkab[iı]|[uü]r[uü]n|iplik|seramik|zeytin|"
    r"f[iı]nd[iı]k|haz[iı]r\s*giyim|ambalaj|plastik|metal|"
    r"çelik|celik|\bsteel\b|demir|"
    r"oyuncak|filament|\bpla\b|\bpetg\b|yaz[iı]c[iı]|maket|"
    r"3[\s\-]?d|bask[iı])"
)
_DIGIT = re.compile(r"\d")


def _fold(text: str) -> str:
    table = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iissgguuoooc")
    return (text or "").casefold().translate(table)


def _short_label(text: str, limit: int = 42) -> str:
    label = re.sub(r"\s+", " ", (text or "").strip())
    if not label:
        return "Bu ürün"
    if len(label) > limit:
        return "Bu ürün"
    return label[0].upper() + label[1:] if label[0].islower() else label


def looks_like_product(text: str) -> bool:
    body = (text or "").strip()
    if not body:
        return False
    if _PRODUCTISH.search(body):
        return True
    if _DIGIT.search(body) and len(body.split()) <= 6:
        return True
    return False


def looks_like_capacity(text: str) -> bool:
    return parse_capacity(text) is not None


def looks_like_stock(text: str) -> bool:
    parsed = parse_capacity(text)
    return bool(parsed and parsed.get("kind") == "stock")


_PRODUCT_STOP = {
    "icin",
    "urun",
    "uretim",
    "satisi",
    "satmak",
    "yapmak",
    "hizmet",
    "the",
    "and",
}

_PRINT3D_HINT = re.compile(
    r"(?i)(3[\s\-]?d|uc\s*boyut|filament|\bpla\b|\bpetg\b|\bfdm\b|\bsla\b|"
    r"oyuncak|minyatur|minyatür|maket|yaz[iı]c[iı]|additive)"
)


_STEEL_HINT = re.compile(
    r"(?i)(çelik|celik|\bsteel\b|metal|demir[\s\-]?çelik|hot[\s\-]?roll)"
)


def product_kind(product: str | None) -> str:
    """woven | print3d | steel | generic — sektör karışmasın."""
    low = _fold(product or "")
    if any(word in low for word in ("etiket", "dokuma", "label")):
        return "woven"
    if _PRINT3D_HINT.search(product or ""):
        return "print3d"
    if _STEEL_HINT.search(product or ""):
        return "steel"
    return "generic"


def is_product_switch(old: str | None, new: str | None) -> bool:
    if not old or not new:
        return False
    old_k, new_k = product_kind(old), product_kind(new)
    if old_k != new_k:
        return True
    a, b = _fold(old), _fold(new)
    if a == b or a in b or b in a:
        return False
    ta = {tok for tok in re.findall(r"[a-z0-9]{3,}", a) if tok not in _PRODUCT_STOP}
    tb = {tok for tok in re.findall(r"[a-z0-9]{3,}", b) if tok not in _PRODUCT_STOP}
    return bool(ta and tb) and not (ta & tb)


_UTTERANCE_NOISE = re.compile(
    r"(?i)("
    r"al[iı]c[iı]lar[iı]?\s*bulabilir\s*misin|"
    r"m[uü][sş]teri\s*(arıyorum|ariyorum|bul|laz[iı]m)|"
    r"tedarik[cç]i\s*(bul|ara)|"
    r"firma\s*bul|"
    r"l[uü]tfen|"
    r"\b(misiniz|misin)\b"
    r")"
)
_MARKET_NOISE = re.compile(
    r"(?i)\b("
    r"almanya|italya|[iı]talya|fransa|hollanda|ispanya|ingiltere|"
    r"avrupa|amerika|bel[cç]ika|polonya|avusturya"
    r")(['’]?(ya|ye))?\b"
)
_SLOT_TAIL = re.compile(
    r"(?i)\s+("
    r"\bi[cç]in\b|"
    r"al[iı]c[iı]|"
    r"tedarik|"
    r"m[uü][sş]teri|"
    r"ihracat|"
    r"ithalat|"
    r"liste|"
    r"bulabilir|"
    r"cikar|"
    r"çıkar"
    r").*$"
)
_KNOWN_PRODUCT = re.compile(
    r"(?i)("
    r"dokuma\s+etiket|"
    r"3[\s\-]?d\s+oyuncak(?:\s+bask[iı]s[iı])?|"
    r"metal\s*çelik|"
    r"metal\s*celik|"
    r"demir[\s\-]?çelik|"
    r"hot[\s\-]?rolled\s*steel(?:\s+coils?)?|"
    r"zeytinya[gğ][iı]"
    r")"
)


def strip_query_noise(text: str) -> str:
    """Alıcı/soru kalıbını düşür; ürün adını bırak."""
    body = _UTTERANCE_NOISE.sub(" ", text or "")
    body = _MARKET_NOISE.sub(" ", body)
    body = re.sub(r"[?!.,;:]+", " ", body)
    body = _SLOT_TAIL.sub(" ", body)
    body = re.sub(r"(?i)\bi[cç]in\b", " ", body)
    return re.sub(r"\s+", " ", body).strip(" ,.-")


def extract_product_slot(text: str | None) -> str | None:
    """Sorgudan yalnızca ürün varlığını al; ham cümleyi alma."""
    raw = (text or "").strip()
    if not raw:
        return None
    known = _KNOWN_PRODUCT.search(raw)
    if known:
        return re.sub(r"\s+", " ", known.group(1)).strip()
    body = strip_query_noise(raw)
    if not body:
        return None
    if len(body.split()) > 6:
        return None
    if looks_like_capacity(body) or looks_like_stock(body):
        return None
    if not looks_like_product(body) and not looks_like_product(raw):
        return None
    return body


def clean_product_name(text: str | None) -> str | None:
    """Ham sorgu cümlesini ürün slotuna yazma."""
    return extract_product_slot(text)


def _title_product(label: str) -> str:
    body = re.sub(r"\s+", " ", (label or "").strip())
    if not body:
        return body
    if body[0].islower():
        return body[0].upper() + body[1:]
    return body


def display_product(product: str | None) -> str:
    """Şablona yalnızca ayıklanmış ürün adı. Ham sorgu yok."""
    slot = extract_product_slot(product)
    if not slot:
        return "ürününüz"
    return slot


def action_product_label(product: str | None) -> str:
    slot = extract_product_slot(product)
    if not slot:
        return "ürününüz"
    return _title_product(slot)


def strip_echoed_query(question: str, advice: str) -> str:
    """Yanıtın başına ham kullanıcı cümlesini yapıştırma.

    Kısa ürün adı (metal çelik) cümle değildir; aksiyon kalıbının başı olabilir.
    """
    query = re.sub(r"\s+", " ", (question or "").strip())
    body = (advice or "").lstrip()
    if not query or not body:
        return advice or ""
    slot = extract_product_slot(query)
    if slot and _fold(slot) == _fold(query):
        return body
    if body.casefold().startswith(query.casefold()):
        rest = body[len(query) :].lstrip(" \n\t:-–—")
        if rest:
            return rest
    first, sep, rest = body.partition("\n")
    if sep and first.strip().casefold() == query.casefold() and rest.strip():
        return rest.lstrip()
    return body


def wants_toy_event_notes(product: str | None) -> bool:
    """Spielwarenmesse / TÜV yalnızca oyuncak veya 3D baskıda."""
    return product_kind(product) == "print3d"


def _incoming_product(text: str, stripped: str | None) -> str | None:
    body = (text or "").strip()
    leftover = (stripped or "").strip()
    if looks_like_stock(body) or looks_like_capacity(body):
        candidate = clean_product_name(leftover)
        if (
            candidate
            and looks_like_product(candidate)
            and not looks_like_capacity(candidate)
            and not looks_like_stock(candidate)
        ):
            return candidate
        return None
    if looks_like_product(body) and not is_general_intake(body):
        return clean_product_name(leftover or body) or clean_product_name(body)
    return None


def apply_utterance_slots(session: SessionState, text: str) -> bool:
    """Yeni ürün gelince eski ürün/stok/kapasiteyi sil. Switch olduysa True."""
    product, capacity, stock = split_product_and_volume(text)
    incoming = _incoming_product(text, product)
    switched = bool(incoming and is_product_switch(session.product, incoming))
    if switched and incoming:
        session.product = incoming
        session.capacity = capacity
        session.stock = stock
        session.market = market_from_text(text)
        session.last_advisor_text = None
        session.last_advisor_kind = None
        return True
    if incoming and not session.product:
        session.product = incoming
    if stock:
        session.stock = stock
    if capacity:
        session.capacity = capacity
    found_market = market_from_text(text)
    if found_market:
        session.market = found_market
    return False


def _is_stock_utterance(body: str) -> bool:
    if _STOCK_HINT.search(body or ""):
        return True
    if _PERIOD_WORD.search(body or "") or re.search(r"(?i)kapasite", body or ""):
        return False
    if re.search(r"(?i)\b(metre|mt)\b", body or ""):
        return False
    return bool(re.search(r"(?i)\badet\b", body or ""))


def _parse_amount(raw: str, kind: str) -> float:
    token = raw.strip()
    if kind == "dot_thousands":
        if "," in token:
            whole, frac = token.split(",", 1)
            return float(whole.replace(".", "") + "." + frac)
        return float(token.replace(".", ""))
    if kind == "space_thousands":
        return float(token.replace(" ", ""))
    return float(token.replace(",", "."))


def _find_amount(text: str) -> tuple[str, float, str] | None:
    """(ham sayı, nicelik, tür) — binlik ayraçlı biçim öncelikli."""
    for pattern, kind in (
        (_NUM_DOT_THOUSANDS, "dot_thousands"),
        (_NUM_SPACE_THOUSANDS, "space_thousands"),
        (_NUM_DECIMAL, "decimal"),
        (_NUM_INT, "int"),
    ):
        match = pattern.search(text)
        if match:
            raw = match.group(0)
            return raw, _parse_amount(raw, kind), kind
    return None


def parse_capacity(text: str) -> dict[str, Any] | None:
    """Kullanıcı kapasitesini tam ölçekte çözer; basamak kırpmaz."""
    body = re.sub(r"\s+", " ", (text or "").strip())
    if not body or not re.search(r"\d", body):
        return None
    found = _find_amount(body)
    if found is None:
        return None
    raw_num, amount, kind = found
    scale_match = _SCALE_WORD.search(body)
    unit_match = _UNIT_WORD.search(body)
    period_match = _PERIOD_WORD.search(body)
    if not (scale_match or unit_match or period_match or kind in ("dot_thousands", "space_thousands")):
        return None
    scale = (scale_match.group(1) if scale_match else "").casefold()
    if scale in {"milyon", "million"}:
        amount *= 1_000_000
        scale_key = "milyon"
    elif scale in {"bin", "thousand"}:
        amount *= 1_000
        scale_key = "bin"
    else:
        scale_key = ""
    unit_raw = (unit_match.group(1) if unit_match else "").casefold()
    if unit_raw in {"mt", "m", "metre"}:
        unit = "metre"
    elif unit_raw:
        unit = unit_raw
    else:
        unit = "metre"
    period_raw = (period_match.group(1) if period_match else "").casefold()
    stock = _is_stock_utterance(body)
    if stock:
        period = ""
        canonical = _format_stock_value(amount, unit=unit, scale_key=scale_key)
        kind = "stock"
    else:
        if period_raw in {"ayda", "aylık", "aylik"}:
            period = "Aylık"
        elif period_raw in {"günlük", "gunluk"}:
            period = "Günlük"
        elif period_raw in {"yıllık", "yillik"}:
            period = "Yıllık"
        else:
            period = "Aylık"
        canonical = _format_capacity_value(amount, unit=unit, period=period, scale_key=scale_key)
        kind = "capacity"
    return {
        "raw": body,
        "amount": amount,
        "unit": unit,
        "period": period,
        "canonical": canonical,
        "number_raw": raw_num,
        "kind": kind,
    }


def _format_millions(value: float) -> str:
    millions = value / 1_000_000
    text = f"{millions:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_thousands(value: float) -> str:
    thousands = value / 1_000
    text = f"{thousands:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_stock_value(
    amount: float,
    *,
    unit: str,
    scale_key: str,
) -> str:
    """Stok: '500 bin adet'. Aylık eklenmez."""
    if scale_key == "milyon" or amount >= 1_000_000:
        return f"{_format_millions(amount)} milyon {unit}"
    if scale_key == "bin" or amount >= 1_000:
        return f"{_format_thousands(amount)} bin {unit}"
    qty = f"{amount:.6f}".rstrip("0").rstrip(".")
    return f"{qty} {unit}"


def _format_capacity_value(
    amount: float,
    *,
    unit: str,
    period: str,
    scale_key: str,
) -> str:
    """Tam ölçek: 2.500.000 → 'Aylık 2.5 milyon metre', asla '2.500'."""
    if scale_key == "milyon" or amount >= 1_000_000:
        return f"{period} {_format_millions(amount)} milyon {unit}"
    if scale_key == "bin" or amount >= 1_000:
        return f"{period} {_format_thousands(amount)} bin {unit}"
    qty = f"{amount:.6f}".rstrip("0").rstrip(".")
    return f"{period} {qty} {unit}"


def extract_capacity(text: str) -> str | None:
    parsed = parse_capacity(text)
    if parsed is None or parsed.get("kind") == "stock":
        return None
    return parsed["canonical"]


def extract_stock(text: str) -> str | None:
    parsed = parse_capacity(text)
    if parsed is None or parsed.get("kind") != "stock":
        return None
    return parsed["canonical"]


def split_product_and_volume(text: str) -> tuple[str, str | None, str | None]:
    body = re.sub(r"\s+", " ", (text or "").strip())
    parsed = parse_capacity(body)
    if parsed is None:
        return body, None, None
    product = body.replace(parsed["number_raw"], " ", 1)
    product = _PERIOD_WORD.sub(" ", product)
    product = _SCALE_WORD.sub(" ", product)
    product = _UNIT_WORD.sub(" ", product)
    product = _STOCK_HINT.sub(" ", product)
    product = re.sub(r"\s+", " ", product).strip(" ,.-")
    if parsed.get("kind") == "stock":
        return product, None, parsed["canonical"]
    return product, parsed["canonical"], None


def split_product_and_capacity(text: str) -> tuple[str, str | None]:
    product, capacity, _stock = split_product_and_volume(text)
    return product, capacity


def format_capacity(raw: str) -> str:
    parsed = parse_capacity(raw)
    if parsed:
        return parsed["canonical"]
    text = re.sub(r"\s+", " ", (raw or "").strip())
    text = re.sub(r"(?i)\bmt\b", "metre", text)
    if text and not re.search(r"(?i)(ayda|ayl[ıi]k|günlük|gunluk|yıllık|yillik)", text):
        text = f"Aylık {text}"
    return text


def capacity_reminder(product: str | None, capacity: str | None) -> str:
    """Hafıza kalıbı: aylık 2.5 milyon metre dokuma etiket."""
    cap = format_capacity(capacity or "")
    label = re.sub(r"\s+", " ", (product or "").strip())
    if cap and label:
        return f"{cap} {label}"
    return cap or label


def option_pair(product: str) -> tuple[str, str]:
    low = _fold(product)
    if any(
        word in low
        for word in ("etiket", "dokuma", "tekstil", "kumas", "iplik", "giyim")
    ):
        return (
            "Avrupa'daki hazır giyim üreticilerine bağlayalım",
            "Almanya pazarı için rakip analizi yapalım",
        )
    if any(word in low for word in ("zeytin", "gida", "findik", "tarim")):
        return (
            "Avrupa gıda ithalatçılarına bağlayalım",
            "Almanya ve Hollanda için rakip analizi yapalım",
        )
    if any(word in low for word in ("mobilya", "ahsap")):
        return (
            "Avrupa mobilya alıcılarına bağlayalım",
            "Almanya pazarı için rakip analizi yapalım",
        )
    return (
        "Avrupa'daki alıcılara doğrudan bağlayalım",
        "öncü pazarda rakip analizi yapalım",
    )


def director_options_reply(
    product: str,
    capacity: str | None = None,
    stock: str | None = None,
    matches: list[Any] | None = None,
) -> str:
    """Keşif bitti: eşleşme + fiyat/kanal emri. Anketör sorusu yok."""
    return advisor_mediate_reply(product, capacity, matches, stock=stock)


def classify_option_pick(
    question: str,
) -> Literal["buyer_finder", "supplier_finder", "trade_advisor"] | None:
    text = (question or "").strip()
    if not text or looks_like_capacity(text):
        return None
    if is_strategy_request(text) or is_method_question(text) or is_language_barrier(text) or is_draft_request(text):
        return "trade_advisor"
    if is_buyer_like(text):
        return "buyer_finder"
    analyze = bool(_ANALYZE_PICK.search(text))
    connect = bool(_CONNECT_PICK.search(text))
    if analyze and not connect:
        return "trade_advisor"
    if connect and not analyze:
        return "buyer_finder"
    if analyze:
        return "trade_advisor"
    if connect:
        return "buyer_finder"
    return None


def is_buyer_like(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)(al[iı]c[iı]|m[uü][sş]teri|buyer|bağla|bagla)",
            text or "",
        )
    )


def embed_query(session: SessionState | None, question: str) -> str:
    parts: list[str] = []
    if session is not None:
        if session.product:
            parts.append(session.product)
        if session.capacity:
            parts.append(session.capacity)
        if getattr(session, "stock", None):
            parts.append(session.stock)
        if session.market:
            parts.append(session.market)
    body = (question or "").strip()
    if body and body not in parts:
        parts.append(body)
    return " ".join(parts).strip() or body


_COUNTRY_NAME = {
    "DE": "Almanya",
    "IT": "İtalya",
    "FR": "Fransa",
    "NL": "Hollanda",
    "GB": "İngiltere",
    "UK": "İngiltere",
    "US": "Amerika",
    "ES": "İspanya",
    "BE": "Belçika",
    "PL": "Polonya",
    "AT": "Avusturya",
    "TR": "Türkiye",
}


def _pretty_country(raw: str) -> str | None:
    code = (raw or "").strip()
    if not code or code in {"?", "-"}:
        return None
    upper = code.upper()
    if upper in _COUNTRY_NAME:
        name = _COUNTRY_NAME[upper]
        if name == "Türkiye":
            return None
        return name
    if len(code) <= 3 and code.isalpha():
        return None
    if _fold(code) in {"turkiye", "turkey"}:
        return None
    return code


def markets_from_matches(matches: list[Any], fallback: tuple[str, str] = ("Almanya", "İtalya")) -> list[str]:
    counts: dict[str, int] = {}
    for item in matches or []:
        dest = _pretty_country(
            getattr(item, "destination_country", None)
            or (item.get("destination_country") if isinstance(item, dict) else None)
            or ""
        )
        if dest:
            counts[dest] = counts.get(dest, 0) + 1
    if not counts:
        for item in matches or []:
            origin = _pretty_country(
                getattr(item, "origin_country", None)
                or (item.get("origin_country") if isinstance(item, dict) else None)
                or ""
            )
            if origin:
                counts[origin] = counts.get(origin, 0) + 1
    ranked = [name for name, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    picked = ranked[:2] or list(fallback)
    if len(picked) == 1:
        extra = next((name for name in fallback if name not in picked), None)
        if extra:
            picked.append(extra)
    return picked[:2]


def party_action_reply(
    *,
    role: Literal["buyer", "supplier"],
    product: str | None,
    matches: list[Any] | None = None,
    question: str | None = None,
) -> str:
    """Sorusuz action-first yanıt. Ham sorgu yok; yalnızca ürün slotu."""
    label = action_product_label(product or question)
    party = "alıcı" if role == "buyer" else "tedarikçi"
    return (
        f"Hemen filtreliyorum; {label} için Almanya ve İtalya'daki "
        f"potansiyel {party} listesini çıkarıyorum."
    )


def report_is_complete(text: str) -> bool:
    """Danışman raporu: üç başlık var, soru yok."""
    body = (text or "").strip()
    if not body or "?" in body:
        return False
    low = _fold(body)
    if "hangisiyle" in low:
        return False
    has_entry = "pazar giris" in low
    has_profile = "hedef musteri" in low
    has_logistics = "lojistik" in low or "gumruk" in low
    return has_entry and has_profile and has_logistics


def _advisor_frame(
    product: str | None,
    capacity: str | None,
    matches: list[Any] | None,
    stock: str | None = None,
) -> tuple[str, str, str, str]:
    label = display_product(product)
    if stock and capacity:
        reminder = f"{format_capacity(capacity)} {stock} stok {label}"
    elif stock:
        reminder = f"{stock} stok {label}"
    elif capacity:
        reminder = capacity_reminder(label, capacity)
    else:
        reminder = _short_label(label)
    markets = markets_from_matches(matches or [])
    market_a, market_b = (markets + ["Almanya", "İtalya"])[:2]
    kind = product_kind(product)
    return reminder, market_a, market_b, kind


def _product_en(product: str | None, woven: bool) -> str:
    if product_kind(product) == "print3d":
        return "boutique 3D-printed toys and models"
    if woven:
        return "woven clothing labels"
    label = display_product(product)
    return "our product" if label == "ürününüz" else label


def _product_de(product: str | None, woven: bool) -> str:
    if product_kind(product) == "print3d":
        return "boutique 3D-gedruckte Spielzeuge und Modelle"
    if woven:
        return "gewebte Bekleidungsetiketten"
    label = display_product(product)
    return "unser Produkt" if label == "ürününüz" else label


def _capacity_foreign(reminder: str) -> str:
    text = reminder or ""
    text = re.sub(r"(?i)^ayl[ıi]k\s+", "monthly ", text)
    text = re.sub(r"(?i)\bmilyon\b", "million", text)
    text = re.sub(r"(?i)\bmetre\b", "meters", text)
    return text


def _match_get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _match_org_name(item: Any) -> str:
    org = str(_match_get(item, "organization_name") or "").strip()
    if not org:
        nested = _match_get(item, "organization") or _match_get(item, "organizations")
        if isinstance(nested, list) and nested:
            nested = nested[0]
        if isinstance(nested, dict):
            org = str(nested.get("name") or "").strip()
    if org and org.casefold() not in {"none", "null", "kayıt yok"}:
        return org
    return ""


def match_firm_label(item: Any) -> str:
    org = _match_org_name(item)
    if org:
        return org
    product = str(_match_get(item, "product_name") or "").strip()
    dest = _pretty_country(str(_match_get(item, "destination_country") or ""))
    if dest and product:
        short = re.sub(r"\s+", " ", product)
        if len(short) > 36:
            short = short[:34].rstrip() + "…"
        return f"{dest} alıcısı · {short}"
    if dest:
        return f"{dest} alıcısı"
    return product or "eşleşen alıcı"


def match_firm_addressee(item: Any) -> str:
    """Mail hitabı: firma adı, yoksa ülke alıcısı (kırpma yok)."""
    org = _match_org_name(item)
    if org:
        return org
    dest = _pretty_country(str(_match_get(item, "destination_country") or ""))
    if dest:
        return f"{dest} alıcısı"
    product = str(_match_get(item, "product_name") or "").strip()
    return product or "Purchasing Team"


def match_firm_labels(matches: list[Any] | None, limit: int = 4) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in matches or []:
        label = match_firm_label(item)
        key = _fold(label)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(label)
        if len(names) >= limit:
            break
    return names


def _price_playbook(kind: str, *, has_stock: bool) -> dict[str, str]:
    """Operasyonel fiyat ve kanal. Uydurma şirket yok."""
    if kind == "print3d":
        return {
            "tr_band": "parça başı 80–250 TL (modele göre)",
            "eu_band": "4–12 EUR/parça küçük seri",
            "domestic": "Etsy, Amazon Handmade ve yerel hediyelik eşya dükkanları",
            "export_who": (
                "butik masaüstü oyun tasarımcıları, mimari maket büroları "
                "ve kişiselleştirilmiş ürün (B2C/B2B2C) alıcıları"
            ),
            "split": (
                "Dev oyuncak üreticisine (Ravensburger vb.) teklif atmayın. "
                "Küçük seri ve kişiselleştirme kanalına gidin."
            ),
            "terms": (
                "PLA ve PETG. Hassasiyet ±0,15–0,20 mm. "
                "Prototip 5–7 gün, onay sonrası kısa seri 10–18 gün. "
                "%40 peşin, %60 teslimde."
            ),
            "rival": (
                "Uzak Doğu kalıp maliyeti yüksek, adet ister. "
                "Siz numune hızı ve kişiselleştirmeyi öne sürün."
            ),
        }
    if kind == "steel":
        return {
            "tr_band": "ton başı iç piyasa, nakit eksi yüzde 3–6",
            "eu_band": "CFR / DAP Avrupa, liste artı net yüzde 8–14",
            "domestic": "İstanbul, Kocaeli ve İzmir çelik tüccarları",
            "export_who": (
                "inşaat yüklenicileri, otomotiv tedarikçileri, "
                "makine imalatçıları ve çelik tüccarları"
            ),
            "split": (
                "İnşaat ve tüccar kanalına nakit tonaj. "
                "Otomotiv ve makineye sertifikalı, daha yüksek marjlı dilim."
            ),
            "terms": (
                "EXW Türkiye veya DAP Duisburg. EN 10025 / CE. "
                "%30 peşin, %70 konşimentoda. Teslim 2–4 hafta."
            ),
            "rival": (
                "Uzak Doğu navlun ve süre riski taşır. "
                "Siz Avrupa teslimini ve mill test sertifikasını öne sürün."
            ),
        }
    if kind == "woven":
        return {
            "tr_band": "0,90–1,30 TL/adet",
            "eu_band": "0,028–0,038 EUR/adet",
            "domestic": "İstanbul Merter ve Osmanbey ile Bursa tekstil toptancıları",
            "export_who": "hazır giyim üreticileri ve etiket toptancıları",
            "split": (
                "İlk 120–150 bin adeti iç piyasaya nakit çevirin. "
                "Kalanı üç ihracat alıcısına bölün. Tek firmaya tüm stoğu mal etmeyin."
                if has_stock
                else "İlk siparişi 30–50 bin adet kilitleyin. Tekrarı aynı tezgâhtan alın."
            ),
            "terms": (
                "EXW Türkiye veya DAP Duisburg. %30 peşin, %70 yüklemede. "
                "Numune 48 saat. Stok varsa teslim 7–10 gün yazın."
                if has_stock
                else "EXW Türkiye veya DAP Duisburg. %30 peşin, %70 yüklemede. Teslim 4–6 hafta."
            ),
            "rival": (
                "Uzak Doğu düşük fiyat verir, süre 8–12 haftadır. "
                "Siz süreyi ve tekrar siparişi öne sürün; fiyatı dip yapmayın."
            ),
        }
    return {
        "tr_band": "iç piyasa toptan, liste eksi yüzde 8–12 nakit",
        "eu_band": "ihracat listesi artı net yüzde 12–18",
        "domestic": "ilgili sektörün İstanbul ve Anadolu toptancı halleri",
        "export_who": "ithalatçı toptancılar ve marka tedarik ekipleri",
        "split": (
            "Önce iç piyasadan nakit alın. Kalan lotu iki ihracat alıcısına bölün."
            if has_stock
            else "İlk konteyneri tek hesapta kilitleyin. İkinci turu fiyat yükseltmeden bağlayın."
        ),
        "terms": (
            "EXW veya DAP. %30 peşin, %70 yüklemede. Hazır mal 7–10 gün."
            if has_stock
            else "EXW veya DAP. %30 peşin, %70 yüklemede. Teslim süresini net yazın."
        ),
        "rival": "Fiyat kırışına girmeyin. Teslim ve kalite belgesini masaya koyun.",
    }


def _volume_lead(
    product: str | None,
    capacity: str | None,
    stock: str | None,
) -> str:
    label = display_product(product)
    cap = format_capacity(capacity) if capacity else None
    stk = (stock or "").strip() or None
    if cap and stk:
        return f"Sizin {cap} kapasitenize, {stk} stokunuza ve {label} ürününüze uygun olarak"
    if stk:
        return f"Sizin {stk} stokunuza ve {label} ürününüze uygun olarak"
    if cap:
        return f"Sizin {cap} kapasitenize ve {label} ürününüze uygun olarak"
    return f"Sizin {label} ürününüze uygun olarak"


_PRINT3D_FALLBACK_ADDRESSEES = (
    "Etsy Handmade sellers desk",
    "Amazon Handmade category buyer",
    "Boutique tabletop game studio",
)


def _domain_notes(kind: str) -> str:
    if kind != "print3d":
        return ""
    return (
        "\n\nEtkinlik / Sertifika notu\n"
        "- Spielwarenmesse: oyuncak fuarıdır; alıcı firma değildir. "
        "Stand ve randevu için kullanın.\n"
        "- TÜV SÜD: CE ve oyuncak güvenliği kapısıdır; alıcı değildir. "
        "Sertifika sürecinde devreye girer."
    )


def advisor_mediate_reply(
    product: str | None,
    capacity: str | None = None,
    matches: list[Any] | None = None,
    stock: str | None = None,
) -> str:
    """Direktör emri: eşleşme + fiyat + kanal + şart. Soru yok."""
    kind = product_kind(product)
    firms = match_firm_labels(matches)
    if kind == "steel":
        label = action_product_label(product)
        lead = (
            f"{label} üretimi için ihracat pazarlarını ve B2B alıcı kanallarını "
            "hemen analiz ediyorum."
        )
        if firms:
            who = (
                " İnşaat, otomotiv, makine imalatı ve çelik ticareti kanallarında "
                f"şu alıcılarla eşleştiniz: {', '.join(firms)}."
            )
        else:
            who = (
                " İnşaat, otomotiv, makine imalatçıları ve çelik tüccarları "
                "kartlarda açıldı."
            )
        _reminder, market_a, market_b, _kind = _advisor_frame(
            product, capacity, matches, stock=stock
        )
        book = _price_playbook(kind, has_stock=bool(stock))
        targets = ", ".join(firms) if firms else book["export_who"]
        order = (
            f"{who} Karlı hamle: nakit {book['domestic']} kanalından, marj "
            f"{market_a} ve {market_b} ihracatından. İç piyasa {book['tr_band']}; "
            f"ihracat {book['eu_band']}. {targets} hesabına bu hafta bu fiyattan "
            f"teklif atın. Şart: {book['terms']} {book['split']} "
            "İngilizce/Almanca mail taslağı için «mail taslağı» yazın."
        )
        return lead + order
    party = "kanallarla" if kind == "print3d" else "firmalarla"
    if firms:
        who = f"pazarınızdaki şu {party} eşleştiniz: {', '.join(firms)}."
    else:
        who = f"sağdaki eşleşen kartlardaki {party} eşleştiniz."
    lead = f"{_volume_lead(product, capacity, stock)} {who}"
    _reminder, market_a, market_b, _kind = _advisor_frame(
        product, capacity, matches, stock=stock
    )
    book = _price_playbook(kind, has_stock=bool(stock))
    targets = ", ".join(firms) if firms else f"{market_a} ve {market_b} {book['export_who']}"
    order = (
        f" Karlı hamle: nakit {book['domestic']} kanalından, marj {market_a} ve "
        f"{market_b} ihracatından. İç piyasa {book['tr_band']}; ihracat "
        f"{book['eu_band']}. {targets} hesabına bu hafta bu fiyattan teklif atın. "
        f"Şart: {book['terms']} {book['split']} "
        "İngilizce/Almanca mail taslağı için «mail taslağı» yazın."
    )
    return lead + order + _domain_notes(kind)


def _single_addressee(name: str) -> str:
    """Bir mailde tek alıcı; virgüllü toplu hitap yok."""
    body = re.sub(r"\s+", " ", (name or "").strip())
    if not body:
        return "Purchasing Team"
    first = re.split(r"\s*[;|/]\s*", body)[0].strip()
    if " and " in first.casefold() and len(first) > 48:
        first = first.split(",")[0].strip()
    return first or "Purchasing Team"


def _draft_bodies(kind: str, en_name: str, de_name: str, cap_en: str) -> tuple[str, str]:
    if kind == "print3d":
        en = (
            f"We provide boutique 3D printing in Turkey for {en_name}. "
            "Materials: PLA and PETG. Typical accuracy is ±0.15–0.20 mm. "
            "Prototype lead time is 5–7 days; short runs ship in 10–18 days after approval. "
            f"Capacity note: {cap_en}."
        )
        de = (
            f"Wir bieten boutique 3D-Druck in der Tuerkei fuer {de_name}. "
            "Material: PLA und PETG. Genauigkeit ca. ±0,15–0,20 mm. "
            "Prototyp in 5–7 Tagen; Kleinserie 10–18 Tage nach Freigabe. "
            f"Kapazitaet: {cap_en}."
        )
        return en, de
    en = (
        f"We manufacture {en_name} in Turkey. Capacity: {cap_en}. "
        "Please find our offer: sample this week, lead time 4-6 weeks, "
        "repeat orders from the same mill."
    )
    de = (
        f"Wir produzieren {de_name} in der Tuerkei. Kapazitaet: {cap_en}. "
        "Angebot: Muster diese Woche, Lieferzeit 4-6 Wochen, "
        "Wiederholauftraege aus derselben Produktion."
    )
    return en, de


def advisor_draft_reply(
    product: str | None,
    capacity: str | None = None,
    matches: list[Any] | None = None,
    stock: str | None = None,
) -> str:
    """Tek alıcıya özel kopyala-yapıştır EN/DE teklif. Toplu Dear yok."""
    reminder, _market_a, _market_b, kind = _advisor_frame(
        product, capacity, matches, stock=stock
    )
    woven = kind == "woven"
    en_name = _product_en(product, woven)
    de_name = _product_de(product, woven)
    cap_src = format_capacity(capacity) if capacity else reminder
    cap_en = _capacity_foreign(cap_src)
    if stock:
        cap_en = f"{_capacity_foreign(stock)} in stock, ship in 7-10 days"
    en_body, de_body = _draft_bodies(kind, en_name, de_name, cap_en)
    firms = match_firm_labels(matches, limit=3)
    addressees: list[str] = []
    seen: set[str] = set()
    for item in matches or []:
        name = _single_addressee(match_firm_addressee(item))
        key = _fold(name)
        if not key or key in seen:
            continue
        seen.add(key)
        addressees.append(name)
        if len(addressees) >= 3:
            break
    if not addressees:
        if kind == "print3d":
            addressees = list(_PRINT3D_FALLBACK_ADDRESSEES[:3])
        else:
            addressees = ["Purchasing Team"]
        firms = firms or list(addressees)
    while len(firms) < len(addressees):
        firms.append(addressees[len(firms)])
    blocks: list[str] = []
    for i, firm in enumerate(addressees):
        header = _single_addressee(firms[i] if i < len(firms) else firm)
        to_en = _single_addressee(firm)
        dear = (
            f"Dear {to_en},"
            if kind == "print3d"
            else f"Dear Purchasing Team at {to_en},"
        )
        sehr = (
            f"Sehr geehrte Damen und Herren bei {to_en},"
            if kind == "print3d"
            else f"Sehr geehrte Damen und Herren, {to_en},"
        )
        blocks.append(
            f"{header}\n"
            "İngilizce teklif\n"
            f"Subject: {en_name} — offer and sample from Turkey\n"
            f"{dear}\n"
            f"{en_body}\n"
            "Best regards\n\n"
            "Almanca teklif\n"
            f"Betreff: {de_name} — Angebot und Muster aus der Tuerkei\n"
            f"{sehr}\n"
            f"{de_body}\n"
            "Mit freundlichen Gruessen"
        )
    intro = (
        "Kopyala-yapıştır hazır. Her blok tek bir alıcıya özel; "
        "firmaları aynı hitap satırına yazmayın."
    )
    return intro + "\n\n" + "\n\n".join(blocks)


def advisor_outreach_reply(
    product: str | None,
    capacity: str | None = None,
    matches: list[Any] | None = None,
    *,
    language_help: bool = False,
) -> str:
    """Kopyala-yapıştır mail + numune taktiği. Soru yok."""
    reminder, market_a, market_b, kind = _advisor_frame(product, capacity, matches)
    woven = kind == "woven"
    en_name = _product_en(product, woven)
    de_name = _product_de(product, woven)
    cap_src = format_capacity(capacity) if capacity else reminder
    cap_en = _capacity_foreign(cap_src)
    if language_help:
        lead = (
            "Yabancı dil şart değil. Aşağıdaki İngilizce ve Almanca metinleri "
            "olduğu gibi kopyalayıp yapıştırın."
        )
    else:
        lead = (
            f"{reminder} için {market_a} ve {market_b}'ya bu hafta numune gidecek. "
            "Maili kopyalayıp gönderin."
        )
    return (
        f"{lead}\n\n"
        "İngilizce e-posta\n"
        f"Subject: {en_name} — sample from Turkey\n"
        "Dear Purchasing Team,\n"
        f"We manufacture {en_name} in Turkey. Capacity: {cap_en}. "
        "We can send a sample this week. Lead time is 4-6 weeks.\n"
        "Best regards\n\n"
        "Almanca e-posta\n"
        f"Betreff: {de_name} — Muster aus der Tuerkei\n"
        "Sehr geehrte Damen und Herren,\n"
        f"Wir produzieren {de_name} in der Tuerkei. Kapazitaet: {cap_en}. "
        "Muster senden wir diese Woche. Lieferzeit 4-6 Wochen.\n"
        "Mit freundlichen Gruessen\n\n"
        "Numune taktiği\n"
        f"- {market_a} ve {market_b}'daki 8-12 firmaya bu hafta gidin.\n"
        "- Pakette numune, kapasite satırı ve teslim süresi olsun.\n"
        "- Maili attıktan sonra 5-7 gün bekleyin. Cevap yoksa aynı metni bir kez daha gönderin.\n"
        "- Firma adını konu satırına ekleyin. Gövdeyi değiştirmeyin."
    )


def advisor_language_reply(
    product: str | None,
    capacity: str | None = None,
    matches: list[Any] | None = None,
) -> str:
    """Dil engeli: şablon + çeviri pratiği. Soru yok."""
    kit = advisor_outreach_reply(
        product, capacity, matches, language_help=True
    )
    return (
        f"{kit}\n\n"
        "Çeviri pratiği\n"
        "- Gelen maili Google Translate'e yapıştırıp okuyun.\n"
        "- Cevabı kendiniz İngilizce yazmaya çalışmayın. Aşağıdaki kısa yanıtı kullanın.\n"
        "- EN: Thank you. Sample is ready. Please share your shipping address.\n"
        "- DE: Vielen Dank. Das Muster ist bereit. Bitte teilen Sie Ihre Lieferadresse mit.\n"
        "- Adres geldikten sonra kargoyu aynı gün çıkarın. Takip numarasını aynı dilde iletin:\n"
        "- EN: Tracking number: ... Sample has been shipped.\n"
        "- DE: Sendungsnummer: ... Das Muster wurde versendet."
    )


def advisor_report_reply(
    product: str | None,
    capacity: str | None = None,
    matches: list[Any] | None = None,
    stock: str | None = None,
) -> str:
    """Direktör raporu: fiyat, kanal, rakip, gümrük. Soru yok."""
    reminder, market_a, market_b, kind = _advisor_frame(
        product, capacity, matches, stock=stock
    )
    book = _price_playbook(kind, has_stock=bool(stock))
    firms = match_firm_labels(matches, limit=3)
    named = ", ".join(firms) if firms else f"{market_a} ve {market_b} {book['export_who']}"
    if kind == "woven":
        steps = (
            f"- Bu hafta {named} hesabına {book['eu_band']} bandından teklif atın.\n"
            f"- İç piyasada {book['domestic']} için {book['tr_band']}.\n"
            f"- Teklifte hacmi açık yazın: {reminder}.\n"
            f"- Şartı öne sürün: {book['terms']}\n"
            f"- {book['split']}"
        )
        profiles = (
            f"- {market_a} ve {market_b}'daki giyim üreticileri. Kendi marka etiketini diktirecek atölyeler.\n"
            f"- {book['domestic']}. Nakit ve hızlı çekim.\n"
            f"- Rakip: {book['rival']}"
        )
        logistics = (
            "- Faturada ürün adı «dokuma etiket» olsun. Gümrük kâğıdında da aynı ad yazsın.\n"
            "- Avrupa sevkiyatında menşe belgesi alın (malın Türkiye'de üretildiğini gösteren kâğıt).\n"
            "- Kara yolu 3–5 gün. Numuneyi ayrı kargolayın. Asıl siparişi ayrı gönderin.\n"
            "- Rulolar ezilmesin, ıslanmasın. Kartona sarın."
        )
    elif kind == "print3d":
        steps = (
            f"- {named} hesabına PLA/PETG, ±0,15–0,20 mm ve 5–7 gün prototip yazarak teklif atın.\n"
            f"- Kanal: {book['domestic']}. Dev oyuncak üreticisine teklif atmayın.\n"
            f"- Teklifte hacmi açık yazın: {reminder}.\n"
            f"- Şart: {book['terms']}\n"
            f"- {book['split']}"
        )
        profiles = (
            "- Etsy ve Amazon Handmade: kişiselleştirilmiş B2C vitrin.\n"
            "- Yerel hediyelik eşya dükkanları ve butik masaüstü oyun tasarımcıları.\n"
            "- Mimari maket büroları. B2B2C kişiselleştirme.\n"
            f"- Rakip: {book['rival']}"
        )
        logistics = (
            "- Faturada ürün adı «3D baskı oyuncak / maket» olsun.\n"
            "- CE / oyuncak güvenliği için TÜV SÜD kapısını not edin; alıcı diye yazmayın.\n"
            "- Parçaları ezilmeyecek şekilde köpükle paketleyin. Numuneyi ayrı kargolayın."
        )
    else:
        steps = (
            f"- {named} hesabına bu hafta {book['eu_band']} ile teklif atın.\n"
            f"- İç piyasa: {book['tr_band']}.\n"
            f"- Teklifte hacmi açık yazın: {reminder}.\n"
            f"- Şart: {book['terms']}\n"
            f"- {book['split']}"
        )
        profiles = (
            f"- {market_a} ve {market_b}'daki üreticiler ve marka alıcıları.\n"
            f"- {book['domestic']}.\n"
            f"- Rakip: {book['rival']}"
        )
        logistics = (
            "- Fatura, ürün adı ve gümrük kâğıdı aynı bilgiyi taşısın.\n"
            "- Numuneyi ayrı, asıl malı ayrı gönderin. Kara yolu 3–5 gün.\n"
            "- Avrupa sevkiyatında menşe belgesi alın (malın nerede üretildiğini gösteren kâğıt).\n"
            "- Ambalajı ezilmeye ve neme göre hazırlayın."
        )
    return (
        f"{reminder} için {market_a} ve {market_b} pazarına giriş notu.\n\n"
        f"Pazar giriş adımları\n{steps}\n\n"
        f"Hedef müşteri profilleri\n{profiles}\n\n"
        f"Lojistik ve gümrük ipuçları\n{logistics}"
    ) + _domain_notes(kind)


def advisor_fair_reply(
    product: str | None,
    capacity: str | None = None,
    matches: list[Any] | None = None,
) -> str:
    reminder, market_a, market_b, kind = _advisor_frame(product, capacity, matches)
    if kind == "woven":
        fairs = (
            "- Almanya: Frankfurt Texprocess. Giyim üreticileri ve etiket alanlar orada olur.\n"
            "- Paris: Texworld. Avrupa giyim alıcıları dolaşır.\n"
            "- Eurocetex: etiket ve aksesuar tarafı. Standları tek tek gezin."
        )
    elif kind == "print3d":
        fairs = (
            "- Spielwarenmesse (Nürnberg): oyuncak fuarıdır; alıcı firma değildir. "
            "Randevu ve stand için gidin, kart listesine yazmayın.\n"
            "- Formnext / yerel maker ve hediyelik fuarları: butik 3D ve maket alıcısı orada olur.\n"
            "- Amazon Handmade ve Etsy vitrinini fuar yerine sürekli kanal sayın."
        )
    else:
        fairs = (
            f"- {market_a} ve {market_b} sektör fuar takvimine bakın.\n"
            "- Almanya ve Fransa'daki büyük ticaret fuarlarını ilk sıraya koyun.\n"
            "- Tekstil ise Texworld ve Eurocetex'i de tarayın."
        )
    return (
        f"{reminder} için fuar yolu.\n\n"
        f"{fairs}\n"
        "- Gitmeden 20 firmaya randevu yazın.\n"
        "- Çantada numune ve fiyat listesi olsun.\n"
        "- Fuar sonrası aynı gün kısa bir teşekkür ve numune teklifi gönderin."
        + _domain_notes(kind)
    )


def advisor_linkedin_reply(
    product: str | None,
    capacity: str | None = None,
    matches: list[Any] | None = None,
) -> str:
    reminder, market_a, market_b, kind = _advisor_frame(product, capacity, matches)
    label = re.sub(r"\s+", " ", (product or "ürününüz").strip()) or "ürününüz"
    sector = (
        "giyim, tekstil"
        if kind == "woven"
        else (
            "masaüstü oyun, mimari maket, hediyelik, kişiselleştirilmiş ürün"
            if kind == "print3d"
            else "ilgili sektör"
        )
    )
    return (
        f"{reminder} için LinkedIn hedefleme.\n\n"
        f"- Ülke filtresi: {market_a}, {market_b}.\n"
        "- Unvan: satın alma, tedarik, üretim. LinkedIn'de purchasing / sourcing diye arayın.\n"
        f"- Sektör: {sector}.\n"
        f"- Kısa mesaj: {label} üretiyoruz. Kapasite: {reminder}. Numune gönderebilirim.\n"
        "- İlk turda 8–12 kişiye yazın.\n"
        "- Aynı kişiye iki günde bir kez daha yazmayın."
    )


def advisor_reach_reply(
    product: str | None,
    capacity: str | None = None,
    matches: list[Any] | None = None,
) -> str:
    reminder, market_a, market_b, kind = _advisor_frame(product, capacity, matches)
    if kind == "woven":
        fair_line = "- Fuar: Almanya Texprocess, Paris Texworld, Eurocetex."
    elif kind == "print3d":
        fair_line = (
            "- Fuar notu: Spielwarenmesse etkinliktir, alıcı kartı değildir. "
            "Asıl kanal Etsy, Amazon Handmade ve yerel hediyelik."
        )
    else:
        fair_line = f"- Fuar: {market_a} ve {market_b} sektör fuarları."
    return (
        f"{reminder} için alıcıya nasıl ulaşacağınız.\n\n"
        f"{fair_line}\n"
        f"- LinkedIn: {market_a} ve {market_b} için purchasing / sourcing unvanına kısa mesaj.\n"
        "- Firma sitesinden satın alma maili. Numune, kapasite, teslim süresi yazın.\n"
        "- İlk turda 8–12 kişi.\n"
        "- Aynı metni herkese yapıştırmayın. Firma adına göre bir cümle değiştirin."
        + _domain_notes(kind)
    )


def _render_advisor(
    focus: str,
    product: str | None,
    capacity: str | None,
    matches: list[Any] | None,
    stock: str | None = None,
) -> str:
    if focus == "language":
        return advisor_language_reply(product, capacity, matches)
    if focus == "draft":
        return advisor_draft_reply(product, capacity, matches, stock=stock)
    if focus == "outreach":
        return advisor_draft_reply(product, capacity, matches, stock=stock)
    if focus == "mediate":
        return advisor_mediate_reply(product, capacity, matches, stock=stock)
    if focus == "fair":
        return advisor_fair_reply(product, capacity, matches)
    if focus == "linkedin":
        return advisor_linkedin_reply(product, capacity, matches)
    if focus == "reach":
        return advisor_reach_reply(product, capacity, matches)
    return advisor_report_reply(product, capacity, matches, stock=stock)


def advisor_answer(
    question: str,
    session: SessionState | None = None,
    matches: list[Any] | None = None,
) -> str:
    """Soruya göre rapor veya kanal notu. Aynı metin arka arkaya basılmaz."""
    if session is not None:
        apply_utterance_slots(session, question)
    product = session.product if session is not None else None
    capacity = session.capacity if session is not None else None
    stock = session.stock if session is not None else None
    focus = advisor_focus(question)
    text = strip_echoed_query(
        question, _render_advisor(focus, product, capacity, matches, stock)
    )
    prev = (session.last_advisor_text or "").strip() if session is not None else ""
    sticky = {"mediate", "draft", "language"}
    if prev and text.strip() == prev and focus not in sticky:
        for alt in ("reach", "fair", "linkedin", "draft", "mediate", "language", "report"):
            if alt == focus:
                continue
            cand = strip_echoed_query(
                question, _render_advisor(alt, product, capacity, matches, stock)
            )
            if cand.strip() != prev:
                text = cand
                focus = alt
                break
    if session is not None:
        session.last_advisor_text = text
        session.last_advisor_kind = focus
    return text


def looks_like_name(text: str) -> bool:
    body = (text or "").strip()
    if not body or looks_like_product(body):
        return False
    words = body.split()
    if len(words) > 4 or _DIGIT.search(body):
        return False
    if re.search(r"(?i)(merhaba|selam|nas[iı]l|teşekkür|tesekkur)", body):
        return False
    return True


@dataclass
class ChatTurn:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class SessionAccessDenied(Exception):
    """Cross-account session access — map to HTTP 404 at the API edge."""


class SessionState:
    # Pilot Security Phase 1: ownership metadata (commercial slots unchanged).
    account_id: str | None = None
    created_by_user_id: str | None = None
    session_id: str
    name: str | None = None
    product: str | None = None
    capacity: str | None = None
    stock: str | None = None
    market: str | None = None
    last_ask: AskSlot = None
    last_intent: str | None = None
    last_advisor_text: str | None = None
    last_advisor_kind: str | None = None
    last_goal: str | None = None
    memories: list[str] = field(default_factory=list)
    messages: list[ChatTurn] = field(default_factory=list)

    def awaiting(self) -> bool:
        return self.last_ask is not None

    def in_progress(self) -> bool:
        return bool(
            self.last_ask
            or self.product
            or self.name
            or self.capacity
            or self.stock
            or self.market
            or self.messages
        )

    def slots_open(self) -> bool:
        return not self.product

    def append(self, role: str, content: str) -> None:
        body = (content or "").strip()
        if not body:
            return
        if self.messages:
            last = self.messages[-1]
            if last.role == role and last.content == body:
                return
        self.messages.append(ChatTurn(role=role, content=body))
        if len(self.messages) > MAX_TURNS:
            self.messages = self.messages[-MAX_TURNS:]

    def history_dicts(self) -> list[dict[str, str]]:
        return [turn.to_dict() for turn in self.messages]


_STORE: dict[str, SessionState] = {}
_LOCK = Lock()

_TITLE_MARKET = re.compile(
    r"(?i)\b("
    r"almanya|italya|[iı]talya|fransa|hollanda|ispanya|ingiltere|"
    r"avrupa|amerika|bel[cç]ika|polonya|avusturya"
    r")\b"
)
_MARKET_PRETTY = {
    "almanya": "Almanya",
    "italya": "İtalya",
    "ıtalya": "İtalya",
    "fransa": "Fransa",
    "hollanda": "Hollanda",
    "ispanya": "İspanya",
    "ingiltere": "İngiltere",
    "avrupa": "Avrupa",
    "amerika": "Amerika",
    "belcika": "Belçika",
    "belçika": "Belçika",
    "polonya": "Polonya",
    "avusturya": "Avusturya",
}


def _normalize_history(raw: list[Any] | None) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    for item in raw or []:
        if isinstance(item, ChatTurn):
            role, content = item.role, item.content
        elif isinstance(item, dict):
            role = str(item.get("role") or "")
            content = str(item.get("content") or "")
        else:
            role = str(getattr(item, "role", ""))
            content = str(getattr(item, "content", ""))
        if role not in ("user", "assistant"):
            continue
        body = content.strip()
        if not body:
            continue
        turns.append(ChatTurn(role=role, content=body))
    return turns[-MAX_TURNS:]


def infer_slots(state: SessionState) -> None:
    """Asistan sorularından last_ask / dolu slotları geri kur."""
    msgs = state.messages
    for index, turn in enumerate(msgs):
        if turn.role != "assistant":
            continue
        low = _fold(turn.content)
        nxt = (
            msgs[index + 1].content.strip()
            if index + 1 < len(msgs) and msgs[index + 1].role == "user"
            else None
        )
        if "hitap" in low:
            if nxt and looks_like_name(nxt):
                state.name = nxt
            if nxt is None:
                state.last_ask = "name"
        if "hangi urun" in low:
            if nxt:
                product, cap, stock = split_product_and_volume(nxt)
                state.product = product or nxt
                if cap:
                    state.capacity = cap
                if stock:
                    state.stock = stock
                state.last_ask = None
            else:
                state.last_ask = "product"
        if "hangisiyle" in low or "baglayalim" in low:
            if nxt and looks_like_stock(nxt):
                state.stock = extract_stock(nxt) or nxt
                state.last_ask = None
            elif nxt and looks_like_capacity(nxt):
                state.capacity = extract_capacity(nxt) or nxt
                state.last_ask = None
            elif nxt and classify_option_pick(nxt):
                state.last_ask = None
            elif nxt is None:
                state.last_ask = None
        if "kapasite" in low and "ciddi bir guc" not in low:
            if nxt:
                if looks_like_stock(nxt):
                    state.stock = extract_stock(nxt) or nxt
                else:
                    state.capacity = extract_capacity(nxt) or nxt
                state.last_ask = None
            elif "hangisiyle" not in low:
                state.last_ask = None

    for turn in msgs:
        if turn.role != "user":
            continue
        apply_utterance_slots(state, turn.content)

    if not state.market:
        state.market = market_from_turns(msgs)

    if msgs and msgs[-1].role == "assistant":
        low = _fold(msgs[-1].content)
        if "hitap" in low:
            state.last_ask = "name"
        elif "hangi urun" in low:
            state.last_ask = "product"
        elif "hangisiyle" in low:
            state.last_ask = None

    for turn in reversed(msgs):
        if turn.role != "assistant":
            continue
        body = (turn.content or "").strip()
        if not body:
            continue
        folded = _fold(body)
        if "hangisiyle" in folded or "hangi urun" in folded:
            state.last_advisor_text = None
            state.last_advisor_kind = None
            break
        state.last_advisor_text = body
        if "eslestiniz" in folded or "mail taslagi" in folded:
            state.last_advisor_kind = "mediate"
        elif "subject:" in folded or "dear purchasing" in folded:
            state.last_advisor_kind = "draft"
        elif "pazar giris adim" in folded:
            state.last_advisor_kind = "report"
        elif "linkedin" in folded:
            state.last_advisor_kind = "linkedin"
        elif "fuar" in folded:
            state.last_advisor_kind = "fair"
        else:
            state.last_advisor_kind = "reach"
        break


def _clear_slots(state: SessionState) -> None:
    state.name = None
    state.product = None
    state.capacity = None
    state.stock = None
    state.market = None
    state.last_ask = None
    state.last_intent = None
    state.last_advisor_text = None
    state.last_advisor_kind = None
    state.last_goal = None


def _pretty_title_word(word: str) -> str:
    if not word:
        return word
    if word.isupper() and len(word) <= 4:
        return word
    return word[0].upper() + word[1:]


def _pretty_title(text: str) -> str:
    body = re.sub(r"\s+", " ", (text or "").strip())
    if not body:
        return ""
    return " ".join(_pretty_title_word(part) for part in body.split())


def market_from_text(text: str) -> str | None:
    match = _TITLE_MARKET.search(text or "")
    if not match:
        return None
    key = _fold(match.group(1))
    return _MARKET_PRETTY.get(key) or _pretty_title(match.group(1))


def market_from_turns(turns: list[ChatTurn]) -> str | None:
    for turn in turns:
        found = market_from_text(turn.content)
        if found:
            return found
    return None


def session_title(session: SessionState | None) -> str:
    if session is None:
        return "Yeni sohbet"
    product = _pretty_title(session.product or "")
    market = (session.market or "").strip() or market_from_turns(session.messages)
    if product and market:
        return f"{product} - {market}"
    if product:
        return product
    for turn in session.messages:
        if turn.role == "user":
            snippet = re.sub(r"\s+", " ", turn.content).strip()
            if snippet:
                return snippet[:48]
    return "Yeni sohbet"


def session_snapshot(session: SessionState) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "account_id": session.account_id,
        "created_by_user_id": session.created_by_user_id,
        "title": session_title(session),
        "product": session.product,
        "capacity": session.capacity,
        "stock": session.stock,
        "market": session.market,
        "last_intent": session.last_intent,
        "history": session.history_dicts(),
    }


def create_session(
    *,
    account_id: str | None = None,
    created_by_user_id: str | None = None,
) -> SessionState:
    """Yeni izole oturum: taze session_id, boş niyet/kapasite (+ ownership)."""
    sid = str(uuid4())
    with _LOCK:
        state = SessionState(
            session_id=sid,
            account_id=(account_id or None),
            created_by_user_id=(created_by_user_id or None),
        )
        _STORE[sid] = state
        return state


def get_session(session_id: str | None) -> SessionState | None:
    sid = (session_id or "").strip()
    if not sid:
        return None
    with _LOCK:
        return _STORE.get(sid)


def session_owned_by(session: SessionState | None, account_id: str | None) -> bool:
    if session is None or not account_id:
        return False
    owned = (session.account_id or "").strip()
    return bool(owned) and owned == account_id.strip()


def get_session_for_account(
    session_id: str | None, account_id: str | None
) -> SessionState | None:
    """Return session only when it belongs to account_id; else None (→ 404)."""
    state = get_session(session_id)
    if state is None:
        return None
    if not session_owned_by(state, account_id):
        return None
    return state


def list_sessions(account_id: str | None = None) -> list[dict[str, Any]]:
    with _LOCK:
        rows = []
        for state in _STORE.values():
            if not (state.messages or state.product or state.capacity or state.stock):
                continue
            if account_id and not session_owned_by(state, account_id):
                continue
            rows.append(session_snapshot(state))
    return list(reversed(rows))


def delete_session(
    session_id: str | None,
    *,
    account_id: str | None = None,
) -> bool:
    """Oturumu bellekten siler. account_id verilirse ownership zorunlu."""
    sid = (session_id or "").strip()
    if not sid:
        return False
    with _LOCK:
        state = _STORE.get(sid)
        if state is None:
            return False
        if account_id is not None and not session_owned_by(state, account_id):
            return False
        _STORE.pop(sid, None)
        return True


def hydrate(
    session_id: str | None,
    history: list[Any] | None,
    *,
    account_id: str | None = None,
    created_by_user_id: str | None = None,
    enforce_ownership: bool = False,
) -> SessionState:
    sid = (session_id or "").strip() or str(uuid4())
    incoming = _normalize_history(history)
    with _LOCK:
        state = _STORE.get(sid)
        if state is None:
            state = SessionState(
                session_id=sid,
                account_id=account_id,
                created_by_user_id=created_by_user_id,
            )
            _STORE[sid] = state
        elif enforce_ownership:
            existing = (state.account_id or "").strip()
            wanted = (account_id or "").strip()
            if existing and wanted and existing != wanted:
                raise SessionAccessDenied(sid)
            if not existing and wanted:
                state.account_id = wanted
                state.created_by_user_id = created_by_user_id
        if history is not None:
            _clear_slots(state)
            state.messages = incoming
            if incoming:
                infer_slots(state)
        return state


def session_notes(session: SessionState | None) -> str:
    if session is None:
        return ""
    bits: list[str] = []
    if session.name:
        bits.append(f"Hitap: {session.name}")
    if session.product:
        bits.append(f"Ürün: {session.product}")
    if session.capacity:
        canonical = format_capacity(session.capacity)
        reminder = capacity_reminder(session.product, canonical)
        bits.append(
            "Kapasite (tam ölçek, basamak kırpma yasak): "
            f"{canonical}. Hatırlatma kalıbı: {reminder}"
        )
    if session.stock:
        bits.append(f"Stok (tam ölçek, basamak kırpma yasak): {session.stock}")
    if session.market:
        bits.append(f"Pazar: {session.market}")
    if not bits:
        return ""
    return "Konuşma notu: " + "; ".join(bits) + "."


def continue_intake(session: SessionState, user_text: str) -> str:
    """Ürün alındıktan sonra keşif sorusu yok; çözüm masaya konur.

    Yeni ürün gelince önceki ürün/stok/kapasite silinir; sektörler karışmaz.
    """
    text = (user_text or "").strip()
    switched = apply_utterance_slots(session, text)
    ask = session.last_ask
    product, capacity, stock = split_product_and_volume(text)

    if ask == "name":
        if looks_like_name(text) and not looks_like_product(text) and not looks_like_capacity(text):
            session.name = text
            session.last_ask = "product"
            return INTAKE_REPLY
        ask = "product"

    if ask == "product" or (ask is None and not session.product):
        if ask is None and not session.product:
            if not (looks_like_product(text) and not is_general_intake(text)):
                session.last_ask = "product"
                return INTAKE_REPLY
        if product:
            session.product = clean_product_name(product) or display_product(product)
        elif not capacity and not stock:
            session.product = clean_product_name(text) or display_product(text)
        if session.product and session.product != "ürününüz":
            session.last_ask = None
            return strip_echoed_query(
                text,
                director_options_reply(
                    session.product, session.capacity, session.stock
                ),
            )
        session.last_ask = "product"
        return INTAKE_REPLY

    if switched:
        session.last_ask = None
        return strip_echoed_query(
            text,
            director_options_reply(
                session.product or "ürün", session.capacity, session.stock
            ),
        )

    if (
        looks_like_stock(text)
        or looks_like_capacity(text)
        or (session.product and not session.capacity and capacity)
        or (session.product and not session.stock and stock)
    ):
        if stock:
            session.stock = stock
        if capacity:
            session.capacity = capacity or format_capacity(text)
        elif looks_like_capacity(text) and not looks_like_stock(text):
            session.capacity = format_capacity(text)
        session.last_ask = None
        return strip_echoed_query(
            text,
            director_options_reply(
                session.product or "ürün", session.capacity, session.stock
            ),
        )

    if session.product:
        session.last_ask = None
        return strip_echoed_query(
            text,
            director_options_reply(
                session.product, session.capacity, session.stock
            ),
        )

    if looks_like_product(text) and not is_general_intake(text):
        session.product = extract_product_slot(text) or display_product(text)
        if session.product and session.product != "ürününüz":
            session.last_ask = None
            return strip_echoed_query(
                text,
                director_options_reply(
                    session.product, session.capacity, session.stock
                ),
            )

    session.last_ask = "product"
    return INTAKE_REPLY


def in_progress(session: SessionState | None) -> bool:
    return bool(session and session.in_progress())
