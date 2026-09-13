"""Ürün odaklı vektör + sözcük eşleştirme."""

from __future__ import annotations

import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .session import (
    SessionState,
    _fold,
    product_kind,
    stale_session_product_conflict,
    strip_query_noise,
    utterance_product,
    wants_toy_event_notes,
)

# Sorgu kökü → gömme genişletmesi + kart filtresi
_PRODUCT_PROFILES: list[tuple[tuple[str, ...], str, frozenset[str]]] = [
    (
        ("etiket", "label", "labels", "dokuma"),
        "dokuma etiket woven labels clothing labels apparel labels "
        "garment labels brand labels damask labels HS 5807 ready-wear",
        frozenset(
            {
                "etiket",
                "label",
                "labels",
                "dokuma",
                "woven",
                "5807",
                "damask",
            }
        ),
    ),
    (
        ("3d", "3-d", "filament", "pla", "petg", "oyuncak", "yazici", "maket", "baski"),
        "boutique 3D printed toys miniatures PLA PETG FDM architectural models "
        "personalized gifts Etsy Amazon Handmade tabletop game designers "
        "gift shops B2C B2B2C",
        frozenset(
            {
                "3d",
                "filament",
                "pla",
                "petg",
                "fdm",
                "etsy",
                "handmade",
                "maket",
                "miniature",
                "minyatur",
                "oyuncak",
                "hediyelik",
            }
        ),
    ),
    (
        ("celik", "steel", "metal", "demir"),
        "hot rolled steel coils plates bars metal steel construction "
        "automotive machinery manufacturers steel traders B2B HS 7208 7210 7308",
        frozenset(
            {
                "celik",
                "steel",
                "metal",
                "7208",
                "7210",
                "7308",
                "coil",
                "insaat",
                "otomotiv",
                "construction",
                "automotive",
                "machinery",
                "trader",
            }
        ),
    ),
    (
        ("zeytin", "olive"),
        "zeytinyağı extra virgin olive oil HS 1509",
        frozenset({"zeytin", "olive", "1509"}),
    ),
    (
        ("pamuk", "cotton"),
        "raw cotton cotton yarn textile HS 5201",
        frozenset({"pamuk", "cotton", "5201"}),
    ),
    (
        ("hali", "carpet", "rug"),
        "hand-knotted wool carpets HS 5701",
        frozenset({"hali", "carpet", "rug", "5701", "hereke"}),
    ),
]

_WOVEN_CROSS = re.compile(
    r"(?i)(etiket|dokuma|woven|\b5807\b|damask|garment\s*label|clothing\s*label)"
)
_MASS_TOY_BRAND = re.compile(
    r"(?i)(ravensburger|\blego\b|hasbro|mattel|playmobil|"
    r"fisher[\s\-]?price|spin\s*master)"
)
_FAIR_OR_CERT = re.compile(
    r"(?i)("
    r"spielwarenmesse|spielwaren\s*messe|nuremberg\s*toy\s*fair|"
    r"t[uü]v\s*s[uü]d|tuv\s*sud"
    r")"
)

_STEEL_CHANNELS: tuple[dict[str, str], ...] = (
    {
        "slug": "construction",
        "name": "Avrupa inşaat yüklenicileri",
        "country": "DE",
        "product": "Structural steel coils plates HS 7208 construction",
        "description": "İnşaat ve yapısal çelik alıcıları. Sıcak haddelenmiş sac ve profil.",
    },
    {
        "slug": "automotive",
        "name": "Otomotiv üreticileri ve tedarikçileri",
        "country": "DE",
        "product": "Automotive steel sheet coil metal",
        "description": "Otomotiv OEM ve yan sanayi. Sac ve çelik girdi alıcıları.",
    },
    {
        "slug": "machinery",
        "name": "Makine imalatçıları",
        "country": "IT",
        "product": "Machinery manufacturing steel bars plates metal",
        "description": "Makine imalatçıları. Çelik çubuk, plaka ve yarı mamul B2B.",
    },
    {
        "slug": "traders",
        "name": "Çelik tüccarları",
        "country": "DE",
        "product": "Steel traders distributors coils metal B2B",
        "description": "Çelik tüccarları ve stokçular. Tonajlı B2B alım-satım.",
    },
)

_PRINT3D_CHANNELS: tuple[dict[str, str], ...] = (
    {
        "slug": "etsy",
        "name": "Etsy",
        "country": "US",
        "product": "Personalized 3D printed toys and gifts (Etsy B2C vitrin)",
        "description": "Kişiselleştirilmiş 3D oyuncak ve hediye pazarı. Ev tipi / butik yazıcı.",
    },
    {
        "slug": "amazon-handmade",
        "name": "Amazon Handmade",
        "country": "US",
        "product": "Boutique 3D printed products Amazon Handmade",
        "description": "Butik 3D baskı ve el yapımı ürün kanalı. Kitle üreticisi değil.",
    },
    {
        "slug": "local-gift",
        "name": "Yerel hediyelik eşya dükkanları",
        "country": "DE",
        "product": "Gift shop 3D printed souvenirs miniatures",
        "description": "Yerel hediyelik eşya ve butik vitrin. Küçük seri PLA/PETG.",
    },
    {
        "slug": "tabletop",
        "name": "Butik masaüstü oyun tasarımcıları",
        "country": "DE",
        "product": "Tabletop game miniatures 3D print PLA PETG",
        "description": "Butik masaüstü oyun tasarımcıları. Prototip ve kısa seri minyatür.",
    },
    {
        "slug": "architecture",
        "name": "Mimari maket büroları",
        "country": "DE",
        "product": "Architectural scale models 3D print",
        "description": "Mimari maket büroları. Hassas 3D baskı, kişiselleştirilmiş B2B2C.",
    },
)


def active_retrieval_product(session: SessionState | None, question: str) -> str:
    """Retrieval ürünü: current utterance, stale session slot'undan önce.

    Same-domain / generic devam → session.product kullanılabilir.
    Explicit product switch veya named-kind conflict → current question.
    """
    stale = (session.product.strip() if session and session.product else "")
    incoming = utterance_product(question)
    if stale and stale_session_product_conflict(stale, question):
        if incoming:
            return incoming
        cleaned = strip_query_noise(question or "")
        return cleaned or (question or "").strip()
    if stale:
        return stale
    if incoming:
        return incoming
    cleaned = strip_query_noise(question or "")
    return cleaned or (question or "").strip()


def _product_from_session(session: SessionState | None, question: str) -> str:
    return active_retrieval_product(session, question)


def expand_product_query(product: str) -> str:
    body = re.sub(r"\s+", " ", (product or "").strip())
    if not body:
        return body
    low = _fold(body)
    extras: list[str] = [body]
    for triggers, expansion, _needles in _PRODUCT_PROFILES:
        if any(trigger in low for trigger in triggers):
            extras.append(expansion)
    return " ".join(dict.fromkeys(extras))


def retrieval_query(
    session: SessionState | None,
    question: str = "",
    *,
    role: str = "buyer",
) -> str:
    """Gömülecek metin: ürün (+ eşanlam). Sohbet cümlesi ve kapasite yok."""
    product = _product_from_session(session, question)
    kind = product_kind(product)
    expanded = expand_product_query(product)
    if kind == "print3d":
        if role in {"buyer", "advisor"}:
            expanded = (
                f"{expanded} Etsy Amazon Handmade gift shops boutique tabletop "
                "designers architectural model makers personalized B2C B2B2C"
            ).strip()
        elif role == "supplier":
            expanded = f"{expanded} filament PLA PETG boutique 3D print studios".strip()
        return expanded or (question or "").strip()
    if kind == "steel":
        extra = (
            "steel coils plates construction automotive machinery "
            "manufacturers steel traders B2B importers"
        )
        if role == "supplier":
            extra = "steel mills exporters coils plates suppliers"
        return f"{expanded} {extra}".strip()
    if kind == "woven" and role == "buyer":
        expanded = f"{expanded} importers buyers apparel brands ready-wear manufacturers".strip()
    elif role == "buyer":
        expanded = f"{expanded} importers buyers manufacturers".strip()
    elif role == "supplier":
        expanded = f"{expanded} suppliers manufacturers exporters".strip()
    return expanded or (question or "").strip()


def product_needles(product: str) -> frozenset[str]:
    low = _fold(product or "")
    for triggers, _expansion, needles in _PRODUCT_PROFILES:
        if any(trigger in low for trigger in triggers):
            return needles
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", low)
        if token not in {"icin", "urun", "this", "with", "from"}
    }
    return frozenset(tokens)


def _item_blob(item: Any) -> str:
    if isinstance(item, dict):
        parts = [
            item.get("product_name") or "",
            item.get("description") or "",
            item.get("hs_code") or "",
            item.get("embedding_text") or "",
        ]
    else:
        parts = [
            getattr(item, "product_name", "") or "",
            getattr(item, "description", "") or "",
            getattr(item, "hs_code", "") or "",
        ]
    return _fold(" ".join(str(part) for part in parts))


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        parts = [
            item.get("organization_name") or "",
            item.get("product_name") or "",
            item.get("description") or "",
            item.get("hs_code") or "",
            item.get("embedding_text") or "",
        ]
    else:
        parts = [
            getattr(item, "organization_name", "") or "",
            getattr(item, "product_name", "") or "",
            getattr(item, "description", "") or "",
            getattr(item, "hs_code", "") or "",
        ]
    return " ".join(str(part) for part in parts)


_OLIVE_CROSS = re.compile(r"(?i)(zeytin|olive|\b1509\b)")
_STEEL_CORE = (
    "steel",
    "celik",
    "7208",
    "7210",
    "7308",
    "coil",
    "hot roll",
)
_STEEL_CHANNEL = (
    "insaat yuklen",
    "insaat yukle",
    "otomotiv uretic",
    "makine imalat",
    "celik tuccar",
    "steel trader",
)


def item_matches_product(item: Any, product: str) -> bool:
    kind = product_kind(product)
    blob = _item_blob(item)
    label = _fold(_item_text(item))
    if kind == "print3d":
        if _WOVEN_CROSS.search(blob) or _WOVEN_CROSS.search(label):
            return False
        if _MASS_TOY_BRAND.search(label):
            return False
        if _FAIR_OR_CERT.search(label):
            return False
        needles = product_needles(product)
        if not needles:
            return False
        return any(needle in blob or needle in label for needle in needles)
    if kind == "steel":
        if _WOVEN_CROSS.search(blob) or _OLIVE_CROSS.search(blob):
            return False
        if any(token in blob or token in label for token in _STEEL_CORE):
            return True
        return any(token in label for token in _STEEL_CHANNEL)
    if kind == "woven":
        if _OLIVE_CROSS.search(blob) or _OLIVE_CROSS.search(label):
            return False
    needles = product_needles(product)
    if not needles:
        return True
    return any(needle in blob for needle in needles)


def filter_matches_for_product(
    matches: list[Any],
    product: str,
    *,
    min_similarity: float = 0.0,
) -> list[Any]:
    """İlgisiz ürün kartlarını (zeytinyağı, pamuk, halı…) düşür."""
    kept: list[Any] = []
    for item in matches:
        score = (
            item.get("similarity")
            if isinstance(item, dict)
            else getattr(item, "similarity", 0)
        ) or 0
        if score < min_similarity:
            continue
        if item_matches_product(item, product):
            kept.append(item)
    return kept


def _match_label(item: Any) -> str:
    return _item_text(item)


def is_fair_or_cert(item: Any) -> bool:
    return bool(_FAIR_OR_CERT.search(_match_label(item)))


def is_mass_toy_brand(item: Any) -> bool:
    return bool(_MASS_TOY_BRAND.search(_match_label(item)))


def domain_channel_rows(kind: str) -> list[dict[str, Any]]:
    """Hedef kanal kartları (3D butik, çelik B2B). Sahte fabrika yok."""
    if kind == "print3d":
        channels, prefix = _PRINT3D_CHANNELS, "print3d"
    elif kind == "steel":
        channels, prefix = _STEEL_CHANNELS, "steel"
    else:
        return []
    rows: list[dict[str, Any]] = []
    for index, channel in enumerate(channels):
        rows.append(
            {
                "id": uuid5(NAMESPACE_URL, f"wtr:channel:{prefix}:{channel['slug']}"),
                "organization_id": None,
                "hs_code": "7208.10" if kind == "steel" else None,
                "product_name": channel["product"],
                "description": channel["description"],
                "origin_country": "TR",
                "destination_country": channel["country"],
                "similarity": round(0.92 - index * 0.01, 3),
                "organization_name": channel["name"],
                "has_volume": False,
                "has_contact": False,
            }
        )
    return rows


def default_domain_notes(kind: str) -> list[str]:
    if kind != "print3d":
        return []
    return [
        "Spielwarenmesse: oyuncak fuarıdır; alıcı firma değildir. Stand ve randevu için kullanın.",
        "TÜV SÜD: CE ve oyuncak güvenliği kapısıdır; alıcı değildir. Sertifika sürecinde devreye girer.",
    ]


def partition_product_matches(
    matches: list[Any],
    product: str,
) -> tuple[list[Any], list[str]]:
    """Alıcı kartları vs fuar/sertifika notları. Dev markalar 3D'de düşer."""
    kind = product_kind(product)
    buyers: list[Any] = []
    notes = default_domain_notes(kind) if wants_toy_event_notes(product) else []
    seen_notes = {_fold(note) for note in notes}
    for item in matches:
        label = _match_label(item)
        if is_fair_or_cert(item):
            if wants_toy_event_notes(product):
                note = (
                    f"{label.strip()[:80]}: etkinlik veya sertifika kurumudur; "
                    "alıcı firma olarak listelenmez."
                )
                key = _fold(note)
                if key not in seen_notes:
                    notes.append(note)
                    seen_notes.add(key)
            continue
        if kind == "print3d" and is_mass_toy_brand(item):
            continue
        buyers.append(item)
    return buyers, notes


def enrich_matches_for_product(
    matches: list[Any],
    product: str,
    *,
    role: str = "buyer",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Sektör süzgeci + 3D kanal kartları. Fuar/TÜV alıcı listesinde yok."""
    filtered = filter_matches_for_product(matches, product)
    buyers, notes = partition_product_matches(filtered, product)
    kind = product_kind(product)
    payloads: list[dict[str, Any]] = []
    for item in buyers:
        payloads.append(item if isinstance(item, dict) else item.model_dump(mode="json"))
    if kind in {"print3d", "steel"} and not (kind == "print3d" and role == "supplier"):
        payloads = merge_match_rows(domain_channel_rows(kind), payloads)
    return payloads, notes


def with_domain_notes(advice: str, notes: list[str], product: str | None = None) -> str:
    """Spielwarenmesse / TÜV notu yalnızca oyuncak veya 3D baskıda."""
    body = (advice or "").strip()
    if not wants_toy_event_notes(product) or not notes:
        return body
    if "Etkinlik / Sertifika notu" in body:
        return body
    block = "\n\nEtkinlik / Sertifika notu\n" + "\n".join(f"- {note}" for note in notes)
    return body + block


def effective_match_threshold(product: str, requested: float) -> float:
    """Çelik/jenerik: boş sonuç olmasın diye eşiği yumuşat. Dokuma sıkı kalsın."""
    kind = product_kind(product)
    asked = requested if requested and requested > 0 else 0.35
    if kind == "steel":
        return min(asked, 0.28)
    if kind == "woven":
        return max(asked, 0.42)
    return max(min(asked, 0.38), 0.30)


def sanitize_term(term: str) -> str:
    return re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\- ]+", "", term or "").strip()[:40]


async def lexical_trade_rows(
    supabase: Any,
    product: str,
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    """Ürün adını trade_items üzerinde ilike ile tara (vektöre ek hat)."""
    if supabase is None or not product.strip():
        return []
    needles = [sanitize_term(term) for term in product_needles(product)]
    needles = [term for term in needles if len(term) >= 4][:8]
    if product_kind(product) == "steel":
        needles = list(
            dict.fromkeys(
                needles
                + ["steel", "celik", "metal", "7208", "coil", "construction"]
            )
        )
    if not needles:
        raw = sanitize_term(product)
        if raw:
            needles = [raw]
    if not needles:
        return []
    clauses: list[str] = []
    for term in needles:
        clauses.append(f"product_name.ilike.%{term}%")
        clauses.append(f"description.ilike.%{term}%")
        clauses.append(f"hs_code.ilike.%{term}%")
    try:
        result = (
            await supabase.table("trade_items")
            .select(
                "id, organization_id, hs_code, product_name, description, "
                "origin_country, destination_country"
            )
            .or_(",".join(clauses[:24]))
            .limit(limit)
            .execute()
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for row in result.data or []:
        payload = dict(row)
        payload.setdefault("similarity", 0.9)
        rows.append(payload)
    return rows


def merge_match_rows(*batches: list[Any]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for batch in batches:
        for item in batch:
            payload = item if isinstance(item, dict) else item.model_dump(mode="json")
            key = str(payload.get("id") or "")
            if not key:
                continue
            prev = merged.get(key)
            if prev is None or float(payload.get("similarity") or 0) >= float(
                prev.get("similarity") or 0
            ):
                merged[key] = payload
    return sorted(
        merged.values(),
        key=lambda row: float(row.get("similarity") or 0),
        reverse=True,
    )


async def gather_match_rows(
    *,
    match,
    supabase: Any,
    embedding: list[float],
    product: str,
    match_count: int,
    match_threshold: float,
    organization_id: Any,
) -> list[Any]:
    """Vektör + anahtar kelime. Çelikte eşik düşük; boş sonuç olmasın."""
    threshold = effective_match_threshold(product, match_threshold)
    vector_rows = await match(
        supabase,
        embedding,
        match_count=max(match_count, 16),
        match_threshold=threshold,
        organization_id=organization_id,
    )
    lexical_rows = await lexical_trade_rows(supabase, product)
    merged = merge_match_rows(vector_rows, lexical_rows)
    kept = filter_matches_for_product(merged, product)
    if len(kept) < 2 and product_kind(product) == "steel":
        extra = await match(
            supabase,
            embedding,
            match_count=max(match_count, 24),
            match_threshold=0.22,
            organization_id=organization_id,
        )
        merged = merge_match_rows(merged, extra)
    return merged


CORPUS_TRADE_ITEMS = "trade_items"


async def retrieve_corpus(
    name: str,
    *,
    match=None,
    supabase: Any = None,
    embedding: list[float] | None = None,
    product: str = "",
    match_count: int = 8,
    match_threshold: float = 0.35,
    organization_id: Any = None,
) -> list[Any]:
    """Kiracı izolasyonlu retrieval. Şimdilik trade_items; ileride belgeler/kararlar."""
    if name != CORPUS_TRADE_ITEMS:
        return []
    if match is None or supabase is None or embedding is None:
        return []
    return await gather_match_rows(
        match=match,
        supabase=supabase,
        embedding=embedding,
        product=product,
        match_count=match_count,
        match_threshold=match_threshold,
        organization_id=organization_id,
    )
