"""Kalıcı danışman belleği. SessionState kısa vade; bu tablo uzun vade."""

from __future__ import annotations

import logging
import re
from typing import Any

from .session import SessionState

log = logging.getLogger("wtr.memory")

_SKIP = re.compile(
    r"(?i)^(merhaba|selam|teşekkür|tesekkur|tamam|evet|hayır|hayir|ok|peki)[\s!.]*$"
)
_KEEP = re.compile(
    r"(?i)("
    r"şirketimiz|sirketimiz|firmamız|firmamiz|"
    r"karar\s*(verdik|aldık|aldik)|"
    r"minimum\s*sipari[sş]|moq|"
    r"ihra[cç]\s*ediyor|"
    r"hedef\s*pazar|"
    r"hedef\s*m[uü][sş]teri|"
    r"distrib[uü]t[oö]r|"
    r"kapasite|"
    r"marj|"
    r"ayl[iı]k|"
    r"fiyat\s*(politik|yakla[sş][iı]m)|"
    r"sat[iı][sş]\s*kanal"
    r")"
)


def should_persist(text: str) -> bool:
    body = (text or "").strip()
    if len(body) < 24 or _SKIP.match(body):
        return False
    return bool(_KEEP.search(body))


def memory_kind(text: str) -> str:
    """Karar kalitesini artıracak kategori. Her cümle fact değildir."""
    low = (text or "").casefold()
    if re.search(r"(?i)(marj|minimum\s*kabul|fiyat\s*politik)", low):
        return "margin_constraint" if "marj" in low else "pricing_constraint"
    if re.search(r"(?i)kapasite", low):
        return "capacity"
    if re.search(r"(?i)(moq|minimum\s*sipari[sş]|stok)", low):
        return "pricing_constraint"
    if re.search(r"(?i)hedef\s*m[uü][sş]teri", low):
        return "target_customer"
    if re.search(r"(?i)hedef\s*pazar", low):
        return "target_market"
    if re.search(r"(?i)(\bürün\b|zeytinya|dokuma|çelik)", low) and "karar" not in low:
        return "product"
    if re.search(r"(?i)(sat[iı][sş]\s*[oö]ncel|kanal)", low):
        return "sales_priority"
    if re.search(r"(?i)karar[ıi]?\s*(verdik|aldık|aldik)", low):
        return "previous_decision"
    if "karar" in low or "strateji" in low:
        return "strategic_decision"
    if re.search(r"(?i)(m[uü][sş]teri|lead|al[iı]c[iı])", low):
        return "customer_context"
    if re.search(r"(?i)(şirket|firma|kısıt)", low):
        return "company_constraint"
    return "company_constraint"


async def search_memories(
    supabase: Any,
    *,
    account_slug: str,
    query: str,
    session: SessionState | None = None,
    limit: int = 8,
) -> list[str]:
    slug = (account_slug or "demo").strip() or "demo"
    found: list[str] = []
    if supabase is not None:
        try:
            result = (
                await supabase.table("consultant_memories")
                .select("content, kind, created_at")
                .eq("account_slug", slug)
                .order("created_at", desc=True)
                .limit(40)
                .execute()
            )
            rows = result.data or []
            needle = (query or "").casefold()
            tokens = [tok for tok in re.findall(r"[a-zA-Zğüşöçıİ0-9]{4,}", needle) if tok]
            for row in rows:
                content = str(row.get("content") or "").strip()
                if not content:
                    continue
                blob = content.casefold()
                if not tokens or any(tok in blob for tok in tokens):
                    found.append(content)
                if len(found) >= limit:
                    break
            if not found:
                found = [
                    str(row.get("content") or "").strip()
                    for row in rows[:limit]
                    if str(row.get("content") or "").strip()
                ]
        except Exception as exc:
            log.debug("consultant_memories okunamadı: %s", exc)
    if session is not None:
        for item in getattr(session, "memories", []) or []:
            if item and item not in found:
                found.append(item)
    return found[:limit]


async def persist_memory(
    supabase: Any,
    *,
    account_slug: str,
    content: str,
    session: SessionState | None = None,
    kind: str | None = None,
) -> None:
    body = (content or "").strip()
    if not should_persist(body):
        return
    label = kind or memory_kind(body)
    slug = (account_slug or "demo").strip() or "demo"
    if session is not None:
        bucket = getattr(session, "memories", None)
        if bucket is None:
            session.memories = []
            bucket = session.memories
        if body not in bucket:
            bucket.append(body)
            if len(bucket) > 24:
                del bucket[:-24]
    if supabase is None:
        return
    try:
        await supabase.table("consultant_memories").insert(
            {
                "account_slug": slug,
                "session_id": session.session_id if session else None,
                "kind": label if label in ("fact", "decision", "preference", "project") else "fact",
                "content": body[:2000],
                "metadata": {},
            }
        ).execute()
    except Exception as exc:
        log.debug("consultant_memories yazılamadı: %s", exc)


async def persist_from_turn(
    supabase: Any,
    *,
    account_slug: str,
    user_text: str,
    assistant_text: str,
    session: SessionState | None = None,
) -> None:
    for blob in (user_text,):
        await persist_memory(
            supabase,
            account_slug=account_slug,
            content=blob,
            session=session,
        )
    if session and session.product and should_persist(user_text):
        bits = [f"Ürün: {session.product}"]
        if session.capacity:
            bits.append(f"Kapasite: {session.capacity}")
        if session.market:
            bits.append(f"Pazar: {session.market}")
        await persist_memory(
            supabase,
            account_slug=account_slug,
            content="; ".join(bits),
            session=session,
            kind="fact",
        )
    _ = assistant_text
