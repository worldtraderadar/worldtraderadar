"""Qwen shadow eval. Production cevap, TTS, memory ve kullanıcıya dokunmaz."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from llm import (
    ollama_base_url,
    ollama_chat_payload,
    reasoning_model,
    shadow_model,
    thinking_from_ollama,
    user_content_from_ollama,
    user_visible_text,
)

log = logging.getLogger("wtr.shadow")

_LAST_SHADOW: dict[str, Any] = {}
_CATEGORY = (
    ("germany_france", ("almanya", "fransa")),
    ("competitor_pressure", ("rakip",)),
    ("price_pressure", ("fiyat bask",)),
    ("raise_price", ("zam",)),
    ("discount", ("indirim",)),
    ("margin_drop", ("marjımız düş", "marjimiz dus")),
    ("sales_drop", ("satışlar düş", "satislar dus")),
    ("new_buyer_priority", ("yeni müşteri", "yeni musteri")),
    ("buyer_abc", ("öncelik vermeli", "oncelik vermeli")),
    ("supplier_choice", ("tedarikçi", "tedarikci")),
    ("capacity", ("en büyük pazar", "en buyuk pazar")),
    ("volume_vs_margin", ("yüksek hacim", "yuksek hacim")),
    ("new_country", ("hiç müşterim yok", "hic musterim yok")),
    ("keep_customer", ("bırakıp", "birakip")),
    ("collection_risk", ("ödemesi", "odemesi")),
    ("moq", ("moq",)),
    ("incoterms", ("fob", "cif")),
    ("today_plan", ("bugün ne", "bugun ne")),
    ("uncertain_data", ("net veri yok",)),
    ("market_choice", ("almanya'da mı", "almanya'da mi")),
)


def last_shadow() -> dict[str, Any]:
    return dict(_LAST_SHADOW)


def reset_shadow() -> None:
    _LAST_SHADOW.clear()


def input_category(question: str) -> str:
    body = (question or "").casefold()
    for name, needles in _CATEGORY:
        if all(n.casefold() in body for n in needles):
            return name
    return "other"


def _ns_to_s(value: Any) -> float | None:
    try:
        n = float(value or 0)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return round(n / 1e9, 3)


async def run_shadow_eval(
    *,
    http,
    question: str,
    user: str,
    system: str,
    production_path: str,
    production_text: str,
) -> dict[str, Any]:
    model = shadow_model()
    result: dict[str, Any] = {
        "enabled": bool(model),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "task": "commercial",
        "input_category": input_category(question),
        "think": False,
        "num_predict": None,
        "latency_s": None,
        "eval_count": None,
        "tokens_per_sec": None,
        "done_reason": "",
        "content_length": 0,
        "thinking_length": 0,
        "prompt_eval_s": None,
        "gen_s": None,
        "load_s": None,
        "validator_result": "",
        "cmo_score": None,
        "BRIEF_ECHO": False,
        "ENGLISH_LEAKAGE": False,
        "FALSE_CERTAINTY": False,
        "UNSUPPORTED_FACT": False,
        "MEMORY_DISHONESTY": False,
        "PROMPT_LEAK": False,
        "final_path": production_path,
        "user_visible": "",
        "leaked_thinking": False,
        "production_path": production_path,
        "error": "",
    }
    if not model or http is None:
        _LAST_SHADOW.update(result)
        return result
    if model.casefold() == reasoning_model().casefold():
        result["error"] = "shadow_equals_production"
        _LAST_SHADOW.update(result)
        return result
    payload = ollama_chat_payload(
        model,
        [
            {"role": "system", "content": system or ""},
            {"role": "user", "content": user},
        ],
        think=False,
        options={"temperature": 0.3},
        keep_alive="0",
    )
    result["num_predict"] = (payload.get("options") or {}).get("num_predict")
    result["think"] = bool(payload.get("think"))
    t0 = time.perf_counter()
    try:
        response = await http.post(
            f"{ollama_base_url()}/api/chat",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        thinking = thinking_from_ollama(data)
        content = user_visible_text(user_content_from_ollama(data), thinking)
        elapsed = time.perf_counter() - t0
        eval_count = data.get("eval_count") or 0
        eval_ns = data.get("eval_duration") or 0
        tps = (eval_count / (eval_ns / 1e9)) if eval_ns else None
        from agents.evaluator import score_cmo_v51
        from agents.validator import make_speakable, validate_response

        spoken = make_speakable(content)
        verdict = validate_response(spoken, question=question)
        scored = score_cmo_v51(spoken, question=question)
        result.update(
            {
                "latency_s": round(elapsed, 3),
                "eval_count": eval_count,
                "tokens_per_sec": round(tps, 2) if tps else None,
                "done_reason": data.get("done_reason") or "",
                "content_length": len(content),
                "thinking_length": len(thinking),
                "prompt_eval_s": _ns_to_s(data.get("prompt_eval_duration")),
                "gen_s": _ns_to_s(eval_ns),
                "load_s": _ns_to_s(data.get("load_duration")),
                "validator_result": verdict.status,
                "cmo_score": scored.total,
                "BRIEF_ECHO": scored.brief_echo,
                "ENGLISH_LEAKAGE": scored.english_leakage,
                "FALSE_CERTAINTY": scored.false_certainty,
                "UNSUPPORTED_FACT": scored.unsupported_fact,
                "MEMORY_DISHONESTY": scored.memory_dishonesty,
                "PROMPT_LEAK": scored.prompt_leak,
                "final_path": f"SHADOW_{verdict.status}",
                "user_visible": content,
                "leaked_thinking": False,
                "think": payload.get("think"),
            }
        )
        if thinking and thinking.strip() and thinking.strip() in (content or ""):
            result["leaked_thinking"] = True
        log.info(
            "shadow model=%s cat=%s content_len=%s thinking_len=%s done=%s "
            "validator=%s cmo=%s prod_path=%s latency=%s",
            model,
            result["input_category"],
            len(content),
            len(thinking),
            result["done_reason"],
            verdict.status,
            scored.total,
            production_path,
            result["latency_s"],
        )
    except Exception as exc:
        result["latency_s"] = round(time.perf_counter() - t0, 3)
        result["error"] = repr(exc)
        log.info("shadow skipped/failed: %s", exc)
    # production_text bilinçli olarak shadow çıktısıyla değiştirilmez.
    del production_text
    _LAST_SHADOW.update(result)
    return result


def schedule_shadow(
    *,
    http,
    question: str,
    user: str,
    system: str,
    production_path: str,
    production_text: str,
) -> None:
    if not shadow_model():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run() -> None:
        await run_shadow_eval(
            http=http,
            question=question,
            user=user,
            system=system,
            production_path=production_path,
            production_text=production_text,
        )

    loop.create_task(_run())
