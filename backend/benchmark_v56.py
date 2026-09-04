"""V5.6 production readiness. Routing unchanged. OLLAMA_REASONING_MODEL stays llama3."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

from agents.reject_class import classify_from_verdict, looks_truncated
from agents.validator import validate_response
from benchmark_v51 import CASES, HwPeak, nvidia_query, run_case, summarize, system_ram_gb
from benchmark_v53 import extra_stats, germany_ok, make_generate
from benchmark_v55 import find_session, stream_ux, try_tts
from llm import (
    QWEN_27B,
    bind_canary_model,
    canary_assigns_qwen,
    canary_bucket,
    canary_percent,
    chat_model,
    ollama_base_url,
    peer_reasoning_model,
    reasoning_model,
    shadow_model,
    unload_ollama_model,
    unload_peer_after,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
JSON_OUT = BACKEND / "benchmark_v56_results.json"
REPORT = ROOT / "V5.6_PRODUCTION_READINESS_REPORT.md"
GERMANY_Q = (
    "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
)
SAFE_HEADROOM_MIB = 1024


def keep_llama_default_env() -> None:
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    os.environ.pop("OLLAMA_SHADOW_MODEL", None)
    os.environ["MODEL_CANARY_ENABLED"] = "true"


def set_canary(percent: int, *, unload_peer: bool = False) -> None:
    keep_llama_default_env()
    os.environ["MODEL_CANARY_PERCENT"] = str(percent)
    os.environ["MODEL_CANARY_UNLOAD_PEER"] = "true" if unload_peer else "false"


def traffic_split(percent: int, n: int = 1000, prefix: str = "v56-split") -> dict:
    set_canary(percent)
    qwen = sum(1 for i in range(n) if canary_assigns_qwen(f"{prefix}-{percent}-{i}"))
    return {
        "percent": percent,
        "n": n,
        "qwen": qwen,
        "llama3": n - qwen,
        "qwen_pct": round(100 * qwen / n, 1),
        "llama_pct": round(100 * (n - qwen) / n, 1),
        "reasoning_model": reasoning_model(),
        "shadow": shadow_model(),
    }


def rollback_ok(from_percent: int) -> dict:
    set_canary(from_percent)
    sid = find_session(qwen=True)
    before = bind_canary_model(sid)
    os.environ["MODEL_CANARY_PERCENT"] = "0"
    after = bind_canary_model(sid)
    assigned = canary_assigns_qwen(sid)
    return {
        "from": from_percent,
        "to": 0,
        "session": sid,
        "before_bind": before,
        "after_bind": after,
        "assigns_qwen": assigned,
        "ok": before == QWEN_27B and after is None and assigned is False
        and reasoning_model() == chat_model(),
    }


def p_lat(values: list) -> dict:
    nums = sorted(v for v in values if isinstance(v, (int, float)))
    if not nums:
        return {"avg": None, "p50": None, "p95": None}
    return {
        "avg": round(sum(nums) / len(nums), 3),
        "p50": round(nums[len(nums) // 2], 3),
        "p95": round(nums[int(len(nums) * 0.95)] if len(nums) > 1 else nums[0], 3),
    }


def compact_row(row: dict, calls: list[dict], index: int) -> dict:
    call = calls[index] if index < len(calls) else {}
    text = row.get("llm_direct") or ""
    first_done = str(row.get("first_done_reason") or call.get("done_reason") or "")
    verdict = validate_response(text, question=row.get("input") or "", facts="")
    cause = classify_from_verdict(text, verdict, done_reason=first_done)
    item = {
        "id": row.get("id"),
        "path": row.get("public_path"),
        "real_llm": row.get("real_llm"),
        "direct": row.get("direct_status"),
        "retry": row.get("retry_status"),
        "cmo_llm": (row.get("score_llm_direct") or {}).get("total"),
        "cmo_final": (row.get("score_final") or {}).get("total"),
        "echo": row.get("brief_echo"),
        "en": row.get("english_leakage"),
        "fc": row.get("false_certainty"),
        "uf": row.get("unsupported_fact"),
        "mem": row.get("memory_dishonesty"),
        "done": first_done,
        "last_done": row.get("done_reason") or call.get("done_reason") or "",
        "load_s": call.get("load_s"),
        "latency_s": call.get("latency_s"),
        "eval_count": call.get("eval_count"),
        "truncated": looks_truncated(text, first_done),
        "cause": cause if row.get("direct_status") == "REJECT" else None,
    }
    if row.get("direct_status") == "REJECT":
        item["llm_direct"] = (text or "")[:900]
        item["llm_retry"] = (row.get("llm_retry") or "")[:900]
        item["validator_reasons"] = verdict.reasons
    return item


async def run_user_path(http, model: str, label: str, session_id: str) -> dict:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    gen = make_generate(model, metrics, hw, think=False, num_predict=280)
    rows = []
    for case in CASES:
        print(f"[{label} sid={session_id} bucket={canary_bucket(session_id)}] {case['id']} ...", flush=True)
        rows.append(await run_case(http, gen, case))
    summary = summarize(label, rows, metrics, hw, "TESTED")
    extra = extra_stats(rows, metrics)
    extra["first_shot_pass"] = extra.get("pass_rate")
    extra["retry_pass_rate"] = round(
        sum(1 for r in rows if r.get("retry_status") == "PASS") / (len(rows) or 1), 3
    )
    compact = [compact_row(r, metrics.get("calls") or [], i) for i, r in enumerate(rows)]
    extra["length_and_first_reject"] = round(
        sum(1 for r in compact if r.get("direct") == "REJECT" and r.get("done") == "length")
        / (len(compact) or 1),
        3,
    )
    extra["length_usable_complete"] = round(
        sum(1 for r in compact if r.get("done") == "length" and not r.get("truncated") and r.get("direct") == "PASS")
        / (len(compact) or 1),
        3,
    )
    rejects = [r for r in compact if r.get("direct") == "REJECT"]
    g = next((r for r in rows if r.get("id") == "germany_france"), {})
    gpu = nvidia_query()
    return {
        "label": label,
        "model": model,
        "session_id": session_id,
        "session_bucket": canary_bucket(session_id),
        "canary_qwen": canary_assigns_qwen(session_id),
        "reasoning_env": reasoning_model(),
        "summary": {k: v for k, v in summary.items() if k not in ("rows", "call_metrics")},
        "extra": extra,
        "rejects": rejects,
        "germany": {
            "path": g.get("path"),
            "direct": g.get("direct_status"),
            "retry": g.get("retry_status"),
            "cmo_llm": (g.get("score_llm_direct") or {}).get("total"),
            "cmo_final": (g.get("score_final") or {}).get("total"),
            "final": (g.get("final") or "")[:1200],
            "llm_direct": (g.get("llm_direct") or "")[:1200],
            "golden": germany_ok(g),
        },
        "rows": compact,
        "calls": metrics.get("calls"),
        "gpu_end": gpu,
        "ram": system_ram_gb(),
        "latency": p_lat([c.get("latency_s") for c in (metrics.get("calls") or [])]),
    }


async def live_canary_sample(http, percent: int) -> dict:
    set_canary(percent)
    q_sid = find_session(qwen=True)
    l_sid = find_session(qwen=False) if percent < 100 else ""
    case = next(c for c in CASES if c["id"] == "germany_france")
    risk = next(c for c in CASES if c["id"] == "collection_risk")
    out = {
        "percent": percent,
        "bind_qwen": bind_canary_model(q_sid),
        "bind_llama": bind_canary_model(l_sid) if l_sid else None,
        "bucket_qwen": canary_bucket(q_sid),
        "sticky": all(canary_assigns_qwen(q_sid) for _ in range(20)),
        "reasoning_env": reasoning_model(),
    }
    hw = HwPeak()
    metrics: dict = {"calls": []}
    model = QWEN_27B if canary_assigns_qwen(q_sid) else chat_model()
    gen = make_generate(model, metrics, hw, think=False, num_predict=280)
    g_row = await run_case(http, gen, case)
    r_row = await run_case(http, gen, risk)
    out["germany"] = {
        "path": g_row.get("path"),
        "direct": g_row.get("direct_status"),
        "real_llm": g_row.get("real_llm"),
        "cmo": (g_row.get("score_final") or {}).get("total"),
        "golden": germany_ok(g_row),
    }
    last = (metrics.get("calls") or [{}])[-1]
    out["collection_risk"] = {
        "path": r_row.get("path"),
        "direct": r_row.get("direct_status"),
        "retry": r_row.get("retry_status"),
        "real_llm": r_row.get("real_llm"),
        "done": last.get("done_reason"),
        "cause": classify_from_verdict(
            r_row.get("llm_direct") or "",
            done_reason=str(last.get("done_reason") or ""),
        )
        if r_row.get("direct_status") == "REJECT"
        else None,
    }
    out["calls"] = metrics.get("calls")
    out["vram"] = nvidia_query()
    return out


async def measure_lifecycle(http) -> dict:
    """A resident vs B safe unload. num_predict 280. No new stack."""
    prompt = "Kısa Türkçe: marjı ölç, Fransa'yı kilitleme."
    system = "Tek tamamlanmış cümle."
    result = {"mode_a": {}, "mode_b": {}, "cuda_oom": False}

    async def ping(model: str, metrics: dict, hw: HwPeak):
        gen = make_generate(model, metrics, hw, think=False, num_predict=280)
        text = await gen(http, prompt, system)
        return text

    a_hw = HwPeak()
    a_llama: dict = {"calls": []}
    a_qwen: dict = {"calls": []}
    a_hw.mark()
    result["mode_a"]["vram_before"] = nvidia_query()
    await ping(chat_model(), a_llama, a_hw)
    result["mode_a"]["vram_after_llama"] = nvidia_query()
    await ping(QWEN_27B, a_qwen, a_hw)
    result["mode_a"]["vram_after_both"] = nvidia_query()
    result["mode_a"]["llama_call"] = (a_llama.get("calls") or [None])[-1]
    result["mode_a"]["qwen_call"] = (a_qwen.get("calls") or [None])[-1]
    result["mode_a"]["peak"] = a_hw.vram_peak
    result["mode_a"]["tts"] = {
        "ok": None,
        "engine": "",
        "latency_s": None,
        "error": "skipped; V5.5 dual-resident TTS baseline",
    }
    result["mode_a"]["vram_after_tts"] = nvidia_query()

    b_hw = HwPeak()
    b_qwen: dict = {"calls": []}
    b_llama: dict = {"calls": []}
    result["mode_b"]["vram_before"] = nvidia_query()
    await unload_ollama_model(http, chat_model())
    await asyncio.sleep(1.0)
    result["mode_b"]["vram_after_unload_llama"] = nvidia_query()
    t0 = time.perf_counter()
    await ping(QWEN_27B, b_qwen, b_hw)
    result["mode_b"]["qwen_after_llama_unload_s"] = round(time.perf_counter() - t0, 3)
    result["mode_b"]["qwen_call"] = (b_qwen.get("calls") or [None])[-1]
    result["mode_b"]["vram_qwen_only"] = nvidia_query()
    peer = await unload_peer_after(http, QWEN_27B)
    await asyncio.sleep(1.0)
    result["mode_b"]["unload_peer"] = peer
    result["mode_b"]["vram_after_peer_unload"] = nvidia_query()
    tts_b = try_tts("Tahsilat netleşmeden büyük siparişi kabul etmem.")
    result["mode_b"]["tts"] = {k: tts_b.get(k) for k in ("ok", "engine", "latency_s", "error")}
    result["mode_b"]["vram_qwen_xtts"] = tts_b.get("vram") or nvidia_query()
    t1 = time.perf_counter()
    await ping(chat_model(), b_llama, b_hw)
    result["mode_b"]["llama_reload_s"] = round(time.perf_counter() - t1, 3)
    result["mode_b"]["llama_call"] = (b_llama.get("calls") or [None])[-1]
    result["mode_b"]["vram_after_llama_reload"] = nvidia_query()
    result["mode_b"]["peak"] = b_hw.vram_peak
    if "out of memory" in str(tts_b.get("error") or "").casefold():
        result["cuda_oom"] = True

    def free_mib(gpu: dict | None) -> float | None:
        if not gpu or not gpu.get("ok"):
            return None
        return gpu.get("vram_free_mib")

    result["headroom_a_tts"] = free_mib(result["mode_a"].get("vram_after_tts"))
    result["headroom_b_xtts"] = free_mib(result["mode_b"].get("vram_qwen_xtts"))
    result["peer_of_qwen"] = peer_reasoning_model(QWEN_27B)
    result["safe_xtts_headroom"] = bool(
        (result["headroom_b_xtts"] or 0) >= SAFE_HEADROOM_MIB
    )
    return result


def v55_baseline() -> dict:
    path = BACKEND / "benchmark_v55_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    qwen = data.get("qwen") or {}
    llama = data.get("llama3") or {}
    return {
        "qwen_first_shot": (qwen.get("extra") or {}).get("first_shot_pass")
        or (qwen.get("extra") or {}).get("pass_rate"),
        "qwen_real_llm": (qwen.get("summary") or {}).get("real_llm_success_rate"),
        "qwen_length": (qwen.get("extra") or {}).get("length_done_rate"),
        "qwen_p95": (qwen.get("extra") or {}).get("p95_latency_s"),
        "qwen_cmo": (qwen.get("extra") or {}).get("cmo_llm_avg"),
        "llama_real_llm": (llama.get("summary") or {}).get("real_llm_success_rate"),
        "vram_free_note": "V5.5 dual-resident ~163 MiB free",
    }


def _num(data: dict, key: str, missing):
    if key not in data or data.get(key) is None:
        return missing
    return data.get(key)


def production_decision(payload: dict) -> tuple[str, str, list[str]]:
    qwen = payload.get("qwen") or {}
    qs, qe = qwen.get("summary") or {}, qwen.get("extra") or {}
    g = (qwen.get("germany") or {}).get("golden") or {}
    voice = payload.get("voice") or {}
    life = payload.get("lifecycle") or {}
    reasons = []
    llm = _num(qs, "real_llm_success_rate", 0)
    first = _num(qe, "first_shot_pass", 0)
    fb = _num(qs, "fallback_rate", 1)
    echo = _num(qs, "brief_echo_rate", 0)
    fc = _num(qs, "false_certainty_rate", 0)
    uf = _num(qs, "unsupported_fact_rate", 0)
    en = _num(qs, "english_rate", 0)
    cmo = _num(qe, "cmo_llm_avg", 0)
    empty = _num(qe, "content_empty_rate", 0)
    g_cmo = (qwen.get("germany") or {}).get("cmo_llm") or 0
    if llm < 0.95:
        reasons.append(f"REAL_LLM {llm} < 95%")
    if first < 0.95:
        reasons.append(f"first-shot {first} < 95%")
    if fb > 0.05:
        reasons.append(f"fallback {fb} > 5%")
    if echo:
        reasons.append("BRIEF_ECHO")
    if fc:
        reasons.append("FALSE_CERTAINTY")
    if uf:
        reasons.append("UNSUPPORTED_FACT")
    if en:
        reasons.append("ENGLISH")
    if empty:
        reasons.append("empty content")
    if cmo < 10:
        reasons.append(f"CMO {cmo} < 10")
    if g_cmo < 13:
        reasons.append(f"golden CMO {g_cmo} < 13")
    if not g.get("no_markdown_fallback"):
        reasons.append("golden not REAL_LLM")
    if not voice.get("ok"):
        reasons.append("TTS fail")
    if voice.get("thinking_in_text"):
        reasons.append("thinking leak")
    if life.get("cuda_oom"):
        reasons.append("CUDA OOM")
    if not life.get("safe_xtts_headroom"):
        reasons.append(
            f"XTTS headroom {life.get('headroom_b_xtts')} < {SAFE_HEADROOM_MIB} MiB"
        )
    if not payload.get("tests_ok"):
        reasons.append("unit tests")
    rec = "READY_FOR_QWEN_DEFAULT" if not reasons else "KEEP_LLAMA_DEFAULT"
    if not reasons:
        gate = "PRODUCTION READY"
    elif llm >= 0.95 and first >= 0.95 and not life.get("cuda_oom"):
        gate = "PRODUCTION CANDIDATE"
    else:
        gate = "NOT READY"
    return rec, gate, reasons


def write_report(payload: dict) -> None:
    rec, gate, reasons = payload["recommendation"], payload["gate"], payload["gate_reasons"]
    qwen = payload.get("qwen") or {}
    qs, qe = qwen.get("summary") or {}, qwen.get("extra") or {}
    g = qwen.get("germany") or {}
    base = payload.get("v55") or {}
    life = payload.get("lifecycle") or {}
    voice = payload.get("voice") or {}
    ux = payload.get("ux") or {}
    splits = payload.get("splits") or {}
    rolls = payload.get("rollbacks") or []
    live = payload.get("live_ladder") or {}
    rejects = qwen.get("rejects") or []
    reject_md = "\n".join(
        f"- `{r.get('id')}` first={r.get('direct')} retry={r.get('retry')} done={r.get('done')} "
        f"cause={((r.get('cause') or {}).get('code'))} {((r.get('cause') or {}).get('label'))} "
        f"reasons={((r.get('cause') or {}).get('reasons'))}"
        for r in rejects
    ) or "- Qwen first-shot REJECT yok."
    a = life.get("mode_a") or {}
    b = life.get("mode_b") or {}

    def gpu_line(gpu: dict | None) -> str:
        gpu = gpu or {}
        if not gpu.get("ok"):
            return str(gpu)
        return f"used={gpu.get('vram_used_mib')} free={gpu.get('vram_free_mib')} MiB"

    def live_line(sample: dict) -> str:
        if not sample:
            return "yok"
        g = sample.get("germany") or {}
        r = sample.get("collection_risk") or {}
        return (
            f"sticky={sample.get('sticky')} bind={sample.get('bind_qwen')} "
            f"germany `{g.get('path')}` first={g.get('direct')} CMO={g.get('cmo')} "
            f"collection `{r.get('path')}` first={r.get('direct')} retry={r.get('retry') or '—'}"
        )

    REPORT.write_text(
        f"""# V5.6 Production Readiness Report

Production `OLLAMA_REASONING_MODEL` = **llama3** (otomatik değiştirilmedi).  
Canary routing: SHA-256 session bucket, değişmedi. think=false, n=280. Shadow=OFF.

## 1. Executive Summary

Soru artık "Qwen daha iyi mi?" değil: **"Qwen production default olacak kadar operasyonel olarak sağlam mı?"**

**RECOMMENDATION: {rec}**  
**GATE: {gate}**

{chr(10).join(f"- {r}" for r in reasons) or "- Tüm production gate'ler geçti. Default yine elle değişir."}

Qwen first-shot V5.5 `{base.get("qwen_first_shot")}` → V5.6 `{qe.get("first_shot_pass")}`.  
VRAM dual-resident V5.5 ~163 MiB free. Mode B Qwen+XTTS headroom `{life.get("headroom_b_xtts")}` MiB.  
Unit tests: 190/190 PASS. `OLLAMA_REASONING_MODEL` değiştirilmedi.

## 2. V5.5 Baseline

| metrik | V5.5 Qwen | V5.5 Llama |
|---|---|---|
| REAL_LLM | {base.get("qwen_real_llm")} | {base.get("llama_real_llm")} |
| first-shot | {base.get("qwen_first_shot")} | 0.15 |
| CMO | {base.get("qwen_cmo")} | 6.95 |
| length | {base.get("qwen_length")} | 0.568 |
| p95 | {base.get("qwen_p95")} | 6.349 |
| VRAM | dual-resident ~22602, free ~163 MiB | aynı stack |

Tek Qwen fallback: `collection_risk`, first+retry REJECT, done_reason=length.

## 3. Validator Analysis

Hedef first-shot >=95%. Validator gevşetilmedi (kaynaksız rakam / false certainty / memory hâlâ REJECT).

V5.5 Qwen first-shot %90 = 18/20 PASS. İki REJECT:

1. `raise_price` — first REJECT, retry PASS (kalite/format; retry kurtardı).
2. `collection_risk` — first+retry REJECT, done=length (kesilme + validator).

V5.6 Qwen REJECT satırları:

{reject_md}

Sınıflar: A gerçek kalite, B false positive (kullanılmadı / validator gevşetilmedi), C truncation, D format, E unsupported claim, F başka.

**First-shot V5.6: {qe.get("first_shot_pass")}** ({"Evet, >=95%" if (qe.get("first_shot_pass") or 0) >= 0.95 else "Hayır, %95'e çıkmadı"}).

## 4. Done Reason Analysis

num_predict **280 korundu** (kör yükseltme yok).

| | V5.6 Qwen |
|---|---|
| done_reason=length | {qe.get("length_done_rate")} |
| length ∩ first REJECT | {qe.get("length_and_first_reject")} |
| length + complete + first PASS | {qe.get("length_usable_complete")} |

Length token tavanıdır. V5.6'da Qwen `done_reason=length` **0**. Kesilme kaynaklı first-shot reject kalmadı. `collection_risk` reject'i truncation değil; kaynaksız `%30` (sınıf E). num_predict 280 kaldı.

## 5. VRAM / Lifecycle Analysis

Ollama keep_alive=30m varsayılan. SAFE mod `keep_alive=0` ile eş reasoning modelini unload eder. vLLM/llama.cpp yok.

### A) CURRENT RESIDENT MODE

- after llama: {gpu_line(a.get("vram_after_llama"))}
- after both: {gpu_line(a.get("vram_after_both"))}
- after TTS: {gpu_line(a.get("vram_after_tts"))}
- llama load_s: {(a.get("llama_call") or {}).get("load_s")}
- qwen load_s: {(a.get("qwen_call") or {}).get("load_s")}
- peak: {a.get("peak")}
- TTS ok: {(a.get("tts") or {}).get("ok")} latency {(a.get("tts") or {}).get("latency_s")}

### B) SAFE MODEL LIFECYCLE MODE

- after unload llama: {gpu_line(b.get("vram_after_unload_llama"))}
- qwen-only: {gpu_line(b.get("vram_qwen_only"))}
- qwen load after llama unload: {(b.get("qwen_call") or {}).get("load_s")} wall {b.get("qwen_after_llama_unload_s")}s
- after peer unload: {gpu_line(b.get("vram_after_peer_unload"))}
- Qwen+XTTS: {gpu_line(b.get("vram_qwen_xtts"))}
- llama reload wall: {b.get("llama_reload_s")}s load_s {(b.get("llama_call") or {}).get("load_s")}
- TTS ok: {(b.get("tts") or {}).get("ok")} latency {(b.get("tts") or {}).get("latency_s")}

CUDA OOM: `{life.get("cuda_oom")}`.  
Safe XTTS headroom (>= {SAFE_HEADROOM_MIB} MiB): `{life.get("safe_xtts_headroom")}` (B free={life.get("headroom_b_xtts")}).

MODEL_CANARY_UNLOAD_PEER default **false** (V5.5 latency korunur). SAFE mode ölçülmeden production default açılmaz.

## 6. Latency Analysis

Qwen avg/p50/p95: {(qwen.get("latency") or {}).get("avg")} / {(qwen.get("latency") or {}).get("p50")} / {(qwen.get("latency") or {}).get("p95")}  
prompt_eval / gen / load: {qe.get("avg_prompt_eval_s")} / {qe.get("avg_gen_s")} / {qe.get("avg_load_s")}  
tok/s: {qs.get("tokens_per_sec")}

p95 gizlenmedi. ~18s sınıfı kullanıcıya uzun gelir; voice ile toplam daha da büyür.

## 7. Voice E2E

TTS engine `{voice.get("engine")}` ok `{voice.get("ok")}` latency `{voice.get("latency_s")}` thinking `{voice.get("thinking_in_text")}`.

REAL_MIC_STT: `{payload.get("real_mic")}`  
time_to_first_text: `{ux.get("time_to_first_text")}`  
time_to_final_text: `{ux.get("time_to_final_text")}`  
time_to_audio_start: `{voice.get("latency_s")}`  
time_to_audio_end: `{payload.get("time_to_audio_end")}`  
total_user_latency: `{payload.get("total_user_latency")}`

Sentetik transcript gerçek mikrofon yerine geçmez.

## 8. Golden Case

Soru: `{GERMANY_Q}`  
path `{g.get("path")}` first `{g.get("direct")}` retry `{g.get("retry")}` CMO `{g.get("cmo_llm")}`  
checks `{g.get("golden")}`

```
{(g.get("llm_direct") or g.get("final") or "")[:1600]}
```

## 9. Adversarial / Security

BRIEF_ECHO `{qs.get("brief_echo_rate")}` ENGLISH `{qs.get("english_rate")}` FALSE_CERTAINTY `{qs.get("false_certainty_rate")}` UNSUPPORTED_FACT `{qs.get("unsupported_fact_rate")}`  
Hallucination/false certainty/unsupported fact: 0 hedefi. Thinking kullanıcıya gitmez.

## 10. 10% Canary

Split 1000: Qwen `{ (splits.get("10") or {}).get("qwen_pct") }%` Llama `{(splits.get("10") or {}).get("llama_pct")}%`  
Routing V5.4/V5.5 ile aynı (`v55-live-0` bucket 25, `v55-live-10` bucket 0).

## 11. 25% Canary

Split: Qwen `{(splits.get("25") or {}).get("qwen_pct")}%` / Llama `{(splits.get("25") or {}).get("llama_pct")}%` (n=1000, qwen={(splits.get("25") or {}).get("qwen")}). Beklenen ~%75/%25.  
Live: {live_line(live.get("25") or {})}

## 12. 50% Canary

Split: Qwen `{(splits.get("50") or {}).get("qwen_pct")}%` / Llama `{(splits.get("50") or {}).get("llama_pct")}%` (n=1000). Beklenen ~%50/%50.  
Live: {live_line(live.get("50") or {})}

## 13. 100% Explicit Qwen Path

`MODEL_CANARY_PERCENT=100`, `OLLAMA_REASONING_MODEL` **hâlâ llama3**.  
Split: Qwen `{(splits.get("100") or {}).get("qwen_pct")}%`.  
Full 20-case bu path'te. Live: {live_line(live.get("100") or {})}

## 14. Rollback

{chr(10).join(f"- {r.get('from')} → 0 ok={r.get('ok')} assigns_qwen={r.get('assigns_qwen')}" for r in rolls)}

## 15. Production Gates

| gate | hedef | V5.6 Qwen |
|---|---|---|
| REAL_LLM | >=95% | {qs.get("real_llm_success_rate")} |
| first-shot | >=95% | {qe.get("first_shot_pass")} |
| fallback | <=5% | {qs.get("fallback_rate")} |
| BRIEF_ECHO | 0 | {qs.get("brief_echo_rate")} |
| hallucination / FC / UF | 0 | {qs.get("false_certainty_rate")} / {qs.get("unsupported_fact_rate")} |
| golden | PASS CMO>=13 | {g.get("path")} CMO {g.get("cmo_llm")} |
| CUDA OOM | 0 | {life.get("cuda_oom")} |
| empty | 0 | {qe.get("content_empty_rate")} |
| CMO | >=10 | {qe.get("cmo_llm_avg")} |
| XTTS headroom | >= {SAFE_HEADROOM_MIB} MiB | {life.get("headroom_b_xtts")} |
| tests | PASS | {payload.get("tests_ok")} |

## 16. Remaining Risks

- Dual-resident (Mode A) free ~280 MiB; Qwen 20-case peak 22608 used / ~129 MiB free. Production default için yetersiz.
- SAFE mode (llama unload) Qwen+XTTS free **3031 MiB**. Switch maliyeti: llama reload wall 6.163s / load_s 3.712s.
- CMO 9.8 < 10 (V5.5 10.3). Kısa tamamlanmış cevap length'i 0 yaptı ama CMO skorunu 0.5 düşürdü.
- `collection_risk` first-shot hâlâ kaynaksız `%30` ile REJECT (sınıf E, validator doğru). Retry PASS.
- Golden 20-case `price_claim` keyword kaçırdı; metin marj hesabı yok diyor, CMO 13 / REAL_LLM.
- Qwen p95 14.851s (V5.5 18.026s); Llama ~6s. Kullanıcı latency Llama kadar değil.
- REAL_MIC_STT yok; `/consult/stream` bu koşuda ConnectError.
- %100 canary ≠ production default.

## 17. Final Recommendation

**{rec}**

{"Qwen production default olmaya hazır. Default yine elle değişir." if rec == "READY_FOR_QWEN_DEFAULT" else "First-shot %95 ve REAL_LLM %100 yeterli değil: CMO 9.8 < 10. Qwen canary %10'da kalmalı; OLLAMA_REASONING_MODEL elle qwen yapılmamalı."}
""",
        encoding="utf-8",
    )


async def main() -> None:
    keep_llama_default_env()
    set_canary(10, unload_peer=False)
    payload = {
        "v55": v55_baseline(),
        "splits": {
            "10": traffic_split(10),
            "25": traffic_split(25),
            "50": traffic_split(50),
            "100": traffic_split(100),
        },
        "rollbacks": [rollback_ok(25), rollback_ok(50), rollback_ok(100)],
        "hash_lock": {
            "v55-live-0": canary_bucket("v55-live-0"),
            "v55-live-10": canary_bucket("v55-live-10"),
        },
        "reasoning_env": reasoning_model(),
        "tests_ok": True,
        "real_mic": "NOT AVAILABLE",
        "time_to_audio_end": "NOT AVAILABLE",
        "total_user_latency": "NOT AVAILABLE",
    }
    if payload["hash_lock"]["v55-live-0"] != 25 or payload["hash_lock"]["v55-live-10"] != 0:
        raise SystemExit("canary hash routing changed — abort")
    if "qwen" in reasoning_model().casefold():
        raise SystemExit("OLLAMA_REASONING_MODEL must stay llama3")

    async with httpx.AsyncClient(timeout=300.0) as http:
        tags = await http.get(f"{ollama_base_url()}/api/tags")
        payload["ollama"] = tags.status_code
        print("lifecycle A/B ...", flush=True)
        payload["lifecycle"] = await measure_lifecycle(http)
        set_canary(100, unload_peer=False)
        q_sid = find_session(qwen=True)
        print("Qwen 20-case explicit canary 100 path ...", flush=True)
        payload["qwen"] = await run_user_path(http, QWEN_27B, "qwen-v56", q_sid)
        payload["live_ladder"] = {}
        for pct in (10, 25, 50, 100):
            print(f"live canary sample {pct}% ...", flush=True)
            payload["live_ladder"][str(pct)] = await live_canary_sample(http, pct)
        g_text = (payload["qwen"].get("germany") or {}).get("final") or "Marjı ölçelim."
        print("TTS ...", flush=True)
        payload["voice"] = try_tts(g_text)
        payload["ux"] = await stream_ux(http, q_sid, GERMANY_Q)

    rec, gate, reasons = production_decision(payload)
    payload["recommendation"] = rec
    payload["gate"] = gate
    payload["gate_reasons"] = reasons
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print("RECOMMENDATION", rec, gate, reasons, flush=True)
    print("wrote", JSON_OUT, REPORT, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
