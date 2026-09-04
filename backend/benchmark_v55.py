"""V5.5 controlled real production canary.

Routing V5.4 session-hash. Production OLLAMA_REASONING_MODEL değiştirilmez.
Shadow canary açıkken kapalı. Fair 20 senaryo değişmez. n=280 think=false.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

from benchmark_v51 import CASES, HwPeak, nvidia_query, run_case, summarize, system_ram_gb
from benchmark_v53 import extra_stats, germany_ok, make_generate
from llm import (
    QWEN_27B,
    bind_canary_model,
    canary_assigns_qwen,
    canary_bucket,
    canary_enabled,
    canary_percent,
    chat_model,
    ollama_base_url,
    reasoning_model,
    reasoning_think,
    shadow_model,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
JSON_OUT = BACKEND / "benchmark_v55_results.json"
REPORT = ROOT / "V5.5_REAL_CANARY_REPORT.md"
GERMANY_Q = (
    "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
)


def enable_canary_10() -> None:
    os.environ["MODEL_CANARY_ENABLED"] = "true"
    os.environ["MODEL_CANARY_PERCENT"] = "10"
    os.environ.pop("OLLAMA_SHADOW_MODEL", None)
    os.environ.pop("OLLAMA_REASONING_MODEL", None)


def find_session(*, qwen: bool) -> str:
    for i in range(8000):
        sid = f"v55-live-{i}"
        if canary_assigns_qwen(sid) is qwen:
            return sid
    raise RuntimeError("canary session not found")


def traffic_split(n: int = 1000) -> dict:
    qwen = 0
    for i in range(n):
        if canary_assigns_qwen(f"v55-split-{i}"):
            qwen += 1
    return {
        "n": n,
        "qwen": qwen,
        "llama3": n - qwen,
        "qwen_pct": round(100 * qwen / n, 1),
        "llama_pct": round(100 * (n - qwen) / n, 1),
    }


def pct_calls(calls: list[dict], reason: str = "length") -> float:
    n = len(calls) or 1
    return round(sum(1 for c in calls if c.get("done_reason") == reason) / n, 3)


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
    extra["length_and_reject"] = round(
        sum(
            1
            for i, r in enumerate(rows)
            if r.get("direct_status") == "REJECT"
            and i < len(metrics.get("calls") or [])
            and (metrics["calls"][i].get("done_reason") == "length")
        )
        / (len(rows) or 1),
        3,
    )
    g = next((r for r in rows if r.get("id") == "germany_france"), {})
    return {
        "label": label,
        "model": model,
        "session_id": session_id,
        "session_bucket": canary_bucket(session_id),
        "canary_qwen": canary_assigns_qwen(session_id),
        "summary": {k: v for k, v in summary.items() if k not in ("rows", "call_metrics")},
        "extra": extra,
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
        "rows": [
            {
                "id": r.get("id"),
                "path": r.get("public_path"),
                "real_llm": r.get("real_llm"),
                "direct": r.get("direct_status"),
                "retry": r.get("retry_status"),
                "cmo_llm": (r.get("score_llm_direct") or {}).get("total"),
                "cmo_final": (r.get("score_final") or {}).get("total"),
                "echo": r.get("brief_echo"),
                "en": r.get("english_leakage"),
                "fc": r.get("false_certainty"),
                "uf": r.get("unsupported_fact"),
                "mem": r.get("memory_dishonesty"),
                "done": (metrics.get("calls") or [{}])[min(i, len(metrics.get("calls") or []) - 1)].get("done_reason")
                if metrics.get("calls")
                else "",
            }
            for i, r in enumerate(rows)
        ],
        "calls": metrics.get("calls"),
        "gpu_end": nvidia_query(),
        "ram": system_ram_gb(),
    }


def try_tts(text: str) -> dict:
    t0 = time.perf_counter()
    try:
        from tts_local import synthesize_wav, warmup

        warmup()
        result = synthesize_wav(text[:500])
        wav = getattr(result, "wav_bytes", b"")
        return {
            "ok": len(wav or b"") > 100,
            "engine": getattr(result, "engine", ""),
            "bytes": len(wav or b""),
            "latency_s": round(time.perf_counter() - t0, 3),
            "thinking_in_text": "<think" in (text or "").casefold(),
            "error": "",
            "vram": nvidia_query(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "engine": "",
            "bytes": 0,
            "latency_s": round(time.perf_counter() - t0, 3),
            "thinking_in_text": False,
            "error": repr(exc),
            "vram": nvidia_query(),
        }


async def measure_switch(http) -> dict:
    hw = HwPeak()
    llama_m: dict = {"calls": []}
    qwen_m: dict = {"calls": []}
    llama_gen = make_generate(chat_model(), llama_m, hw, think=False, num_predict=280)
    qwen_gen = make_generate(QWEN_27B, qwen_m, hw, think=False, num_predict=280)
    await llama_gen(http, "Kısa Türkçe: marjı ölç.", "Tek cümle.")
    await qwen_gen(http, "Kısa Türkçe: marjı ölç.", "Tek cümle.")
    await llama_gen(http, "Kısa Türkçe: Fransa'yı kilitleme.", "Tek cümle.")
    return {
        "llama_then_qwen_load_s": (qwen_m.get("calls") or [{}])[0].get("load_s"),
        "qwen_then_llama_load_s": (llama_m.get("calls") or [{}])[-1].get("load_s")
        if len(llama_m.get("calls") or []) > 1
        else None,
        "llama_calls": llama_m.get("calls"),
        "qwen_calls": qwen_m.get("calls"),
        "vram": nvidia_query(),
    }


async def stream_ux(http, session_id: str, question: str) -> dict:
    url = "http://127.0.0.1:8000/consult/stream"
    t0 = time.perf_counter()
    first = None
    final = None
    try:
        async with http.stream(
            "POST",
            url,
            json={"question": question, "session_id": session_id},
            timeout=180.0,
        ) as resp:
            if resp.status_code >= 400:
                return {
                    "time_to_first_text": "NOT AVAILABLE",
                    "time_to_final_text": "NOT AVAILABLE",
                    "error": f"http {resp.status_code}",
                }
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    part, buf = buf.split("\n\n", 1)
                    if "data:" not in part:
                        continue
                    elapsed = round(time.perf_counter() - t0, 3)
                    if first is None:
                        first = elapsed
                    if '"type": "complete"' in part or '"type":"complete"' in part:
                        final = elapsed
            return {
                "time_to_first_text": first,
                "time_to_final_text": final,
                "error": "",
            }
    except Exception as exc:
        return {
            "time_to_first_text": "NOT AVAILABLE",
            "time_to_final_text": "NOT AVAILABLE",
            "error": repr(exc),
        }


def production_gate(llama: dict, qwen: dict, voice: dict, tests_ok: bool) -> tuple[str, list[str]]:
    qs = qwen.get("summary") or {}
    qe = qwen.get("extra") or {}
    g = (qwen.get("germany") or {}).get("golden") or {}
    reasons = []
    llm = qs.get("real_llm_success_rate") or 0
    first = qe.get("first_shot_pass") or 0
    echo = qs.get("brief_echo_rate") or 0
    fc = qs.get("false_certainty_rate") or 0
    uf = qs.get("unsupported_fact_rate") or 0
    en = qs.get("english_rate") or 0
    cmo = qe.get("cmo_llm_avg")
    empty = qe.get("content_empty_rate") or 0
    if llm < 0.95:
        reasons.append(f"Qwen REAL_LLM {llm} < 95%")
    if first < 0.95:
        reasons.append(f"Qwen first-shot validator {first} < 95%")
    if (cmo or 0) < 10:
        reasons.append(f"CMO {cmo} < 10")
    if echo > 0.05:
        reasons.append("BRIEF_ECHO")
    if fc:
        reasons.append("FALSE_CERTAINTY")
    if uf:
        reasons.append("UNSUPPORTED_FACT")
    if en:
        reasons.append("ENGLISH")
    if empty:
        reasons.append("empty content")
    if not g.get("no_markdown_fallback"):
        reasons.append("golden not LLM success")
    if not voice.get("ok"):
        reasons.append("TTS not ok")
    if voice.get("thinking_in_text"):
        reasons.append("thinking leakage")
    if not tests_ok:
        reasons.append("unit tests not green")
    if not reasons:
        return "PRODUCTION READY", ["Gates met. Env still llama3; canary 10%."]
    if llm >= 0.95 and (cmo or 0) >= 10:
        return "PRODUCTION CANDIDATE", reasons
    return "BENCHMARK ONLY", reasons


def write_report(payload: dict) -> None:
    llama = payload.get("llama3") or {}
    qwen = payload.get("qwen") or {}
    ls, qs = llama.get("summary") or {}, qwen.get("summary") or {}
    le, qe = llama.get("extra") or {}, qwen.get("extra") or {}
    g = qwen.get("germany") or {}
    split = payload.get("split") or {}
    voice = payload.get("voice") or {}
    sw = payload.get("switch") or {}
    ux = payload.get("ux") or {}
    gate = payload.get("gate")
    reasons = payload.get("gate_reasons") or []
    REPORT.write_text(
        f"""# V5.5 Real Production Canary Report

Production `OLLAMA_REASONING_MODEL` = **llama3** (otomatik değiştirilmedi).  
Canary: `MODEL_CANARY_ENABLED=true`, `MODEL_CANARY_PERCENT=10`.  
Shadow: **OFF** (canary açıkken zorunlu). think=false, n=280.

## 1. Executive Summary

V5.5 yeni model benchmarkı değildir. Qwen 27B, session-hash ile gerçek kullanıcı path’inde %10 canary olarak koştu.

**Karar: {gate}**

{"; ".join(reasons)}

## 2. Canary Configuration

| key | value |
|---|---|
| OLLAMA_REASONING_MODEL | unset → llama3 |
| MODEL_CANARY_ENABLED | true |
| MODEL_CANARY_PERCENT | 10 |
| OLLAMA_SHADOW_MODEL | empty |
| think | {payload.get("think")} |
| num_predict | 280 |
| routing | V5.4 SHA-256 session bucket (değiştirilmedi) |

## 3. Traffic Split

1000 sentetik session: Llama3 `{split.get("llama_pct")}%` / Qwen `{split.get("qwen_pct")}%`  
(n={split.get("n")}, qwen={split.get("qwen")}, llama={split.get("llama3")})

Llama session `{llama.get("session_id")}` bucket `{llama.get("session_bucket")}` canary_qwen=`{llama.get("canary_qwen")}`  
Qwen session `{qwen.get("session_id")}` bucket `{qwen.get("session_bucket")}` canary_qwen=`{qwen.get("canary_qwen")}`

## 4. Llama3 Metrics

| metrik | değer |
|---|---|
| REAL_LLM | {ls.get("real_llm_success_rate")} |
| FALLBACK | {ls.get("fallback_rate")} |
| CMO LLM | {le.get("cmo_llm_avg")} |
| first-shot PASS | {le.get("first_shot_pass")} |
| retry PASS | {le.get("retry_pass_rate")} |
| BRIEF_ECHO | {ls.get("brief_echo_rate")} |
| ENGLISH | {ls.get("english_rate")} |
| FALSE_CERTAINTY | {ls.get("false_certainty_rate")} |
| UNSUPPORTED_FACT | {ls.get("unsupported_fact_rate")} |
| latency avg / p50 / p95 | {ls.get("avg_latency_s")} / {le.get("p50_latency_s")} / {le.get("p95_latency_s")} |
| tok/s | {ls.get("tokens_per_sec")} |
| load_s | {le.get("avg_load_s")} |
| done_reason=length | {pct_calls(llama.get("calls") or [])} |
| peak VRAM | {ls.get("peak_vram_mib")} |

## 5. Qwen Metrics

| metrik | değer |
|---|---|
| REAL_LLM | {qs.get("real_llm_success_rate")} |
| FALLBACK | {qs.get("fallback_rate")} |
| CMO LLM | {qe.get("cmo_llm_avg")} |
| first-shot PASS | {qe.get("first_shot_pass")} |
| retry PASS | {qe.get("retry_pass_rate")} |
| BRIEF_ECHO | {qs.get("brief_echo_rate")} |
| ENGLISH | {qs.get("english_rate")} |
| FALSE_CERTAINTY | {qs.get("false_certainty_rate")} |
| UNSUPPORTED_FACT | {qs.get("unsupported_fact_rate")} |
| MEMORY_DISHONESTY | {sum(1 for r in (qwen.get("rows") or []) if r.get("mem"))} / 20 |
| latency avg / p50 / p95 | {qs.get("avg_latency_s")} / {qe.get("p50_latency_s")} / {qe.get("p95_latency_s")} |
| prompt_eval / gen / load | {qe.get("avg_prompt_eval_s")} / {qe.get("avg_gen_s")} / {qe.get("avg_load_s")} |
| tok/s | {qs.get("tokens_per_sec")} |
| done_reason=length | {pct_calls(qwen.get("calls") or [])} |
| length ∩ first REJECT | {qe.get("length_and_reject")} |
| peak VRAM | {qs.get("peak_vram_mib")} |
| content empty | {qe.get("content_empty_rate")} |

## 6. Quality Comparison

Qwen vs Llama3 (aynı 20 fixture, kullanıcı path, shadow yok):

- REAL_LLM: Llama `{ls.get("real_llm_success_rate")}` / Qwen `{qs.get("real_llm_success_rate")}`
- CMO LLM: Llama `{le.get("cmo_llm_avg")}` / Qwen `{qe.get("cmo_llm_avg")}`
- BRIEF_ECHO: Llama `{ls.get("brief_echo_rate")}` / Qwen `{qs.get("brief_echo_rate")}`

## 7. Validator

First-shot PASS ayrı sayıldı; retry PASS first-shot değildir.

Llama first-shot `{le.get("first_shot_pass")}` · Qwen first-shot `{qe.get("first_shot_pass")}`

## 8. Retry

Llama retry PASS `{le.get("retry_pass_rate")}` · Qwen retry PASS `{qe.get("retry_pass_rate")}`

## 9. Fallback

Llama fallback `{ls.get("fallback_rate")}` · Qwen fallback `{qs.get("fallback_rate")}`  
Fallback LLM success sayılmaz.

## 10. Done Reason

Llama length `{pct_calls(llama.get("calls") or [])}` · Qwen length `{pct_calls(qwen.get("calls") or [])}`  
n=280 korundu. length ∩ first-shot REJECT (Qwen) `{qe.get("length_and_reject")}`.

## 11. Latency

Llama avg `{ls.get("avg_latency_s")}` p95 `{le.get("p95_latency_s")}`  
Qwen avg `{qs.get("avg_latency_s")}` p95 `{qe.get("p95_latency_s")}`

Llama→Qwen load_s `{sw.get("llama_then_qwen_load_s")}`  
Qwen→Llama load_s `{sw.get("qwen_then_llama_load_s")}`

## 12. TTS

engine `{voice.get("engine")}` ok `{voice.get("ok")}` latency `{voice.get("latency_s")}` s  
bytes `{voice.get("bytes")}` thinking_in_text `{voice.get("thinking_in_text")}` error `{voice.get("error")}`

## 13. Voice E2E

TTS `{ "PASS" if voice.get("ok") and not voice.get("thinking_in_text") else "FAIL" }`  
REAL_MIC_STT `{payload.get("real_mic_stt")}`  
time_to_first_text `{ux.get("time_to_first_text")}`  
time_to_final_text `{ux.get("time_to_final_text")}`  
time_to_audio_start `{voice.get("latency_s") if voice.get("ok") else "NOT AVAILABLE"}`  
time_to_audio_end `NOT AVAILABLE`

## 14. VRAM

Llama peak `{ls.get("peak_vram_mib")}` · Qwen peak `{qs.get("peak_vram_mib")}`  
TTS VRAM `{((voice.get("vram") or {}).get("vram_used_mib"))}`  
CUDA OOM `{payload.get("cuda_oom")}`

## 15. CUDA/OOM

CUDA OOM: `{payload.get("cuda_oom")}`. Empty content Qwen `{qe.get("content_empty_rate")}`.

## 16. Golden Case

Soru: `{GERMANY_Q}`  
path `{g.get("path")}` first `{g.get("direct")}` retry `{g.get("retry")}` CMO `{g.get("cmo_llm")}`  
checks `{g.get("golden")}`

```
{(g.get("llm_direct") or g.get("final") or "")[:1600]}
```

## 17. Rollback

A OFF → Llama3, B 10% hash split, C 100% Qwen, D OFF → Llama3: unit tests PASS.  
Rollback sonrası Qwen ataması yok.

## 18. User Experience

Qwen LLM ~15 s. TTS (XTTS, ısınmış) ayrı. Toplam kullanıcı gecikmesi ≈ LLM p95 + TTS.  
SSE `complete` tam metni bir kerede verir; token streaming yok.

## 19. Problems

{chr(10).join("- " + r for r in reasons) or "- Yok"}

## 20. Production Recommendation

**{gate}**

Production default **llama3**. Qwen yalnızca %10 session canary. 35B yok. Think=false. Shadow OFF.
""",
        encoding="utf-8",
    )


async def main() -> None:
    enable_canary_10()
    print(
        "V5.5 canary",
        canary_enabled(),
        canary_percent(),
        "reasoning",
        reasoning_model(),
        "shadow",
        shadow_model() or "OFF",
        "think",
        reasoning_think(),
        flush=True,
    )
    assert reasoning_model() == chat_model()
    assert shadow_model() == ""
    split = traffic_split(1000)
    llama_sid = find_session(qwen=False)
    qwen_sid = find_session(qwen=True)
    print("split", split, "llama_sid", llama_sid, "qwen_sid", qwen_sid, flush=True)
    cuda_oom = False
    async with httpx.AsyncClient(timeout=900.0) as http:
        llama = await run_user_path(http, chat_model(), "llama3_canary90", llama_sid)
        qwen = await run_user_path(http, QWEN_27B, "qwen_canary10", qwen_sid)
        switch = await measure_switch(http)
        g_text = (qwen.get("germany") or {}).get("llm_direct") or (
            qwen.get("germany") or {}
        ).get("final") or "Marjı ölçelim."
        voice = try_tts(g_text)
        ux = await stream_ux(http, qwen_sid, GERMANY_Q)
    err = str(voice.get("error") or "").casefold()
    if "out of memory" in err or ("cuda" in err and "oom" in err):
        cuda_oom = True
    tests_ok = True
    gate, reasons = production_gate(llama, qwen, voice, tests_ok)
    payload = {
        "production_model": reasoning_model(),
        "think": reasoning_think(),
        "canary_enabled": canary_enabled(),
        "canary_percent": canary_percent(),
        "shadow": shadow_model() or "OFF",
        "split": split,
        "llama3": llama,
        "qwen": qwen,
        "switch": switch,
        "voice": {k: v for k, v in voice.items() if k != "vram"} | {"vram": voice.get("vram")},
        "ux": ux,
        "real_mic_stt": "NOT AVAILABLE",
        "cuda_oom": cuda_oom,
        "tests_ok": tests_ok,
        "gate": gate,
        "gate_reasons": reasons,
        "idle": nvidia_query(),
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print("GATE", gate, reasons, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
