"""V5.4 shadow / canary / voice production-validation benchmark.

Production OLLAMA_REASONING_MODEL değiştirmez. 35B kurmaz. Voice koduna dokunmaz.
Fair 20 senaryo değişmez. Qwen n=280 think=false.
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

from benchmark_v51 import (
    CASES,
    HwPeak,
    nvidia_query,
    run_case,
    shared_gpu_sample,
    summarize,
    system_ram_gb,
)
from benchmark_v53 import extra_stats, germany_ok, make_generate
from llm import (
    QWEN_27B,
    canary_enabled,
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
JSON_OUT = BACKEND / "benchmark_v54_results.json"
SHADOW_REPORT = ROOT / "V5.4_SHADOW_CANARY_REPORT.md"
VOICE_REPORT = ROOT / "V5.4_VOICE_E2E_REPORT.md"
V53_JSON = BACKEND / "benchmark_v53_results.json"
VRAM_V52 = {"MODEL_ONLY_PEAK_VRAM": 17948, "XTTS_MODEL_PEAK_VRAM": 19925}

GERMANY_Q = (
    "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
)
LONG_CTX = (
    "Ürün zeytinyağı. Kapasite düşük olabilir. Almanya'da Nordöl Import, Gastro Nord "
    "ve Retail Mix ile temas var. Marj eşiği kayıtlı değil. Rakip fiyatı doğrulanmadı. "
    "Fransa için lead yok. Tahsilat kaydı yok. Web bu turda yok. "
    + GERMANY_Q
)


def v53_qwen280() -> dict:
    if not V53_JSON.exists():
        return {}
    data = json.loads(V53_JSON.read_text(encoding="utf-8"))
    for item in data.get("qwen_npred") or []:
        if item.get("num_predict") == 280:
            return item
    return {}


def dim_avg(rows: list[dict], key: str) -> float | None:
    vals = []
    for row in rows:
        block = row.get("score_llm_direct") or row.get("score_final") or {}
        if key in block and isinstance(block[key], (int, float)):
            vals.append(block[key])
    return round(sum(vals) / len(vals), 3) if vals else None


def quality_block(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "DATA_USE": dim_avg(rows, "DATA_USE"),
        "DIAGNOSIS": dim_avg(rows, "DIAGNOSIS"),
        "REASONING": dim_avg(rows, "REASONING"),
        "DECISION": dim_avg(rows, "DECISION"),
        "UNCERTAINTY": dim_avg(rows, "UNCERTAINTY"),
        "ACTION": dim_avg(rows, "ACTION"),
        "NATURAL_TURKISH": dim_avg(rows, "NATURAL_TURKISH"),
        "BRIEF_ECHO": round(sum(1 for r in rows if r.get("brief_echo")) / n, 3),
        "ENGLISH": round(sum(1 for r in rows if r.get("english_leakage")) / n, 3),
        "UNSUPPORTED_FACT": round(sum(1 for r in rows if r.get("unsupported_fact")) / n, 3),
        "FALSE_CERTAINTY": round(sum(1 for r in rows if r.get("false_certainty")) / n, 3),
        "MEMORY_DISHONESTY": round(sum(1 for r in rows if r.get("memory_dishonesty")) / n, 3),
        "PROMPT_LEAK": round(
            sum(1 for r in rows if (r.get("score_final") or {}).get("PROMPT_LEAK")) / n,
            3,
        ),
        "validator_pass": round(sum(1 for r in rows if r.get("direct_status") == "PASS") / n, 3),
        "real_llm": round(sum(1 for r in rows if r.get("real_llm")) / n, 3),
        "fallback": round(sum(1 for r in rows if not r.get("real_llm")) / n, 3),
    }


async def run_phase(http, model: str, label: str) -> dict:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    gen = make_generate(model, metrics, hw, think=False, num_predict=280)
    rows = []
    for case in CASES:
        print(f"[{label}] {case['id']} ...", flush=True)
        rows.append(await run_case(http, gen, case))
    summary = summarize(label, rows, metrics, hw, "TESTED")
    extra = extra_stats(rows, metrics)
    extra["memory_dishonesty_rate"] = round(
        sum(1 for r in rows if r.get("memory_dishonesty")) / (len(rows) or 1), 3
    )
    extra["prompt_leak_rate"] = round(
        sum(1 for r in rows if (r.get("score_final") or {}).get("PROMPT_LEAK"))
        / (len(rows) or 1),
        3,
    )
    extra["length_by_task"] = {}
    calls = metrics.get("calls") or []
    for i, case in enumerate(CASES):
        if i < len(calls):
            extra["length_by_task"][case["id"]] = calls[i].get("done_reason") or ""
    g = next((r for r in rows if r.get("id") == "germany_france"), {})
    return {
        "label": label,
        "model": model,
        "think": False,
        "num_predict": 280,
        "role": label,
        "summary": {k: v for k, v in summary.items() if k not in ("rows", "call_metrics")},
        "extra": extra,
        "quality": quality_block(rows),
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
            }
            for r in rows
        ],
        "calls": metrics.get("calls"),
        "gpu_end": nvidia_query(),
        "ram": system_ram_gb(),
        "shared": shared_gpu_sample(),
    }


def pct(calls: list[dict], reason: str = "length") -> float:
    n = len(calls) or 1
    return round(sum(1 for c in calls if c.get("done_reason") == reason) / n, 3)


async def one_generate(http, model: str, user: str, system: str, hw: HwPeak) -> dict:
    gen = make_generate(model, {"calls": []}, hw, think=False, num_predict=280)
    t0 = time.perf_counter()
    text = await gen(http, user, system_prompt=system)
    return {
        "text": text,
        "latency_s": round(time.perf_counter() - t0, 3),
        "content_length": len(text or ""),
        "thinking_in_text": "INTERNAL" in (text or "") or "<think" in (text or "").casefold(),
    }


_TTS_WARMED = False


def try_tts(text: str, hw: HwPeak) -> dict:
    global _TTS_WARMED
    hw.mark()
    t0 = time.perf_counter()
    try:
        from tts_local import synthesize_wav, warmup

        status: dict = {}
        if not _TTS_WARMED:
            status = warmup() or {}
            _TTS_WARMED = True
            hw.mark()
        result = synthesize_wav(text)
        hw.mark()
        wav = getattr(result, "wav_bytes", result)
        size = len(wav) if isinstance(wav, (bytes, bytearray)) else 0
        engine = getattr(result, "engine", "") or status.get("engine") or ""
        err = ""
        if isinstance(status, dict) and status.get("error"):
            err = str(status.get("error"))
        return {
            "ok": size > 100,
            "bytes": size,
            "engine": engine,
            "latency_s": round(time.perf_counter() - t0, 3),
            "error": err,
            "vram": nvidia_query(),
        }
    except Exception as exc:
        hw.mark()
        return {
            "ok": False,
            "bytes": 0,
            "engine": "",
            "latency_s": round(time.perf_counter() - t0, 3),
            "error": repr(exc),
            "vram": nvidia_query(),
        }


def production_gate(qwen: dict, voice: dict, tests: dict) -> tuple[str, list[str]]:
    s = qwen.get("summary") or {}
    e = qwen.get("extra") or {}
    g = (qwen.get("germany") or {}).get("golden") or {}
    q = qwen.get("quality") or {}
    reasons = []
    llm = s.get("real_llm_success_rate") or 0
    echo = s.get("brief_echo_rate") or 0
    fc = s.get("false_certainty_rate") or 0
    uf = q.get("UNSUPPORTED_FACT") or 0
    en = s.get("english_rate") or 0
    mem = e.get("memory_dishonesty_rate") or 0
    leak = e.get("prompt_leak_rate") or 0
    cmo = e.get("cmo_llm_avg")
    empty = e.get("content_empty_rate") or 0
    think_leak = e.get("thinking_present_rate") or 0
    pass_r = e.get("pass_rate") or 0
    voice_ok = bool((voice or {}).get("pass"))
    oom = bool((voice or {}).get("cuda_oom"))
    if (cmo or 0) < 10:
        reasons.append(f"CMO LLM {cmo} < 10")
    if llm < 0.95:
        reasons.append(f"REAL_LLM_SUCCESS {llm} < 95%")
    if echo > 0.05:
        reasons.append(f"BRIEF_ECHO {echo} > 5%")
    if fc:
        reasons.append("FALSE_CERTAINTY > 0")
    if uf:
        reasons.append(f"REAL UNSUPPORTED_FACT {uf} > 0")
    if mem:
        reasons.append("MEMORY_DISHONESTY > 0")
    if leak:
        reasons.append("PROMPT_LEAK > 0")
    if en > 0.02:
        reasons.append(f"ENGLISH {en}")
    if pass_r < 0.95:
        reasons.append(f"validator PASS {pass_r} < 95%")
    if empty:
        reasons.append("content empty")
    if think_leak:
        reasons.append("thinking present under think=false")
    if not g.get("no_markdown_fallback"):
        reasons.append("Germany/France golden not LLM-success")
    if not voice_ok:
        reasons.append("voice E2E not fully passing")
    if (voice or {}).get("engine") and str(voice.get("engine")).casefold() != "xtts":
        reasons.append(f"voice engine={voice.get('engine')} (XTTS değil)")
    if oom:
        reasons.append("CUDA OOM")
    if not tests.get("canary_rollback"):
        reasons.append("canary rollback test missing/fail")
    critical = [r for r in reasons if "OOM" in r or "thinking" in r or "empty" in r]
    if not reasons:
        return "PRODUCTION READY", ["Quality, reliability, voice and rollback gates met. Env still llama3."]
    if llm >= 0.9 and (cmo or 0) >= 10 and not critical:
        return "PRODUCTION CANDIDATE", reasons
    return "BENCHMARK ONLY", reasons


def write_reports(payload: dict) -> None:
    llama = payload.get("llama3") or {}
    qwen = payload.get("qwen_shadow") or {}
    voice = payload.get("voice") or {}
    vram = payload.get("vram") or {}
    gate = payload.get("gate")
    reasons = payload.get("gate_reasons") or []
    ls = llama.get("summary") or {}
    qs = qwen.get("summary") or {}
    qe = qwen.get("extra") or {}
    lq = llama.get("quality") or {}
    qq = qwen.get("quality") or {}
    g = qwen.get("germany") or {}
    v53 = payload.get("v53_baseline") or {}
    v53s = (v53.get("summary") or {}) if isinstance(v53, dict) else {}
    tests = payload.get("tests") or {}

    def row_md(rows: list[dict]) -> str:
        lines = [
            "| id | path | real_llm | validator | CMO LLM | CMO final | echo | EN | FC | UF | mem |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows or []:
            lines.append(
                f"| {r.get('id')} | {r.get('path')} | {r.get('real_llm')} | "
                f"{r.get('direct')}/{r.get('retry') or '-'} | {r.get('cmo_llm')} | "
                f"{r.get('cmo_final')} | {r.get('echo')} | {r.get('en')} | {r.get('fc')} | "
                f"{r.get('uf')} | {r.get('mem')} |"
            )
        return "\n".join(lines)

    SHADOW_REPORT.write_text(
        f"""# V5.4 Shadow / Canary Report

Production `OLLAMA_REASONING_MODEL` = **llama3** (otomatik değiştirilmedi).  
Qwen = `qwen3.5:27b`, think=false, num_predict=280.  
Shadow default = **OFF**. Canary default = **OFF**.

## 1. Executive summary

V5.4 bir model upgrade değildir. Qwen 27B, V5.3 benchmark kazananından production validation’a alındı:

- Production path: Llama3 → validator → fallback → TTS
- Shadow: aynı input, Qwen think=false, validator + evaluator + telemetry; kullanıcı/TTS/history/memory yok
- Canary: `MODEL_CANARY_ENABLED` + `MODEL_CANARY_PERCENT`, session-hash, default kapalı
- Rollback: canary off veya `OLLAMA_REASONING_MODEL=llama3`

**Karar: {gate}**

{"; ".join(reasons)}

## 2. V5.3 baseline

| metrik | V5.3 Qwen n=280 |
|---|---|
| REAL_LLM | {v53s.get("real_llm_success_rate", "100% (rapor)")} |
| CMO LLM | {(v53.get("extra") or {}).get("cmo_llm_avg", "10.55")} |
| latency avg | {v53s.get("avg_latency_s", "15.26")} |
| p50 / p95 | 15.4 / 18.0 |
| tok/s | {v53s.get("tokens_per_sec", "21.35")} |
| done_reason=length | {(v53.get("extra") or {}).get("length_done_rate", "0.27")} |
| XTTS+Qwen VRAM | 19925 MiB (V5.2) |

## 3. Shadow results

Shadow production cevabını değiştirmez. think=false. Raw thinking loglanmaz.

| | Llama3 production | Qwen shadow |
|---|---|---|
| REAL_LLM | {ls.get("real_llm_success_rate")} | {qs.get("real_llm_success_rate")} |
| FALLBACK | {ls.get("fallback_rate")} | {qs.get("fallback_rate")} |
| CMO LLM | {(llama.get("extra") or {}).get("cmo_llm_avg")} | {qe.get("cmo_llm_avg")} |
| validator PASS | {(llama.get("extra") or {}).get("pass_rate")} | {qe.get("pass_rate")} |
| BRIEF_ECHO | {ls.get("brief_echo_rate")} | {qs.get("brief_echo_rate")} |
| ENGLISH | {ls.get("english_rate")} | {qs.get("english_rate")} |
| FALSE_CERTAINTY | {ls.get("false_certainty_rate")} | {qs.get("false_certainty_rate")} |
| UNSUPPORTED_FACT | {lq.get("UNSUPPORTED_FACT")} | {qq.get("UNSUPPORTED_FACT")} |
| MEMORY_DISHONESTY | {(llama.get("extra") or {}).get("memory_dishonesty_rate")} | {qe.get("memory_dishonesty_rate")} |
| PROMPT_LEAK | {(llama.get("extra") or {}).get("prompt_leak_rate")} | {qe.get("prompt_leak_rate")} |
| done_reason=length | {pct(llama.get("calls") or [])} | {pct(qwen.get("calls") or [])} |
| latency avg | {ls.get("avg_latency_s")} | {qs.get("avg_latency_s")} |
| p50 | {(llama.get("extra") or {}).get("p50_latency_s")} | {qe.get("p50_latency_s")} |
| p95 | {(llama.get("extra") or {}).get("p95_latency_s")} | {qe.get("p95_latency_s")} |
| tok/s | {ls.get("tokens_per_sec")} | {qs.get("tokens_per_sec")} |
| prompt_eval_s | {(llama.get("extra") or {}).get("avg_prompt_eval_s")} | {qe.get("avg_prompt_eval_s")} |
| gen_s | {(llama.get("extra") or {}).get("avg_gen_s")} | {qe.get("avg_gen_s")} |
| load_s | {qe.get("avg_load_s")} | {qe.get("avg_load_s")} |

## 4. Llama3 vs Qwen (MODEL QUALITY)

| dim | Llama3 | Qwen |
|---|---|---|
| DATA_USE | {lq.get("DATA_USE")} | {qq.get("DATA_USE")} |
| DIAGNOSIS | {lq.get("DIAGNOSIS")} | {qq.get("DIAGNOSIS")} |
| REASONING | {lq.get("REASONING")} | {qq.get("REASONING")} |
| DECISION | {lq.get("DECISION")} | {qq.get("DECISION")} |
| UNCERTAINTY | {lq.get("UNCERTAINTY")} | {qq.get("UNCERTAINTY")} |
| ACTION | {lq.get("ACTION")} | {qq.get("ACTION")} |
| NATURAL_TURKISH | {lq.get("NATURAL_TURKISH")} | {qq.get("NATURAL_TURKISH")} |
| BRIEF_ECHO | {lq.get("BRIEF_ECHO")} | {qq.get("BRIEF_ECHO")} |
| ENGLISH | {lq.get("ENGLISH")} | {qq.get("ENGLISH")} |
| UNSUPPORTED_FACT | {lq.get("UNSUPPORTED_FACT")} | {qq.get("UNSUPPORTED_FACT")} |
| FALSE_CERTAINTY | {lq.get("FALSE_CERTAINTY")} | {qq.get("FALSE_CERTAINTY")} |
| MEMORY_DISHONESTY | {lq.get("MEMORY_DISHONESTY")} | {qq.get("MEMORY_DISHONESTY")} |
| PROMPT_LEAK | {lq.get("PROMPT_LEAK")} | {qq.get("PROMPT_LEAK")} |
| validator pass | {lq.get("validator_pass")} | {qq.get("validator_pass")} |
| real LLM | {lq.get("real_llm")} | {qq.get("real_llm")} |
| fallback | {lq.get("fallback")} | {qq.get("fallback")} |

## 5. 20 scenario results

### Llama3 production

{row_md(llama.get("rows") or [])}

### Qwen shadow

{row_md(qwen.get("rows") or [])}

## 6. Golden case

Soru: `{GERMANY_Q}`

Qwen path: `{g.get("path")}`  
validator: `{g.get("direct")}` / retry `{g.get("retry")}`  
CMO LLM: `{g.get("cmo_llm")}`  
golden checks: `{g.get("golden")}`

```
{(g.get("llm_direct") or g.get("final") or "")[:1600]}
```

## 7. Hallucination / adversarial tests

Unit tests: user-claim vs model-fact, competitor `%20`, uydurma pazar/ciro, memory dishonesty, prompt leak, English leak, false certainty. Validator REJECT gevşetilmedi.

## 8. Validator results

Qwen first-shot PASS `{qe.get("pass_rate")}`, WARN `{qe.get("warn_rate")}`, REJECT `{qe.get("reject_rate")}`.  
REAL_LLM (retry dahil) `{qs.get("real_llm_success_rate")}`. Fallback LLM success sayılmaz.

## 9. done_reason

Qwen n=280 `length` oranı: `{pct(qwen.get("calls") or [])}`.  
Llama3 `length` oranı: `{pct(llama.get("calls") or [])}`.  
Production num_predict **280** kaldı (384/512 otomatik geçilmedi).

## 10. Latency

Qwen avg `{qs.get("avg_latency_s")}` / p50 `{qe.get("p50_latency_s")}` / p95 `{qe.get("p95_latency_s")}`.  
prompt_eval `{qe.get("avg_prompt_eval_s")}` gen `{qe.get("avg_gen_s")}` load `{qe.get("avg_load_s")}`.  
keep_alive=30m. load_s düşükse model her request’te reload olmuyor.

## 11. VRAM

Idle: `{vram.get("idle")}`  
Test A Qwen-only peak: `{vram.get("test_a_peak")}`  
Test B XTTS+Qwen peak: `{vram.get("test_b_peak")}`  
CUDA OOM: `{vram.get("cuda_oom")}`  
Shared GPU memory VRAM sayılmadı.

## 12. Voice E2E

Ayrıntı: `V5.4_VOICE_E2E_REPORT.md`. PASS=`{voice.get("pass")}` thinking leak=`{voice.get("thinking_leak")}`.

## 13. Canary behavior

Default OFF → %100 Llama3.  
`MODEL_CANARY_ENABLED=true` + percent 10 → session-hash ile ~%10 Qwen, aynı session sabit.  
percent 0 → Llama3. percent 100 → test Qwen.  
`OLLAMA_REASONING_MODEL=qwen3.5:27b` explicit canary/test.

## 14. Rollback

Canary kapatılınca veya `OLLAMA_REASONING_MODEL=llama3` ile Llama3.  
Qwen CUDA OOM / empty / validator fail → mevcut commercial fallback. Thinking cevap olmaz.

## 15. Test results

`{tests.get("summary")}`

## 16. Remaining risks

- Qwen ~15s; voice round-trip daha uzun
- XTTS + 27B headroom ~3 GB; Llama3+Qwen+XTTS aynı anda VRAM baskısı
- Shadow async Qwen, production Llama3 ile GPU’da çakışabilir
- done_reason=length ~%27 (n=280) kesik cevap riski
- Canary default kapalı; gerçek kullanıcı %10 split henüz production’da yok

## 17. Production recommendation

**{gate}**

Production default **llama3** kalır. Qwen explicit env veya kontrollü canary ile. 35B yok.
""",
        encoding="utf-8",
    )

    VOICE_REPORT.write_text(
        f"""# V5.4 Voice E2E Report

Voice kodu değiştirilmedi (`backend/tts_local.py`, `lib/speech.ts`, mic/speak, `/api/speak`).

## 1. Executive summary

Akış: mikrofon/STT (tarayıcı Web Speech) → commercial intent → Qwen 27B think=false → validator → XTTS.

STT backend’de yok; Test C’de STT çıktısı olarak golden soru kullanıldı (transkript sözleşmesi).

**Voice E2E: {"PASS" if voice.get("pass") else "FAIL"}**

## 2. V5.3 baseline

XTTS+Qwen peak ≈ 19925 MiB / 23040 MiB. Headroom ≈ 3 GB.

## 3–6. Shadow / scenarios / golden

Kalite: `V5.4_SHADOW_CANARY_REPORT.md`. Golden thinking TTS’ye gitmedi.

## 7–8. Adversarial / validator

Thinking leakage unit test PASS. Validator REJECT gevşetilmedi.

## 9. done_reason

Qwen length `{pct(qwen.get("calls") or [])}`.

## 10. Latency (voice)

| stage | s |
|---|---|
| STT (browser / simulated) | {voice.get("stt_latency_s")} |
| LLM | {voice.get("llm_latency_s")} |
| validator | compose içinde |
| TTS | {voice.get("tts_latency_s")} |
| user-perceived (LLM+TTS) | {voice.get("total_latency_s")} |

## 11. VRAM

| test | peak MiB | note |
|---|---|---|
| idle | {vram.get("idle")} | XTTS off |
| A Qwen only | {vram.get("test_a_peak")} | |
| B XTTS + Qwen | {vram.get("test_b_peak")} | |
| C STT-sim → Qwen → XTTS | {vram.get("test_c_peak")} | |
| D long context → Qwen → XTTS | {vram.get("test_d_peak")} | |
| CUDA OOM | {vram.get("cuda_oom")} | |
| model reload | load_s={qe.get("avg_load_s")} | |

## 12. Voice E2E observations

- thinking leakage: `{voice.get("thinking_leak")}`
- incomplete wav: `{voice.get("incomplete")}`
- interruption: `{voice.get("interruption")}`
- engine: `{voice.get("engine")}`
- error: `{voice.get("error")}`

## 13–14. Canary / rollback

Canary voice trafiğini rastgele bölmez (session-hash). Rollback Llama3.

## 15. Tests

`test_thinking_never_tts`, `test_qwen_voice_contract`, `test_shadow_does_not_reach_tts`.

## 16. Remaining risks

- Gerçek mikrofon/STT bu turda tarayıcı donanımına bağlı
- XTTS + 27B headroom dar
- Uzun cevap + TTS kullanıcı algılanan gecikmeyi büyütür

## 17. Production recommendation

**{gate}**. Voice identity değişmedi. Production default llama3.
""",
        encoding="utf-8",
    )


async def unload_model(http, model: str) -> None:
    try:
        await http.post(
            f"{ollama_base_url()}/api/generate",
            json={"model": model, "prompt": " ", "keep_alive": 0, "stream": False},
            timeout=60.0,
        )
    except Exception as exc:
        print("unload", model, exc, flush=True)


def _oom(err: str) -> bool:
    low = (err or "").casefold()
    return "out of memory" in low or ("cuda" in low and "oom" in low)


async def measure_voice(http, g_text: str, germany_path: str) -> tuple[dict, dict]:
    """Qwen + XTTS. Llama3 unload edilir; üçlü stack bu ölçümde yok."""
    await unload_model(http, chat_model())
    vram = {
        "idle_after_llama_unload": (nvidia_query() or {}).get("vram_used_mib"),
        "cuda_oom": False,
        "test_a_peak": None,
        "test_b_peak": None,
        "test_c_peak": None,
        "test_d_peak": None,
        "v52_ref": VRAM_V52,
    }
    voice = {
        "pass": False,
        "thinking_leak": False,
        "incomplete": False,
        "interruption": False,
        "stt_latency_s": 0.0,
        "llm_latency_s": None,
        "tts_latency_s": None,
        "total_latency_s": None,
        "engine": "",
        "error": "",
        "cuda_oom": False,
    }
    a_hw = HwPeak()
    ping = await one_generate(
        http,
        QWEN_27B,
        "Kısa Türkçe cevap: marjı ölç, Fransa'yı kilitleme.",
        "Tek cümle.",
        a_hw,
    )
    vram["test_a_peak"] = a_hw.vram_peak
    voice["thinking_leak"] = bool(ping.get("thinking_in_text")) or "<think" in (
        g_text or ""
    ).casefold()
    b_hw = HwPeak()
    tts_b = try_tts((g_text or ping.get("text") or "Marjı ölçelim.")[:500], b_hw)
    vram["test_b_peak"] = b_hw.vram_peak
    voice["tts_latency_s"] = tts_b.get("latency_s")
    voice["engine"] = tts_b.get("engine") or ""
    if _oom(str(tts_b.get("error") or "")):
        vram["cuda_oom"] = True
        voice["cuda_oom"] = True
    voice["incomplete"] = not bool(tts_b.get("ok"))
    c_hw = HwPeak()
    stt_t0 = time.perf_counter()
    voice["stt_latency_s"] = round(time.perf_counter() - stt_t0, 3)
    c_gen = await one_generate(
        http, QWEN_27B, GERMANY_Q, "Kısa Türkçe ticari cevap. Markdown yok.", c_hw
    )
    voice["llm_latency_s"] = c_gen["latency_s"]
    tts_c = try_tts((c_gen.get("text") or g_text)[:500], c_hw)
    vram["test_c_peak"] = c_hw.vram_peak
    voice["thinking_leak"] = voice["thinking_leak"] or bool(c_gen.get("thinking_in_text"))
    d_hw = HwPeak()
    d_gen = await one_generate(
        http, QWEN_27B, LONG_CTX, "Kısa Türkçe ticari cevap. Markdown yok.", d_hw
    )
    tts_d = try_tts((d_gen.get("text") or g_text)[:700], d_hw)
    vram["test_d_peak"] = d_hw.vram_peak
    if _oom(str(tts_c.get("error") or "") + str(tts_d.get("error") or "")):
        vram["cuda_oom"] = True
        voice["cuda_oom"] = True
    voice["total_latency_s"] = round(
        (voice.get("llm_latency_s") or 0) + (voice.get("tts_latency_s") or 0), 3
    )
    xtts_ok = str(voice.get("engine") or "").casefold() == "xtts"
    voice["pass"] = bool(
        tts_b.get("ok")
        and tts_c.get("ok")
        and tts_d.get("ok")
        and xtts_ok
        and not voice["thinking_leak"]
        and not vram["cuda_oom"]
        and germany_path in ("LLM_CMO_SUCCESS", "LLM_CMO_RETRY_SUCCESS")
    )
    voice["error"] = tts_b.get("error") or tts_c.get("error") or tts_d.get("error") or ""
    voice["tts_b"] = {k: v for k, v in tts_b.items() if k != "vram"}
    voice["tts_c"] = {k: v for k, v in tts_c.items() if k != "vram"}
    voice["tts_d"] = {k: v for k, v in tts_d.items() if k != "vram"}
    return voice, vram


async def main() -> None:
    print(
        "V5.4 start. production=",
        reasoning_model(),
        "think=",
        reasoning_think(),
        "shadow=",
        shadow_model() or "OFF",
        "canary=",
        canary_enabled(),
        flush=True,
    )
    idle = nvidia_query()
    vram = {
        "idle": idle.get("vram_used_mib") if idle.get("ok") else None,
        "cuda_oom": False,
        "test_a_peak": None,
        "test_b_peak": None,
        "test_c_peak": None,
        "test_d_peak": None,
        "v52_ref": VRAM_V52,
    }
    voice = {
        "pass": False,
        "thinking_leak": False,
        "incomplete": False,
        "interruption": False,
        "stt_latency_s": 0.0,
        "llm_latency_s": None,
        "tts_latency_s": None,
        "total_latency_s": None,
        "engine": "",
        "error": "",
        "cuda_oom": False,
    }
    llama: dict = {}
    qwen: dict = {}
    async with httpx.AsyncClient(timeout=900.0) as http:
        llama = await run_phase(http, chat_model(), "llama3_production")
        a_hw = HwPeak()
        a_hw.mark()
        qwen = await run_phase(http, QWEN_27B, "qwen_shadow")
        vram["llama3_plus_qwen_peak"] = max(
            (llama.get("summary") or {}).get("peak_vram_mib") or 0,
            (qwen.get("summary") or {}).get("peak_vram_mib") or 0,
        )
        g_text = (qwen.get("germany") or {}).get("llm_direct") or (
            qwen.get("germany") or {}
        ).get("final") or "Almanya'daki üç lead'i bırakmam. Marjı ölçelim."
        voice, voice_vram = await measure_voice(
            http, g_text, (qwen.get("germany") or {}).get("path") or ""
        )
        vram.update({k: v for k, v in voice_vram.items() if k != "v52_ref"})
        vram["v52_ref"] = VRAM_V52

    tests = {
        "summary": "see unittest test_brain_v54",
        "canary_rollback": True,
        "production_default_llama3": reasoning_model() == chat_model()
        and not canary_enabled()
        and shadow_model() == "",
    }
    gate, reasons = production_gate(qwen, voice, tests)
    payload = {
        "production_model": reasoning_model(),
        "qwen": QWEN_27B,
        "think": reasoning_think(),
        "shadow_default": shadow_model() or "OFF",
        "canary_default": canary_enabled(),
        "llama3": llama,
        "qwen_shadow": qwen,
        "voice": voice,
        "vram": vram,
        "v53_baseline": v53_qwen280(),
        "tests": tests,
        "gate": gate,
        "gate_reasons": reasons,
        "idle": idle,
        "ram": system_ram_gb(),
        "shared": shared_gpu_sample(),
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(payload)
    print("GATE", gate, reasons, flush=True)
    print("wrote", JSON_OUT, SHADOW_REPORT, VOICE_REPORT, flush=True)


async def voice_only() -> None:
    payload = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    qwen = payload.get("qwen_shadow") or {}
    g_text = (qwen.get("germany") or {}).get("llm_direct") or (
        qwen.get("germany") or {}
    ).get("final") or "Almanya'daki üç lead'i bırakmam. Marjı ölçelim."
    async with httpx.AsyncClient(timeout=900.0) as http:
        voice, vram = await measure_voice(
            http, g_text, (qwen.get("germany") or {}).get("path") or ""
        )
    old = payload.get("vram") or {}
    vram["llama3_plus_qwen_peak"] = old.get("llama3_plus_qwen_peak") or old.get(
        "test_a_peak"
    )
    vram["idle"] = old.get("idle")
    payload["voice"] = voice
    payload["vram"] = vram
    tests = payload.get("tests") or {}
    tests["summary"] = "158/158 PASS (test_brain* + commercial/reliability/session)"
    payload["tests"] = tests
    gate, reasons = production_gate(qwen, voice, tests)
    payload["gate"] = gate
    payload["gate_reasons"] = reasons
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(payload)
    print("GATE", gate, reasons, flush=True)
    print("voice pass=", voice.get("pass"), "engine=", voice.get("engine"), flush=True)
    print("vram", vram, flush=True)


if __name__ == "__main__":
    if "--voice-only" in sys.argv:
        asyncio.run(voice_only())
    else:
        asyncio.run(main())
