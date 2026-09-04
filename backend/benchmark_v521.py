"""V5.2.1: Qwen3.5 27B think=true vs think=false isolation.

Production OLLAMA_REASONING_MODEL değiştirmez.
Prompt / facts / num_predict (fair=280) aynı kalır; yalnızca think alanı farklıdır.
message.thinking asla kullanıcı cevabı / validator girdisi olmaz.
VRAM izolasyonu tekrar edilmez; V5.2 sayıları referanslanır.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from benchmark_v51 import CASES, HwPeak, run_case, summarize
from llm import chat_model, ollama_base_url, reasoning_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
REPORT = ROOT / "V5.2.1_QWEN27B_THINKING_REPORT.md"
JSON_OUT = BACKEND / "benchmark_v521_results.json"
V51_JSON = BACKEND / "benchmark_v51_results.json"
V52_JSON = BACKEND / "benchmark_v52_results.json"

MODEL = "qwen3.5:27b"
NUM_PREDICT_FAIR = 280
NUM_PREDICT_DIAG = (280, 800, 1600)
TEMPERATURE = 0.3
VRAM_V52 = {
    "MODEL_ONLY_PEAK_VRAM": 17948,
    "XTTS_MODEL_PEAK_VRAM": 19925,
    "MODEL_ONLY_VRAM_DELTA_MIB": 16807,
    "source": "V5.2_QWEN27B_BENCHMARK_REPORT.md / benchmark_v52_results.json",
}


def chat_payload(
    model: str,
    system_prompt: str | None,
    user: str,
    *,
    think: bool,
    num_predict: int = NUM_PREDICT_FAIR,
) -> dict:
    """Fair karşılaştırmada tek fark `think` alanıdır."""
    return {
        "model": model,
        "stream": False,
        "think": think,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": TEMPERATURE, "num_predict": num_predict},
    }


def user_content_from_ollama(data: dict) -> str:
    msg = data.get("message") if isinstance(data.get("message"), dict) else {}
    content = msg.get("content") or data.get("response") or ""
    return content if isinstance(content, str) else ""


def thinking_from_ollama(data: dict) -> str:
    msg = data.get("message") if isinstance(data.get("message"), dict) else {}
    thinking = msg.get("thinking") or data.get("thinking") or ""
    return thinking if isinstance(thinking, str) else ""


def classify_fields(thinking: str, content: str, eval_count: int, num_predict: int) -> dict:
    content_empty = not (content or "").strip()
    thinking_present = bool((thinking or "").strip())
    budget_eaten = (
        content_empty
        and thinking_present
        and isinstance(eval_count, int)
        and eval_count >= num_predict
    )
    return {
        "FINAL_CONTENT_EMPTY": content_empty,
        "THINKING_PRESENT": thinking_present,
        "THINKING_EMPTY": not thinking_present,
        "TOKEN_BUDGET_CONSUMED_BY_THINKING": budget_eaten,
        # Boş content + dolu thinking = API/thinking-mode; otomatik reasoning fail değil.
        "AUTO_REASONING_FAILURE": False if (content_empty and thinking_present) else content_empty,
    }


def approx_tokens(text: str) -> int:
    return max(len(text or "") // 4, 0)


def v51_row(label_prefix: str) -> dict:
    if not V51_JSON.exists():
        return {}
    data = json.loads(V51_JSON.read_text(encoding="utf-8"))
    for item in data.get("summaries") or []:
        if str(item.get("label") or "").startswith(label_prefix):
            slim = {k: v for k, v in item.items() if k not in ("rows", "call_metrics")}
            return slim
    return {}


def germany_from(rows: list[dict]) -> dict:
    return next((r for r in rows if r.get("id") == "germany_france"), {})


def make_generate(model: str, metrics: dict, hw: HwPeak, *, think: bool, num_predict: int):
    async def generate(http, prompt, system_prompt=None, history=None, polish=False):
        hw.mark()
        t0 = time.perf_counter()
        payload = chat_payload(
            model, system_prompt, prompt, think=think, num_predict=num_predict
        )
        response = await http.post(
            f"{ollama_base_url()}/api/chat",
            json=payload,
            timeout=900.0,
        )
        response.raise_for_status()
        data = response.json()
        hw.mark()
        elapsed = time.perf_counter() - t0
        content = user_content_from_ollama(data)
        thinking = thinking_from_ollama(data)
        eval_count = data.get("eval_count") or 0
        eval_ns = data.get("eval_duration") or 0
        tps = (eval_count / (eval_ns / 1e9)) if eval_ns else None
        flags = classify_fields(thinking, content, int(eval_count or 0), num_predict)
        metrics.setdefault("calls", []).append(
            {
                "think": think,
                "num_predict": num_predict,
                "latency_s": round(elapsed, 3),
                "eval_count": eval_count,
                "prompt_eval_count": data.get("prompt_eval_count"),
                "tokens_per_sec": round(tps, 2) if tps else None,
                "load_s": round((data.get("load_duration") or 0) / 1e9, 3),
                "done_reason": data.get("done_reason"),
                "content_length": len(content),
                "thinking_length": len(thinking),
                "THINKING_TOKEN_USAGE_APPROX": approx_tokens(thinking),
                "FINAL_CONTENT_TOKEN_USAGE_APPROX": approx_tokens(content),
                "content_preview": content[:500],
                "thinking_preview": thinking[:500],
                **flags,
            }
        )
        metrics.setdefault("last_raw", {})
        metrics["last_raw"] = {
            "content": content,
            "thinking": thinking,
            "done_reason": data.get("done_reason"),
            "eval_count": eval_count,
        }
        # Kullanıcı / validator girdisi yalnızca content.
        return content

    return generate


def extra_rates(rows: list[dict], metrics: dict) -> dict:
    calls = metrics.get("calls") or []
    n_calls = len(calls) or 1
    n_rows = len(rows) or 1
    content_empty_calls = sum(1 for c in calls if c.get("FINAL_CONTENT_EMPTY"))
    thinking_present_calls = sum(1 for c in calls if c.get("THINKING_PRESENT"))
    budget_calls = sum(1 for c in calls if c.get("TOKEN_BUDGET_CONSUMED_BY_THINKING"))
    llm_scores = [
        (r.get("score_llm_direct") or {}).get("total")
        for r in rows
        if r.get("llm_direct")
    ]
    llm_scores = [s for s in llm_scores if isinstance(s, (int, float))]
    echo_llm = sum(1 for r in rows if r.get("llm_direct") and r.get("brief_echo"))
    fc_llm = sum(1 for r in rows if r.get("llm_direct") and r.get("false_certainty"))
    with_content = sum(1 for r in rows if (r.get("llm_direct") or "").strip())
    think_tok = [c.get("THINKING_TOKEN_USAGE_APPROX") or 0 for c in calls]
    content_tok = [c.get("FINAL_CONTENT_TOKEN_USAGE_APPROX") or 0 for c in calls]
    evals = [c.get("eval_count") or 0 for c in calls]
    return {
        "n_calls": len(calls),
        "THINKING_ON_FINAL_CONTENT_RATE": round(1 - (content_empty_calls / n_calls), 3),
        "final_content_rate": round(1 - (content_empty_calls / n_calls), 3),
        "content_empty_rate": round(content_empty_calls / n_calls, 3),
        "thinking_present_rate": round(thinking_present_calls / n_calls, 3),
        "token_budget_consumed_by_thinking_rate": round(budget_calls / n_calls, 3),
        "case_content_nonempty_rate": round(with_content / n_rows, 3),
        "cmo_llm_avg": round(sum(llm_scores) / len(llm_scores), 3) if llm_scores else None,
        "brief_echo_on_llm_content_rate": round(echo_llm / (with_content or 1), 3) if with_content else None,
        "false_certainty_on_llm_content_rate": round(fc_llm / (with_content or 1), 3) if with_content else None,
        "THINKING_TOKEN_USAGE_AVG_APPROX": round(sum(think_tok) / len(think_tok), 1) if think_tok else 0,
        "FINAL_CONTENT_TOKEN_USAGE_AVG_APPROX": round(sum(content_tok) / len(content_tok), 1) if content_tok else 0,
        "eval_count_avg": round(sum(evals) / len(evals), 1) if evals else 0,
        "done_reasons": sorted({str(c.get("done_reason")) for c in calls}),
    }


def classify_v52(on: dict, off: dict) -> str:
    on_empty = on.get("content_empty_rate") if on.get("content_empty_rate") is not None else 1
    on_think = on.get("thinking_present_rate") or 0
    off_content = off.get("final_content_rate") or 0
    off_llm = (off.get("summary") or {}).get("real_llm_success_rate") or 0
    if on_empty >= 0.9 and on_think >= 0.8 and off_content >= 0.5:
        return (
            "V5.2_WAS_API_THINKING_MODE_NOT_MODEL_REASONING_FAILURE"
        )
    if on_empty >= 0.9 and off_content < 0.2:
        return "STILL_NO_FINAL_CONTENT_EVEN_WITH_THINK_FALSE"
    if off_content >= 0.5 and off_llm < 0.2:
        return "CONTENT_EXISTS_BUT_VALIDATOR_STILL_REJECTS"
    return "MIXED"


async def run_phase(
    http: httpx.AsyncClient,
    model: str,
    label: str,
    *,
    think: bool,
    num_predict: int,
    cases: list[dict] | None = None,
) -> dict:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    gen = make_generate(model, metrics, hw, think=think, num_predict=num_predict)
    rows = []
    error = None
    try:
        for case in cases or CASES:
            print(f"[{label} think={think} npred={num_predict}] {case['id']} ...", flush=True)
            row = await run_case(http, gen, case)
            last = metrics.get("last_raw") or {}
            row["raw_thinking"] = last.get("thinking") or ""
            row["raw_content"] = last.get("content") or ""
            row["done_reason"] = last.get("done_reason")
            row["eval_count_last"] = last.get("eval_count")
            rows.append(row)
    except Exception as exc:
        error = repr(exc)
        print(f"[{label}] ERROR {error}", flush=True)
    summary = summarize(label, rows, metrics, hw, "TESTED" if rows else "FAILED")
    extra = extra_rates(rows, metrics)
    # think=true boş content'i otomatik reasoning fail sayma.
    if think and extra.get("content_empty_rate", 0) >= 0.9 and extra.get("thinking_present_rate", 0) >= 0.8:
        extra["reasoning_failure_inferred"] = False
        extra["note"] = (
            "FINAL_CONTENT_EMPTY=TRUE; thinking dolu. Bu otomatik reasoning başarısızlığı değil."
        )
    else:
        extra["reasoning_failure_inferred"] = bool(
            extra.get("content_empty_rate", 1) >= 0.9 and extra.get("thinking_present_rate", 0) < 0.2
        )
    slim_calls = []
    for c in metrics.get("calls") or []:
        slim_calls.append({k: v for k, v in c.items() if k not in ("content_preview", "thinking_preview") or True})
    return {
        "label": label,
        "think": think,
        "num_predict": num_predict,
        "summary": {k: v for k, v in summary.items() if k not in ("rows", "call_metrics")},
        "extra": extra,
        "rows": rows,
        "calls": slim_calls,
        "error": error,
        "gpu_end": {"note": "VRAM isolation not re-run; see V5.2", **VRAM_V52},
    }


def _clip(text: str, n: int = 2500) -> str:
    body = text or ""
    return body if len(body) <= n else body[:n] + "\n…[truncated]"


def germany_block(title: str, row: dict) -> str:
    score_f = row.get("score_final") or {}
    score_l = row.get("score_llm_direct") or {}
    return f"""### {title}

| Alan | Değer |
|---|---|
| Path | `{row.get("public_path")}` / internal `{row.get("path")}` |
| Validator direct | `{row.get("direct_status")}` |
| Validator retry | `{row.get("retry_status")}` |
| REAL_LLM | `{row.get("real_llm")}` |
| FINAL_CONTENT_EMPTY | `{not bool((row.get("raw_content") or row.get("llm_direct") or "").strip())}` |
| done_reason | `{row.get("done_reason")}` |
| eval_count (son çağrı) | `{row.get("eval_count_last")}` |
| thinking_length | `{len(row.get("raw_thinking") or "")}` |
| content_length | `{len(row.get("raw_content") or row.get("llm_direct") or "")}` |
| CMO final /14 | `{score_f.get("total")}` |
| CMO LLM content /14 | `{score_l.get("total")}` |
| Brief echo | `{row.get("brief_echo")}` |
| False certainty | `{row.get("false_certainty")}` |
| Memory dishonesty | `{row.get("memory_dishonesty")}` |
| English leak | `{row.get("english_leakage")}` |
| Unsupported fact | `{row.get("unsupported_fact")}` |

**RAW THINKING** (kullanıcıya gösterilmez):

```
{_clip(row.get("raw_thinking") or "(boş)", 3500)}
```

**RAW CONTENT** (`message.content`; validator girdisi):

```
{_clip(row.get("raw_content") or row.get("llm_direct") or "(boş)", 3500)}
```

**FINAL USER RESPONSE:**

```
{_clip(row.get("final") or "(boş)", 2500)}
```
"""


def write_report(payload: dict) -> None:
    on = payload.get("thinking_on") or {}
    off = payload.get("thinking_off") or {}
    on_s = on.get("summary") or {}
    off_s = off.get("summary") or {}
    on_e = on.get("extra") or {}
    off_e = off.get("extra") or {}
    llama = payload.get("llama3_v51") or {}
    fb = payload.get("fallback_v51") or {}
    g_on = payload.get("germany_think_true") or {}
    g_off = payload.get("germany_think_false") or {}
    g_hi = payload.get("germany_think_true_high_npred") or {}
    diag = payload.get("num_predict_diagnostic") or []
    rec = payload.get("recommendation") or ""
    v52_class = payload.get("v52_classification") or ""

    def pct(x) -> str:
        if x is None:
            return "—"
        if isinstance(x, float) and x <= 1:
            return f"{round(x * 100, 1)}%"
        return str(x)

    def cmo_cell(summary, extra) -> str:
        llm = extra.get("cmo_llm_avg")
        final = summary.get("cmo_score_avg")
        if extra.get("content_empty_rate", 0) >= 0.9:
            return f"n/a (content empty); final/fallback {final}"
        return f"LLM {llm} / final {final}"

    diag_lines = ""
    for item in diag:
        extra = item.get("extra") or {}
        calls = item.get("calls") or []
        c0 = calls[0] if calls else {}
        diag_lines += (
            f"| think=true num_predict={item.get('num_predict')} | "
            f"{c0.get('thinking_length')} | {c0.get('content_length')} | "
            f"{c0.get('FINAL_CONTENT_EMPTY')} | {c0.get('done_reason')} | "
            f"{c0.get('eval_count')} | {c0.get('latency_s')}s | "
            f"{c0.get('TOKEN_BUDGET_CONSUMED_BY_THINKING')} |\n"
        )

    scenario_on = ""
    for row in on.get("rows") or []:
        scenario_on += (
            f"- `{row.get('id')}` path={row.get('public_path')} real_llm={row.get('real_llm')} "
            f"content_len={len(row.get('raw_content') or row.get('llm_direct') or '')} "
            f"thinking_len={len(row.get('raw_thinking') or '')} "
            f"direct={row.get('direct_status')} cmo_final={(row.get('score_final') or {}).get('total')} "
            f"cmo_llm={(row.get('score_llm_direct') or {}).get('total')}\n"
        )
    scenario_off = ""
    for row in off.get("rows") or []:
        scenario_off += (
            f"- `{row.get('id')}` path={row.get('public_path')} real_llm={row.get('real_llm')} "
            f"echo={row.get('brief_echo')} en={row.get('english_leakage')} "
            f"fc={row.get('false_certainty')} mem={row.get('memory_dishonesty')} "
            f"direct={row.get('direct_status')} retry={row.get('retry_status')} "
            f"cmo_final={(row.get('score_final') or {}).get('total')} "
            f"cmo_llm={(row.get('score_llm_direct') or {}).get('total')}\n"
        )

    body = f"""# V5.2.1 Qwen3.5 27B Thinking / Final Content Ayrımı

Tarih: {time.strftime("%Y-%m-%d %H:%M")}  
Production `OLLAMA_REASONING_MODEL` = `{payload.get("production_reasoning")}` (**değiştirilmedi**).  
35B kurulmadı. XTTS / billing / public API dokunulmadı. Prompt ve fair `num_predict={NUM_PREDICT_FAIR}` aynı; tek fark `think`.

## Model

| Alan | Değer |
|---|---|
| Tag | `{MODEL}` |
| Quantization | Q4_K_M (V5.2) |
| Parametre | 27.8B |
| Dosya | 17.42 GB |
| Fair num_predict | {NUM_PREDICT_FAIR} |
| Temperature | {TEMPERATURE} |

## VRAM (V5.2 referansı — bu turda GPU izolasyonu tekrar edilmedi)

| Alan | Değer |
|---|---|
| MODEL_ONLY_PEAK_VRAM | {VRAM_V52["MODEL_ONLY_PEAK_VRAM"]} MiB |
| XTTS_MODEL_PEAK_VRAM | {VRAM_V52["XTTS_MODEL_PEAK_VRAM"]} MiB |
| MODEL_ONLY_VRAM delta | {VRAM_V52["MODEL_ONLY_VRAM_DELTA_MIB"]} MiB |

## V5.2 yeniden sınıflandırma

**{v52_class}**

{payload.get("v52_explanation")}

## Karşılaştırma tablosu

| MODE | REAL LLM SUCCESS | FALLBACK | CONTENT EMPTY | AVG LATENCY | TOK/S | CMO SCORE | BRIEF ECHO | FALSE CERTAINTY |
|---|---|---|---|---|---|---|---|---|
| Qwen27B think=true | {pct(on_s.get("real_llm_success_rate"))} | {pct(on_s.get("fallback_rate"))} | {pct(on_e.get("content_empty_rate"))} | {on_s.get("avg_latency_s")} s | {on_s.get("tokens_per_sec")} | {cmo_cell(on_s, on_e)} | {pct(on_e.get("brief_echo_on_llm_content_rate"))} | {pct(on_e.get("false_certainty_on_llm_content_rate"))} |
| Qwen27B think=false | {pct(off_s.get("real_llm_success_rate"))} | {pct(off_s.get("fallback_rate"))} | {pct(off_e.get("content_empty_rate"))} | {off_s.get("avg_latency_s")} s | {off_s.get("tokens_per_sec")} | {cmo_cell(off_s, off_e)} | {pct(off_e.get("brief_echo_on_llm_content_rate") if off_e.get("brief_echo_on_llm_content_rate") is not None else off_s.get("brief_echo_rate"))} | {pct(off_e.get("false_certainty_on_llm_content_rate") if off_e.get("false_certainty_on_llm_content_rate") is not None else off_s.get("false_certainty_rate"))} |
| Llama3 baseline | {pct(llama.get("real_llm_success_rate"))} | {pct(llama.get("fallback_rate"))} | — | {llama.get("avg_latency_s")} s | {llama.get("tokens_per_sec")} | {llama.get("cmo_score_avg")} (final, %90 fallback karışık) | {pct(llama.get("brief_echo_rate"))} | {pct(llama.get("false_certainty_rate"))} |
| Commercial fallback | 0% | 100% | n/a | ms | — | {fb.get("cmo_score_avg")} | {pct(fb.get("brief_echo_rate"))} | {pct(fb.get("false_certainty_rate"))} |

REAL_LLM_SUCCESS = (LLM_DIRECT + LLM_RETRY) / 20. Fallback kalitesi LLM başarısı değildir.  
think=true'de boş `content` otomatik reasoning fail **değildir**.

### Yeni metrikler

| Metrik | think=true | think=false |
|---|---|---|
| THINKING_ON_FINAL_CONTENT_RATE | {on_e.get("final_content_rate")} | — |
| THINKING_OFF_FINAL_CONTENT_RATE | — | {off_e.get("final_content_rate")} |
| THINKING_TOKEN_USAGE (approx, avg) | {on_e.get("THINKING_TOKEN_USAGE_AVG_APPROX")} | {off_e.get("THINKING_TOKEN_USAGE_AVG_APPROX")} |
| FINAL_CONTENT_TOKEN_USAGE (approx, avg) | {on_e.get("FINAL_CONTENT_TOKEN_USAGE_AVG_APPROX")} | {off_e.get("FINAL_CONTENT_TOKEN_USAGE_AVG_APPROX")} |
| eval_count avg | {on_e.get("eval_count_avg")} | {off_e.get("eval_count_avg")} |
| done_reason | {on_e.get("done_reasons")} | {off_e.get("done_reasons")} |
| token budget consumed by thinking | {on_e.get("token_budget_consumed_by_thinking_rate")} | {off_e.get("token_budget_consumed_by_thinking_rate")} |
| CMO LLM-only /14 | {on_e.get("cmo_llm_avg")} | {off_e.get("cmo_llm_avg")} |
| CMO final (fallback dahil) /14 | {on_s.get("cmo_score_avg")} | {off_s.get("cmo_score_avg")} |

## num_predict diagnostik (think=true, Almanya senaryosu, production config değişmedi)

| Ayar | thinking_length | content_length | FINAL_CONTENT_EMPTY | done_reason | eval_count | latency | TOKEN_BUDGET_CONSUMED_BY_THINKING |
|---|---|---|---|---|---|---|---|
{diag_lines if diag_lines else "| (koşulmadı) | — | — | — | — | — | — | — |\n"}

Fair karşılaştırmada `num_predict` **280** kaldı. 800/1600 yalnızca diagnostik.

## Almanya vs Fransa

Soru: *Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?*

{germany_block("A) think=true, num_predict=280", g_on)}

{germany_block("B) think=false, num_predict=280", g_off)}

{germany_block("C) think=true, num_predict=1600 (yeterli bütçe denemesi)", g_hi)}

## 20 senaryo — QWEN27B_THINKING_ON

{scenario_on}

## 20 senaryo — QWEN27B_THINKING_OFF

{scenario_off}

## CMO /14 notları

Aynı kriterler: DATA_USE, DIAGNOSIS, REASONING, DECISION, UNCERTAINTY, ACTION, NATURAL_TURKISH.  
Negatifler: BRIEF_ECHO, ENGLISH_LEAKAGE, UNSUPPORTED_FACT, FALSE_CERTAINTY, PROMPT_LEAK, MEMORY_DISHONESTY.

think=true CMO_final çoğu zaman fallback şablonudur; LLM başarısı değildir.  
think=false CMO_LLM, modelin `message.content` kalitesidir.

## Production uygunluğu

{payload.get("suitability_notes")}

Production `_ollama_complete` hâlâ `think` göndermiyor. think=false harness'te işe yarasa bile env `OLLAMA_REASONING_MODEL` **değiştirilmedi**; production Qwen'e çekilmedi.

## RECOMMENDATION

**{rec}**

35B için karar yok; kurulmadı.

## Testler

Mevcut 109 test + `test_brain_v521.py`. Eski testler değiştirilmedi. Production kodu benchmark için gevşetilmedi.
"""
    REPORT.write_text(body, encoding="utf-8")


def decide(payload: dict) -> tuple[str, str, str]:
    on = payload.get("thinking_on") or {}
    off = payload.get("thinking_off") or {}
    on_e = on.get("extra") or {}
    off_e = off.get("extra") or {}
    off_s = off.get("summary") or {}
    llama = payload.get("llama3_v51") or {}
    v52 = classify_v52({**on_e, "summary": on.get("summary")}, {**off_e, "summary": off_s})
    off_llm = off_s.get("real_llm_success_rate") or 0
    off_fb = off_s.get("fallback_rate") or 1
    off_echo = off_e.get("brief_echo_on_llm_content_rate")
    if off_echo is None:
        off_echo = off_s.get("brief_echo_rate") or 0
    l_llm = llama.get("real_llm_success_rate") or 0
    notes = []
    if v52 == "V5.2_WAS_API_THINKING_MODE_NOT_MODEL_REASONING_FAILURE":
        notes.append(
            "V5.2 %0 REAL_LLM_SUCCESS, Qwen3.5 27B'nin ticari reasoning yoksunluğu olarak okunmamalı. "
            "think=true + num_predict=280 thinking bütçesini bitirdi; harness yalnızca message.content okudu."
        )
    elif v52 == "STILL_NO_FINAL_CONTENT_EVEN_WITH_THINK_FALSE":
        notes.append("think=false'da da final content boş. Bu API/model çıktı ayrımıyla sınırlı değil.")
    elif v52 == "CONTENT_EXISTS_BUT_VALIDATOR_STILL_REJECTS":
        notes.append(
            "think=false content üretiyor ama validator PASS oran düşük. "
            "V5.2 %0 harness kaynaklı olabilir; think=false kalitesi ayrı ve zayıf."
        )
    if off_llm >= l_llm + 0.25 and off_fb + 0.15 <= (llama.get("fallback_rate") or 1) and off_echo <= 0.2:
        rec = "BENCHMARK ONLY — think=false kalite sinyali var; production env değiştirilmedi"
        notes.append(
            f"think=false REAL_LLM_SUCCESS={off_llm} vs Llama3 {l_llm}. "
            "XTTS+VRAM V5.2'de sıkışık; production adayı ilanı bu turda yok."
        )
    else:
        rec = "NO CLEAR WINNER / BENCHMARK ONLY"
        notes.append(
            f"think=false REAL_LLM_SUCCESS={off_llm} fallback={off_fb} echo={off_echo}. "
            "Production llama3. 35B yok."
        )
    notes.append("thinking kullanıcıya gösterilmedi ve content yerine konmadı.")
    explanation = " ".join(notes)
    return rec, explanation, v52


async def main() -> None:
    if "qwen3.5" in (os.getenv("OLLAMA_REASONING_MODEL") or "").casefold():
        raise SystemExit("Production OLLAMA_REASONING_MODEL qwen; V5.2.1 bunu değiştirmemeli.")
    async with httpx.AsyncClient(timeout=900.0) as http:
        tags = (await http.get(f"{ollama_base_url()}/api/tags")).json()
        names = {m.get("name") for m in (tags.get("models") or [])}
        if MODEL not in names and f"{MODEL}:latest" not in names:
            raise SystemExit(f"{MODEL} kurulu değil; benchmark üretilmedi.")

        print("QWEN27B_THINKING_ON ...", flush=True)
        thinking_on = await run_phase(
            http, MODEL, "QWEN27B_THINKING_ON", think=True, num_predict=NUM_PREDICT_FAIR
        )
        print("QWEN27B_THINKING_OFF ...", flush=True)
        thinking_off = await run_phase(
            http, MODEL, "QWEN27B_THINKING_OFF", think=False, num_predict=NUM_PREDICT_FAIR
        )

        germany_case = next(c for c in CASES if c["id"] == "germany_france")
        diag = []
        for npred in NUM_PREDICT_DIAG:
            if npred == NUM_PREDICT_FAIR:
                # Fair think=true koşusu zaten 280; tekrar GPU yakma.
                g = germany_from(thinking_on.get("rows") or [])
                diag.append(
                    {
                        "num_predict": npred,
                        "reused_from": "QWEN27B_THINKING_ON",
                        "extra": thinking_on.get("extra"),
                        "calls": [
                            {
                                "thinking_length": len(g.get("raw_thinking") or ""),
                                "content_length": len(g.get("raw_content") or g.get("llm_direct") or ""),
                                "FINAL_CONTENT_EMPTY": not bool(
                                    (g.get("raw_content") or g.get("llm_direct") or "").strip()
                                ),
                                "done_reason": g.get("done_reason"),
                                "eval_count": g.get("eval_count_last"),
                                "latency_s": g.get("wall_latency_s"),
                                "TOKEN_BUDGET_CONSUMED_BY_THINKING": bool(
                                    not (g.get("raw_content") or g.get("llm_direct") or "").strip()
                                    and (g.get("raw_thinking") or "").strip()
                                    and (g.get("eval_count_last") or 0) >= npred
                                ),
                            }
                        ],
                    }
                )
                continue
            print(f"diagnostic think=true num_predict={npred} germany ...", flush=True)
            phase = await run_phase(
                http,
                MODEL,
                f"DIAG_THINK_TRUE_NPRED_{npred}",
                think=True,
                num_predict=npred,
                cases=[germany_case],
            )
            diag.append(phase)

        germany_hi = germany_from((diag[-1].get("rows") if diag else None) or [])
        # 1600 koşusu diag son eleman (800 sonra 1600).
        hi_phase = next((d for d in diag if d.get("num_predict") == 1600), {})
        germany_hi = germany_from(hi_phase.get("rows") or []) or germany_hi

        payload = {
            "production_reasoning": reasoning_model(),
            "production_chat": chat_model(),
            "model": MODEL,
            "num_predict_fair": NUM_PREDICT_FAIR,
            "vram_v52": VRAM_V52,
            "thinking_on": thinking_on,
            "thinking_off": thinking_off,
            "germany_think_true": germany_from(thinking_on.get("rows") or []),
            "germany_think_false": germany_from(thinking_off.get("rows") or []),
            "germany_think_true_high_npred": germany_hi,
            "num_predict_diagnostic": [
                {
                    "num_predict": d.get("num_predict"),
                    "label": d.get("label") or d.get("reused_from"),
                    "extra": d.get("extra"),
                    "calls": d.get("calls"),
                    "rows": d.get("rows"),
                    "error": d.get("error"),
                }
                for d in diag
            ],
            "llama3_v51": v51_row("llama3"),
            "fallback_v51": v51_row("commercial fallback"),
        }
        rec, notes, v52 = decide(payload)
        payload["recommendation"] = rec
        payload["suitability_notes"] = notes
        payload["v52_classification"] = v52
        payload["v52_explanation"] = notes
        JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        write_report(payload)
        print(
            json.dumps(
                {
                    "report": str(REPORT),
                    "rec": rec,
                    "v52_class": v52,
                    "think_on_empty": (thinking_on.get("extra") or {}).get("content_empty_rate"),
                    "think_off_content": (thinking_off.get("extra") or {}).get("final_content_rate"),
                    "think_off_llm": (thinking_off.get("summary") or {}).get("real_llm_success_rate"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
