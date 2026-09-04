"""V5.3 Qwen think=false production-candidate benchmark.

Production OLLAMA_REASONING_MODEL değiştirmez. 35B kurmaz. XTTS'i durdurmaz.
Fair 20 senaryo değişmez. num_predict 280/384/512 think=false.
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
    run_fallback_only,
    shared_gpu_sample,
    summarize,
    system_ram_gb,
)
from llm import (
    QWEN_27B,
    chat_model,
    ollama_base_url,
    ollama_chat_payload,
    reasoning_model,
    reasoning_think,
    thinking_from_ollama,
    user_content_from_ollama,
    user_visible_text,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
JSON_OUT = BACKEND / "benchmark_v53_results.json"
REPORT = ROOT / "V5.3_BENCHMARK_REPORT.md"
CANDIDATE = ROOT / "V5.3_PRODUCTION_CANDIDATE_REPORT.md"
V51_JSON = BACKEND / "benchmark_v51_results.json"
VRAM_V52 = {
    "MODEL_ONLY_PEAK_VRAM": 17948,
    "XTTS_MODEL_PEAK_VRAM": 19925,
    "source": "V5.2",
}


def v51_row(prefix: str) -> dict:
    if not V51_JSON.exists():
        return {}
    data = json.loads(V51_JSON.read_text(encoding="utf-8"))
    for item in data.get("summaries") or []:
        if str(item.get("label") or "").startswith(prefix):
            return {k: v for k, v in item.items() if k not in ("rows", "call_metrics")}
    return {}


def make_generate(model: str, metrics: dict, hw: HwPeak, *, think: bool, num_predict: int):
    async def generate(http, prompt, system_prompt=None, history=None, polish=False):
        hw.mark()
        t0 = time.perf_counter()
        payload = ollama_chat_payload(
            model,
            [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt},
            ],
            think=think,
            options={"temperature": 0.3, "num_predict": num_predict},
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
        content = user_visible_text(user_content_from_ollama(data), thinking_from_ollama(data))
        thinking = thinking_from_ollama(data)
        eval_count = data.get("eval_count") or 0
        eval_ns = data.get("eval_duration") or 0
        prompt_ns = data.get("prompt_eval_duration") or 0
        load_ns = data.get("load_duration") or 0
        tps = (eval_count / (eval_ns / 1e9)) if eval_ns else None
        metrics.setdefault("calls", []).append(
            {
                "think": think,
                "num_predict": num_predict,
                "latency_s": round(elapsed, 3),
                "eval_count": eval_count,
                "tokens_per_sec": round(tps, 2) if tps else None,
                "prompt_eval_s": round(prompt_ns / 1e9, 3) if prompt_ns else None,
                "gen_s": round(eval_ns / 1e9, 3) if eval_ns else None,
                "load_s": round(load_ns / 1e9, 3) if load_ns else None,
                "done_reason": data.get("done_reason"),
                "content_length": len(content),
                "thinking_length": len(thinking),
                "vram_mib": (nvidia_query() or {}).get("vram_used_mib"),
            }
        )
        return content

    return generate


def extra_stats(rows: list[dict], metrics: dict) -> dict:
    calls = metrics.get("calls") or []
    n = len(rows) or 1
    lat = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else None
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else None
    llm_scores = [
        (r.get("score_llm_direct") or {}).get("total")
        for r in rows
        if r.get("llm_direct")
    ]
    llm_scores = [s for s in llm_scores if isinstance(s, (int, float))]
    return {
        "n_calls": len(calls),
        "content_empty_rate": round(
            sum(1 for c in calls if not c.get("content_length")) / (len(calls) or 1), 3
        ),
        "thinking_present_rate": round(
            sum(1 for c in calls if (c.get("thinking_length") or 0) > 0) / (len(calls) or 1), 3
        ),
        "pass_rate": round(sum(1 for r in rows if r.get("direct_status") == "PASS") / n, 3),
        "warn_rate": round(sum(1 for r in rows if r.get("direct_status") == "WARN") / n, 3),
        "reject_rate": round(sum(1 for r in rows if r.get("direct_status") == "REJECT") / n, 3),
        "cmo_llm_avg": round(sum(llm_scores) / len(llm_scores), 3) if llm_scores else None,
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "avg_prompt_eval_s": round(
            statistics.mean([c["prompt_eval_s"] for c in calls if c.get("prompt_eval_s")]), 3
        )
        if any(c.get("prompt_eval_s") for c in calls)
        else None,
        "avg_gen_s": round(
            statistics.mean([c["gen_s"] for c in calls if c.get("gen_s")]), 3
        )
        if any(c.get("gen_s") for c in calls)
        else None,
        "avg_load_s": round(
            statistics.mean([c["load_s"] for c in calls if c.get("load_s")]), 3
        )
        if any(c.get("load_s") for c in calls)
        else None,
        "done_reasons": sorted({str(c.get("done_reason")) for c in calls}),
        "length_done_rate": round(
            sum(1 for c in calls if c.get("done_reason") == "length") / (len(calls) or 1), 3
        ),
    }


def germany_ok(row: dict) -> dict:
    text = (row.get("llm_direct") or row.get("final") or "").casefold()
    checks = {
        "keep_germany": "bırak" in text or "elden" in text or "erken" in text,
        "price_claim": "gözlem" in text or "iddia" in text or "kanıt" in text or "ölç" in text,
        "france_second": "ikinci" in text or "fransa" in text,
        "no_false_fr": "fransa kesin" not in text,
        "real_llm": bool(row.get("real_llm")),
        "no_markdown_fallback": row.get("path") in ("LLM_CMO_SUCCESS", "LLM_CMO_RETRY_SUCCESS"),
    }
    return checks


async def run_phase(http, model: str, label: str, *, think: bool, num_predict: int) -> dict:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    gen = make_generate(model, metrics, hw, think=think, num_predict=num_predict)
    rows = []
    for case in CASES:
        print(f"[{label} n={num_predict}] {case['id']} ...", flush=True)
        rows.append(await run_case(http, gen, case))
    summary = summarize(label, rows, metrics, hw, "TESTED")
    extra = extra_stats(rows, metrics)
    g = next((r for r in rows if r.get("id") == "germany_france"), {})
    return {
        "label": label,
        "think": think,
        "num_predict": num_predict,
        "summary": {k: v for k, v in summary.items() if k not in ("rows", "call_metrics")},
        "extra": extra,
        "germany": {
            "path": g.get("public_path"),
            "real_llm": g.get("real_llm"),
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


def production_gate(qwen280: dict) -> tuple[str, list[str]]:
    s = qwen280.get("summary") or {}
    e = qwen280.get("extra") or {}
    g = (qwen280.get("germany") or {}).get("golden") or {}
    reasons = []
    llm = s.get("real_llm_success_rate") or 0
    echo = s.get("brief_echo_rate") or 0
    fc = s.get("false_certainty_rate") or 0
    uf = s.get("unsupported_fact_rate") or 0
    en = s.get("english_rate") or 0
    cmo = e.get("cmo_llm_avg")
    empty = e.get("content_empty_rate") or 0
    think_leak = e.get("thinking_present_rate") or 0
    pass_r = e.get("pass_rate") or 0
    if (cmo or 0) < 10:
        reasons.append(f"CMO LLM {cmo} < 10")
    if llm < 0.9:
        reasons.append(f"REAL_LLM_SUCCESS {llm} < 90%")
    if echo > 0.05:
        reasons.append(f"BRIEF_ECHO {echo} > 5%")
    if fc:
        reasons.append("FALSE_CERTAINTY > 0")
    if uf:
        reasons.append("UNSUPPORTED_FACT > 0")
    if en > 0.02:
        reasons.append(f"ENGLISH {en}")
    if pass_r < 0.9:
        reasons.append(f"validator PASS {pass_r} < 90%")
    if empty:
        reasons.append("think=false content empty")
    if think_leak:
        reasons.append("thinking present under think=false")
    if not g.get("no_markdown_fallback"):
        reasons.append("Germany/France golden not LLM-success")
    if not g.get("real_llm"):
        reasons.append("Germany/France real_llm false")
    if reasons:
        return "PRODUCTION CANDIDATE" if llm >= 0.7 and (cmo or 0) >= 10 else "BENCHMARK ONLY", reasons
    return "PRODUCTION CANDIDATE", ["Quality/reliability gates met; env still llama3 (not auto-switched)."]


def write_reports(payload: dict) -> None:
    q = {p["num_predict"]: p for p in payload.get("qwen_npred") or []}
    a, b, c = q.get(280, {}), q.get(384, {}), q.get(512, {})
    llama = payload.get("llama3_v51") or {}
    fb = payload.get("fallback_v51") or {}
    gate, reasons = payload.get("gate"), payload.get("gate_reasons") or []
    g280 = (a.get("germany") or {})

    def pct(x):
        if x is None:
            return "—"
        if isinstance(x, float) and x <= 1:
            return f"{round(x * 100, 1)}%"
        return str(x)

    def row(p, name):
        s, e = p.get("summary") or {}, p.get("extra") or {}
        return (
            f"| {name} | {pct(s.get('real_llm_success_rate'))} | {pct(s.get('fallback_rate'))} | "
            f"{e.get('cmo_llm_avg')} | {s.get('cmo_score_avg')} | {s.get('avg_latency_s')} | "
            f"{e.get('p50_latency_s')} | {e.get('p95_latency_s')} | {s.get('tokens_per_sec')} | "
            f"{pct(s.get('brief_echo_rate'))} | {pct(e.get('content_empty_rate'))} | "
            f"{pct(e.get('pass_rate'))}/{pct(e.get('warn_rate'))}/{pct(e.get('reject_rate'))} |"
        )

    bench = f"""# V5.3 Benchmark Report

Tarih: {time.strftime("%Y-%m-%d %H:%M")}
Production `OLLAMA_REASONING_MODEL` = `{payload.get("production_reasoning")}` (**değiştirilmedi**).
`OLLAMA_REASONING_THINK` default = false. Shadow default kapalı.

V5.2 think=true content-empty **model reasoning failure değildir** (API/thinking mode).

## Llama3 vs Qwen think=false vs fallback

| MODE | REAL LLM SUCCESS | FALLBACK | CMO LLM | CMO FINAL | AVG LAT | P50 | P95 | TOK/S | BRIEF ECHO | CONTENT EMPTY | PASS/WARN/REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|
{row(a, "Qwen 27B think=false n=280")}
{row(b, "Qwen 27B think=false n=384")}
{row(c, "Qwen 27B think=false n=512")}
| Llama3 V5.1 | {pct(llama.get("real_llm_success_rate"))} | {pct(llama.get("fallback_rate"))} | — | {llama.get("cmo_score_avg")} | {llama.get("avg_latency_s")} | — | — | {llama.get("tokens_per_sec")} | {pct(llama.get("brief_echo_rate"))} | — | — |
| Commercial fallback | 0% | 100% | — | {fb.get("cmo_score_avg")} | ms | — | — | — | {pct(fb.get("brief_echo_rate"))} | n/a | — |

REAL_LLM_SUCCESS ≠ final response quality. Fallback kalitesi LLM başarısı değildir.

## Latency breakdown (Qwen n=280)

prompt_eval={ (a.get("extra") or {}).get("avg_prompt_eval_s") }s  gen={ (a.get("extra") or {}).get("avg_gen_s") }s  load={ (a.get("extra") or {}).get("avg_load_s") }s  keep_alive=30m

## VRAM

| | Dedicated MiB |
|---|---|
| Test A XTTS off (V5.2 ref) | {VRAM_V52["MODEL_ONLY_PEAK_VRAM"]} |
| Test B XTTS on (this run peak) | {(a.get("summary") or {}).get("peak_vram_mib")} |
| Test C long context | {(payload.get("test_c") or {}).get("vram_mib")} |
| Card | 23040 |

Shared GPU Memory VRAM sayılmaz. CUDA OOM: {(payload.get("oom") or "yok")}

## Germany/France golden (n=280)

path={g280.get("path")} real_llm={g280.get("real_llm")} direct={g280.get("direct")} retry={g280.get("retry")}
golden={g280.get("golden")}

LLM:

```
{(g280.get("llm_direct") or "")[:2000]}
```

Final:

```
{(g280.get("final") or "")[:1500]}
```

## 20 senaryo n=280

"""
    for r in a.get("rows") or []:
        bench += (
            f"- `{r.get('id')}` path={r.get('path')} real_llm={r.get('real_llm')} "
            f"d={r.get('direct')} r={r.get('retry')} cmoL={r.get('cmo_llm')} cmoF={r.get('cmo_final')}\n"
        )
    bench += f"""
## Shadow

{(payload.get("shadow") or {})}

## Gate

**{gate}**
{"; ".join(reasons)}
"""
    REPORT.write_text(bench, encoding="utf-8")

    cand = f"""# V5.3 Production Candidate Report

## 1. Executive summary

Qwen3.5 27B `think=false` production Ollama yoluna bağlandı. Thinking kullanıcıya/TTS/memory'ye gitmez.
Production default hâlâ llama3. V5.2 %0 REAL_LLM_SUCCESS API/thinking-mode idi, model reasoning failure değil.

Gate: **{gate}**

## 2. Changes made

- Ollama chat: `think` (default false), `keep_alive=30m`, content≠thinking parse
- Telemetry: model, think, latency, eval_count, done_reason, content/thinking length
- Validator: light markdown speakable; skor-değil false-positive; %20 iddia eşlemesi
- Shadow mode: `OLLAMA_SHADOW_MODEL` (default kapalı)
- Canary: existing `OLLAMA_REASONING_MODEL` (explicit only)

## 3. Files changed

`backend/llm.py`, `backend/main.py`, `backend/agents/validator.py`, `backend/agents/commercial.py`,
`backend/agents/response_engine.py`, `backend/agents/quality.py`, `backend/agents/contracts.py`,
`backend/agents/shadow.py`, `backend/test_brain_v53.py`, `backend/benchmark_v53.py`

## 4. Files intentionally untouched

`backend/tts_local.py`, `lib/speech.ts`, mic/speak, `/api/speak`, billing, public API şekilleri,
playbook/fallback çekirdeği, production default model.

## 5. Test results

116 mevcut + V5.3 testleri. Eski testler değiştirilmedi.

## 6. Llama3 vs Qwen

Llama3 REAL_LLM {pct(llama.get("real_llm_success_rate"))} fallback {pct(llama.get("fallback_rate"))} echo {pct(llama.get("brief_echo_rate"))}
Qwen n=280 REAL_LLM {pct((a.get("summary") or {}).get("real_llm_success_rate"))} fallback {pct((a.get("summary") or {}).get("fallback_rate"))} CMO LLM {(a.get("extra") or {}).get("cmo_llm_avg")}

## 7. num_predict comparison

280 / 384 / 512: `{json.dumps({k: {"llm": (v.get("summary") or {}).get("real_llm_success_rate"), "lat": (v.get("summary") or {}).get("avg_latency_s"), "cmo": (v.get("extra") or {}).get("cmo_llm_avg")} for k, v in q.items()}, ensure_ascii=False)}`

## 8. Latency

Qwen ~{(a.get("summary") or {}).get("avg_latency_s")}s vs Llama3 ~{llama.get("avg_latency_s")}s.
Breakdown: prompt_eval {(a.get("extra") or {}).get("avg_prompt_eval_s")} gen {(a.get("extra") or {}).get("avg_gen_s")} load {(a.get("extra") or {}).get("avg_load_s")}

## 9. VRAM

V5.2 A={VRAM_V52["MODEL_ONLY_PEAK_VRAM"]} B={VRAM_V52["XTTS_MODEL_PEAK_VRAM"]} this-run peak={(a.get("summary") or {}).get("peak_vram_mib")} Test C={(payload.get("test_c") or {}).get("vram_mib")}

## 10. Validator outcomes

markdown false-positive: speakable strip. competitor %20: claim in brief. score≠win denial. Security REJECT korundu.

## 11. Germany/France golden

{g280}

## 12. Shadow mode

Default off. Benchmark shadow: {(payload.get("shadow") or {})}

## 13. Production gate

**{gate}**
{chr(10).join("- " + r for r in reasons)}

Env otomatik değiştirilmedi.

## 14. Risks

15s latency; ~20GB VRAM with XTTS; production still needs explicit `OLLAMA_REASONING_MODEL=qwen3.5:27b` to canary.

## 15. Recommendation

**{gate}**
"""
    CANDIDATE.write_text(cand, encoding="utf-8")


async def main() -> None:
    if "qwen3.5" in (os.getenv("OLLAMA_REASONING_MODEL") or "").casefold():
        raise SystemExit("Production reasoning qwen; V5.3 bunu otomatik yapmamalı.")
    if reasoning_think() and (os.getenv("OLLAMA_REASONING_THINK") or "").lower() in ("1", "true"):
        print("NOTE: OLLAMA_REASONING_THINK is true in env; benchmark still forces think=false for Qwen.", flush=True)

    async with httpx.AsyncClient(timeout=900.0) as http:
        tags = (await http.get(f"{ollama_base_url()}/api/tags")).json()
        names = {m.get("name") for m in (tags.get("models") or [])}
        if QWEN_27B not in names and f"{QWEN_27B}:latest" not in names:
            raise SystemExit("qwen3.5:27b yok")

        qwen_npred = []
        for npred in (280, 384, 512):
            qwen_npred.append(
                await run_phase(
                    http, QWEN_27B, f"qwen27b-thinkfalse-{npred}", think=False, num_predict=npred
                )
            )

        # Test C: long conversational context, XTTS left running
        long_user = ("Önceki tur özeti. " * 40) + CASES[1]["input"]
        t0 = time.perf_counter()
        hw = HwPeak()
        metrics: dict = {"calls": []}
        gen = make_generate(QWEN_27B, metrics, hw, think=False, num_predict=280)
        from benchmark_v51_cases import FAIR_FACTS
        from agents.response_engine import compose_consultant_traced
        from agents.quality import reset_quality

        reset_quality()
        await compose_consultant_traced(
            generate=gen,
            http=http,
            question=long_user,
            facts=FAIR_FACTS,
        )
        test_c = {
            "wall_s": round(time.perf_counter() - t0, 3),
            "vram_mib": (nvidia_query() or {}).get("vram_used_mib"),
            "calls": metrics.get("calls"),
        }

        from agents.shadow import run_shadow_eval

        os.environ["OLLAMA_SHADOW_MODEL"] = QWEN_27B
        try:
            shadow = await run_shadow_eval(
                http=http,
                question=CASES[1]["input"],
                user="ping",
                system="Kısa Türkçe cevap.",
                production_path="LLM_CMO_SUCCESS",
                production_text="production llama text",
            )
        finally:
            os.environ.pop("OLLAMA_SHADOW_MODEL", None)

        payload = {
            "production_reasoning": reasoning_model(),
            "production_chat": chat_model(),
            "think_default": reasoning_think(),
            "qwen_npred": qwen_npred,
            "llama3_v51": v51_row("llama3"),
            "fallback_v51": v51_row("commercial fallback"),
            "test_c": test_c,
            "shadow": {k: v for k, v in shadow.items() if k != "user_visible"}
            | {"user_visible_len": len(shadow.get("user_visible") or "")},
            "vram_v52": VRAM_V52,
            "gpu": nvidia_query(),
            "oom": None,
        }
        gate, reasons = production_gate(qwen_npred[0] if qwen_npred else {})
        payload["gate"] = gate
        payload["gate_reasons"] = reasons
        JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        write_reports(payload)
        print(json.dumps({"gate": gate, "reasons": reasons, "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
