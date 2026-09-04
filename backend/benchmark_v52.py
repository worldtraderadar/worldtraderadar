"""V5.2 Qwen3.5 27B hardware benchmark. Production OLLAMA_REASONING_MODEL değiştirmez."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from benchmark_v51 import (
    CASES,
    HwPeak,
    nvidia_query,
    ollama_tags,
    resolve_installed,
    run_case,
    run_fallback_only,
    shared_gpu_sample,
    summarize,
)
from llm import chat_model, ollama_base_url, reasoning_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
REPORT = ROOT / "V5.2_QWEN27B_BENCHMARK_REPORT.md"
JSON_OUT = BACKEND / "benchmark_v52_results.json"
V51_JSON = BACKEND / "benchmark_v51_results.json"
ALIASES = ("qwen3.5:27b", "qwen3.5:27b-q4_k_m", "qwen3.5:27b-q4", "qwen3.5:27b-instruct")


def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"ERR:{exc}"


def ram_gb() -> dict:
    ps = (
        "$cs=Get-CimInstance Win32_ComputerSystem; $os=Get-CimInstance Win32_OperatingSystem; "
        "$inv=[cultureinfo]::InvariantCulture; "
        "Write-Output (([math]::Round($cs.TotalPhysicalMemory/1GB,1).ToString($inv)) + '|' + "
        "([math]::Round($os.FreePhysicalMemory/1MB,1).ToString($inv)))"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", ps]).strip()
    try:
        total, free = raw.split("|")
        return {
            "ram_total_gb": float(total),
            "ram_free_gb": float(free),
            "ram_used_gb": round(float(total) - float(free), 1),
        }
    except Exception:
        return {"ram_total_gb": None, "ram_free_gb": None, "raw": raw}


def cpu_pct() -> float | None:
    ps = (
        "try { [math]::Round((Get-Counter '\\Processor(_Total)\\% Processor Time')"
        ".CounterSamples[0].CookedValue,1) } catch { 'ERR' }"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", ps]).strip()
    try:
        return float(raw.replace(",", "."))
    except Exception:
        return None


def uvicorn_pids() -> list[int]:
    ps = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'uvicorn' -and $_.CommandLine -match 'main:app' } "
        "| ForEach-Object { $_.ProcessId }"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", ps])
    pids = []
    for line in raw.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def stop_xtts_server() -> dict:
    pids = uvicorn_pids()
    killed = []
    for pid in pids:
        _run(["taskkill", "/PID", str(pid), "/T", "/F"])
        killed.append(pid)
    time.sleep(4)
    return {"killed": killed, "gpu_after": nvidia_query(), "ram_after": ram_gb()}


def start_xtts_server() -> dict:
    log = BACKEND / "v52_uvicorn.log"
    proc = subprocess.Popen(
        [
            str(BACKEND / "venv" / "Scripts" / "python.exe"),
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(BACKEND),
        stdout=log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    tts = None
    deadline = time.time() + 180
    last = ""
    while time.time() < deadline:
        try:
            data = httpx.get("http://127.0.0.1:8000/health", timeout=5.0).json()
            tts = data.get("tts") or {}
            last = str(tts)
            if tts.get("ready") and str(tts.get("engine") or "").lower() == "xtts":
                break
        except Exception as exc:
            last = str(exc)
        time.sleep(3)
    time.sleep(2)
    return {
        "pid": proc.pid,
        "tts": tts,
        "last": last,
        "gpu": nvidia_query(),
        "ram": ram_gb(),
    }


def make_generate(model: str, metrics: dict, hw: HwPeak):
    async def generate(http, prompt, system_prompt=None, history=None, polish=False):
        hw.mark()
        t0 = time.perf_counter()
        response = await http.post(
            f"{ollama_base_url()}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0.3, "num_predict": 280},
            },
            timeout=600.0,
        )
        response.raise_for_status()
        data = response.json()
        hw.mark()
        elapsed = time.perf_counter() - t0
        eval_count = data.get("eval_count") or 0
        eval_ns = data.get("eval_duration") or 0
        load_ns = data.get("load_duration") or 0
        total_ns = data.get("total_duration") or 0
        prompt_ns = data.get("prompt_eval_duration") or 0
        tps = (eval_count / (eval_ns / 1e9)) if eval_ns else None
        metrics.setdefault("calls", []).append(
            {
                "latency_s": round(elapsed, 3),
                "eval_count": eval_count,
                "tokens_per_sec": round(tps, 2) if tps else None,
                "load_s": round(load_ns / 1e9, 3) if load_ns else None,
                "prompt_eval_s": round(prompt_ns / 1e9, 3) if prompt_ns else None,
                "total_s": round(total_ns / 1e9, 3) if total_ns else None,
                "vram_after_mib": (nvidia_query() or {}).get("vram_used_mib"),
                "cpu_pct": cpu_pct(),
                "ram": ram_gb(),
            }
        )
        return (data.get("message") or {}).get("content") or data.get("response") or ""

    return generate


def offload_label(hw: HwPeak, metrics: dict, idle_vram: float | None) -> str:
    calls = metrics.get("calls") or []
    tps = [c.get("tokens_per_sec") or 0 for c in calls]
    avg_tps = sum(tps) / len(tps) if tps else 0
    peak = hw.vram_peak or 0
    idle = idle_vram or hw.vram_before or 0
    delta = peak - idle
    if peak >= 21000:
        return "TIGHT_FIT_OR_SPILL"
    if delta >= 8000 and avg_tps >= 15:
        return "NO"
    if delta < 2500 and avg_tps and avg_tps < 12:
        return "LIKELY_YES"
    if avg_tps and avg_tps < 8:
        return "LIKELY_YES"
    return "UNKNOWN"


def v51_row(label_prefix: str) -> dict | None:
    if not V51_JSON.exists():
        return None
    data = json.loads(V51_JSON.read_text(encoding="utf-8"))
    for item in data.get("summaries") or []:
        if str(item.get("label") or "").startswith(label_prefix):
            return item
    return None


def germany_from(rows: list[dict]) -> dict | None:
    return next((r for r in rows if r.get("id") == "germany_france"), None)


async def load_probe(http: httpx.AsyncClient, model: str) -> dict:
    idle = nvidia_query()
    ram0 = ram_gb()
    cpu0 = cpu_pct()
    t0 = time.perf_counter()
    response = await http.post(
        f"{ollama_base_url()}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": "ping"}],
            "options": {"temperature": 0, "num_predict": 8},
        },
        timeout=600.0,
    )
    load_wall = time.perf_counter() - t0
    data = response.json()
    after = nvidia_query()
    return {
        "idle_vram_mib": idle.get("vram_used_mib"),
        "idle_vram_free_mib": idle.get("vram_free_mib"),
        "after_load_vram_mib": after.get("vram_used_mib"),
        "model_only_vram_delta_mib": round(
            (after.get("vram_used_mib") or 0) - (idle.get("vram_used_mib") or 0), 1
        ),
        "load_wall_s": round(load_wall, 3),
        "load_s": round((data.get("load_duration") or 0) / 1e9, 3),
        "eval_count": data.get("eval_count"),
        "ram_before": ram0,
        "ram_after": ram_gb(),
        "cpu_before": cpu0,
        "cpu_after": cpu_pct(),
        "shared_after": shared_gpu_sample(),
        "snippet": ((data.get("message") or {}).get("content") or "")[:200],
    }


def write_report(payload: dict) -> None:
    a = payload.get("test_a") or {}
    b = payload.get("test_b") or {}
    sa = a.get("summary") or {}
    sb = b.get("summary") or {}
    llama = payload.get("llama3_v51") or {}
    fb = payload.get("fallback_v51") or {}
    inst = payload.get("installed_model") or {}
    rec = payload.get("recommendation") or "NO CLEAR WINNER"
    ga = germany_from(a.get("rows") or []) or {}
    gb = germany_from(b.get("rows") or []) or ga
    body = f"""# V5.2 Qwen3.5 27B Gerçek Donanım Benchmarkı

Tarih: {time.strftime("%Y-%m-%d %H:%M")}
Production `OLLAMA_REASONING_MODEL` = `{payload.get("production_reasoning")}` (**değiştirilmedi**).
35B bu turda kurulmadı.

## Model kurulumu

| Alan | Değer |
|---|---|
| İstenen tag | `qwen3.5:27b` |
| Kurulu tag | `{inst.get("name")}` |
| Durum | `{payload.get("install_status")}` |
| Quantization | `{inst.get("quant")}` |
| Parameter | `{inst.get("param")}` |
| Dosya boyutu | `{inst.get("size_gb")} GB` |

## Donanım

Quadro RTX 6000 dedicated **23040 MiB (~22.5 GB)**. Shared GPU Memory VRAM değildir.

| Ölçüm | Test A (XTTS kapalı) | Test B (XTTS açık) |
|---|---|---|
| Idle dedicated VRAM | {a.get("idle_vram_mib")} MiB | {b.get("xtts_baseline_vram_mib")} MiB |
| Load sonrası VRAM | {a.get("load", {}).get("after_load_vram_mib")} MiB | {b.get("load", {}).get("after_load_vram_mib")} MiB |
| Inference peak VRAM | {sa.get("peak_vram_mib")} MiB | {sb.get("peak_vram_mib")} MiB |
| MODEL_ONLY_VRAM (delta vs A idle) | {a.get("load", {}).get("model_only_vram_delta_mib")} MiB | — |
| Shared GPU Memory | `{str((a.get("shared") or {}).get("shared_counter") or "")[:160]}` | `{str((b.get("shared") or {}).get("shared_counter") or "")[:160]}` |
| System RAM | {a.get("ram")} | {b.get("ram")} |
| GPU util peak | {sa.get("gpu_util_peak")} | {sb.get("gpu_util_peak")} |
| CPU (load probe) | {a.get("load", {}).get("cpu_after")} | {b.get("load", {}).get("cpu_after")} |
| Load wall | {a.get("load", {}).get("load_wall_s")} s | {b.get("load", {}).get("load_wall_s")} s |
| Avg latency | {sa.get("avg_latency_s")} s | {sb.get("avg_latency_s")} s |
| tok/s | {sa.get("tokens_per_sec")} | {sb.get("tokens_per_sec")} |
| Offload | {a.get("offload")} | {b.get("offload")} |
| Hata | {a.get("error") or "yok"} | {b.get("error") or "yok"} |

## Llama3 vs Qwen3.5 27B vs fallback

Llama3 ve fallback sayıları V5.1 aynı 20 senaryodan (yeniden koşulmadı). Qwen 27B bu turda koşuldu.

| MODEL | CMO SCORE | REAL LLM SUCCESS | FALLBACK RATE | BRIEF ECHO | ENGLISH | UNSUPPORTED FACT | FALSE CERTAINTY | MEMORY DISHONESTY | AVG LATENCY | TOKENS/SEC | MODEL-ONLY PEAK VRAM | XTTS+MODEL PEAK VRAM | PEAK SYSTEM RAM | OFFLOAD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama3 | {llama.get("cmo_score_avg")} | {llama.get("real_llm_success_rate")} | {llama.get("fallback_rate")} | {llama.get("brief_echo_rate")} | {llama.get("english_rate")} | {llama.get("unsupported_fact_rate")} | {llama.get("false_certainty_rate")} | — | {llama.get("avg_latency_s")} | {llama.get("tokens_per_sec")} | — | {llama.get("peak_vram_mib")} MiB (XTTS+llama, V5.1) | {llama.get("peak_ram_gb")} | {llama.get("offload")} |
| Qwen3.5 27B | {sa.get("cmo_score_avg")} | {sa.get("real_llm_success_rate")} | {sa.get("fallback_rate")} | {sa.get("brief_echo_rate")} | {sa.get("english_rate")} | {sa.get("unsupported_fact_rate")} | {sa.get("false_certainty_rate")} | {a.get("memory_dishonesty_rate")} | {sa.get("avg_latency_s")} | {sa.get("tokens_per_sec")} | {sa.get("peak_vram_mib")} MiB | {sb.get("peak_vram_mib")} MiB | {sa.get("peak_ram_gb")} | A:{a.get("offload")} / B:{b.get("offload")} |
| Commercial fallback | {fb.get("cmo_score_avg")} | 0.0 | 1.0 | {fb.get("brief_echo_rate")} | {fb.get("english_rate")} | {fb.get("unsupported_fact_rate")} | {fb.get("false_certainty_rate")} | — | <1 ms | — | — | — | — | n/a |

REAL_LLM_SUCCESS = (LLM_DIRECT + LLM_RETRY) / 20. Fallback kalitesi LLM başarısı değildir.

Test A Qwen memory_dishonesty_rate = {a.get("memory_dishonesty_rate")}; Test B = {b.get("memory_dishonesty_rate")}.

## Almanya vs Fransa (Qwen 27B)

Soru: *Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?*

Test B path (XTTS açık, üretim koşulu): `{gb.get("public_path")}` internal `{gb.get("path")}`
Validator direct `{gb.get("direct_status")}` retry `{gb.get("retry_status")}`
CMO final /14: `{ (gb.get("score_final") or {}).get("total") }`
CMO LLM direct /14: `{ (gb.get("score_llm_direct") or {}).get("total") }`
Brief echo: `{gb.get("brief_echo")}` False certainty: `{gb.get("false_certainty")}` Memory dishonesty: `{gb.get("memory_dishonesty")}` English: `{gb.get("english_leakage")}`

**LLM direct raw:**

```
{(gb.get("llm_direct") or "(yok)")[:2500]}
```

**LLM retry raw:**

```
{(gb.get("llm_retry") or "(yok)")[:2500]}
```

**Final:**

```
{(gb.get("final") or "")[:2500]}
```

Test A (XTTS kapalı) path: `{ga.get("public_path")}` brief_echo={ga.get("brief_echo")} real_llm={ga.get("real_llm")}

## 20 senaryo (Test A, model-only)

"""
    for row in a.get("rows") or []:
        body += (
            f"- `{row.get('id')}` path={row.get('public_path')} "
            f"real_llm={row.get('real_llm')} echo={row.get('brief_echo')} "
            f"cmo={(row.get('score_final') or {}).get('total')} "
            f"direct={row.get('direct_status')} retry={row.get('retry_status')}\n"
        )
    body += f"""

## Production suitability

{payload.get("suitability_notes")}

## RECOMMENDATION

**{rec}**

35B için karar verilmedi; kurulmadı.

## Testler

V5.1 106 test + `test_brain_v52.py`. Production env değiştirilmedi.
Ses / billing / public API kodu değiştirilmedi. XTTS yalnızca süreç olarak durdurulup yeniden başlatıldı.
"""
    REPORT.write_text(body, encoding="utf-8")


def decide(payload: dict) -> tuple[str, str]:
    a = payload.get("test_a") or {}
    b = payload.get("test_b") or {}
    sa = a.get("summary") or {}
    sb = b.get("summary") or {}
    llama = payload.get("llama3_v51") or {}
    q_llm = sa.get("real_llm_success_rate") or 0
    l_llm = llama.get("real_llm_success_rate") or 0
    q_echo = sa.get("brief_echo_rate") or 0
    l_echo = llama.get("brief_echo_rate") or 1
    q_fb = sa.get("fallback_rate") or 1
    lat_b = sb.get("avg_latency_s")
    err_b = b.get("error")
    peak_b = sb.get("peak_vram_mib") or 0
    tps_b = sb.get("tokens_per_sec") or 0
    notes = []
    quality = q_llm >= l_llm + 0.25 and q_echo + 0.2 <= l_echo and q_fb + 0.15 <= (llama.get("fallback_rate") or 1)
    coexist = not err_b and lat_b is not None and lat_b < 25 and tps_b >= 12 and peak_b < 22500
    if err_b:
        notes.append(f"Test B hata: {err_b}")
        rec = "C) NO CLEAR WINNER" if not quality else "B) QWEN3.5 27B BENCHMARK WINNER BUT NOT PRODUCTION READY"
    elif quality and coexist:
        rec = "A) QWEN3.5 27B PRODUCTION ADAYI"
        notes.append("Kalite Llama3'ten belirgin; XTTS ile latency/VRAM kabul edilebilir. Env yine de otomatik değiştirilmedi.")
    elif quality:
        rec = "B) QWEN3.5 27B BENCHMARK WINNER BUT NOT PRODUCTION READY"
        notes.append("Reasoning daha iyi görünüyor ancak XTTS+GPU koşulu production için zayıf (latency/VRAM/offload).")
    else:
        rec = "C) NO CLEAR WINNER"
        notes.append("Fark Llama3'e göre production kararı için yeterince net değil veya kalite hâlâ fallback'e bağlı.")
    notes.append(
        f"Qwen REAL_LLM_SUCCESS={q_llm} vs Llama3 {l_llm}; fallback {q_fb} vs {llama.get('fallback_rate')}; "
        f"echo {q_echo} vs {l_echo}."
    )
    return rec, " ".join(notes)


async def run_phase(http, model: str, label: str, idle_vram: float | None) -> dict:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    gen = make_generate(model, metrics, hw)
    rows = []
    error = None
    try:
        for case in CASES:
            print(f"[{label}] {case['id']} ...", flush=True)
            rows.append(await run_case(http, gen, case))
    except Exception as exc:
        error = repr(exc)
        print(f"[{label}] ERROR {error}", flush=True)
    summary = summarize(label, rows, metrics, hw, "TESTED" if rows else "FAILED")
    mem_rate = round(sum(1 for r in rows if r.get("memory_dishonesty")) / (len(rows) or 1), 3)
    return {
        "summary": summary,
        "rows": rows,
        "metrics": metrics,
        "error": error,
        "offload": offload_label(hw, metrics, idle_vram),
        "memory_dishonesty_rate": mem_rate,
        "shared": shared_gpu_sample(),
        "ram": ram_gb(),
        "gpu_end": nvidia_query(),
    }


async def main() -> None:
    if "qwen3.5" in (os.getenv("OLLAMA_REASONING_MODEL") or "").casefold():
        raise SystemExit("Production OLLAMA_REASONING_MODEL qwen; V5.2 bunu değiştirmemeli / kullanmamalı.")
    async with httpx.AsyncClient(timeout=600.0) as http:
        tags = await ollama_tags(http)
        hit = resolve_installed(tags, ALIASES)
        inst = next((t for t in tags if t.get("name") == hit), None) if hit else None
        if not hit:
            payload = {
                "install_status": "MODEL GEREKLİ / PULL FAILED / NOT TESTED",
                "installed_model": {},
                "production_reasoning": reasoning_model(),
                "tags": tags,
            }
            JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            write_report({**payload, "recommendation": "C) NO CLEAR WINNER", "suitability_notes": "Model kurulamadı."})
            raise SystemExit("qwen3.5:27b kurulu değil; benchmark üretilmedi.")

        print("Stopping XTTS/uvicorn for Test A...", flush=True)
        stop_info = stop_xtts_server()
        time.sleep(3)
        idle_a = nvidia_query()
        print(f"Test A idle VRAM={idle_a.get('vram_used_mib')}", flush=True)
        _run(["ollama", "stop", "llama3"])
        _run(["ollama", "stop", "qwen2.5:latest"])
        load_a = await load_probe(http, hit)
        print(f"Test A load delta={load_a.get('model_only_vram_delta_mib')} MiB wall={load_a.get('load_wall_s')}s", flush=True)
        phase_a = await run_phase(http, hit, "qwen3.5:27b-A", idle_a.get("vram_used_mib"))
        phase_a["idle_vram_mib"] = idle_a.get("vram_used_mib")
        phase_a["load"] = load_a
        phase_a["stop_info"] = stop_info

        print("Starting XTTS for Test B...", flush=True)
        start_info = start_xtts_server()
        time.sleep(2)
        xtts_base = nvidia_query()
        print(f"Test B XTTS baseline VRAM={xtts_base.get('vram_used_mib')} tts={start_info.get('tts')}", flush=True)
        load_b = await load_probe(http, hit)
        phase_b = await run_phase(http, hit, "qwen3.5:27b-B", xtts_base.get("vram_used_mib"))
        phase_b["xtts_baseline_vram_mib"] = xtts_base.get("vram_used_mib")
        phase_b["load"] = load_b
        phase_b["start_info"] = start_info

        payload = {
            "install_status": f"MODEL KURULU ({hit})",
            "installed_model": inst or {"name": hit},
            "production_reasoning": reasoning_model(),
            "production_chat": chat_model(),
            "test_a": phase_a,
            "test_b": phase_b,
            "llama3_v51": v51_row("llama3"),
            "fallback_v51": v51_row("commercial fallback"),
            "tags": tags,
        }
        rec, notes = decide(payload)
        payload["recommendation"] = rec
        payload["suitability_notes"] = notes
        JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        write_report(payload)
        print(json.dumps({"report": str(REPORT), "rec": rec, "install": hit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
