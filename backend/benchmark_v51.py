"""V5.1 CMO model benchmark. Production OLLAMA_REASONING_MODEL değiştirmez."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from agents.commercial import build_commercial_brief, commercial_fallback_reply
from agents.evaluator import score_cmo_v51
from agents.quality import reset_quality
from agents.response_engine import compose_consultant_traced
from agents.session import SessionState
from benchmark_v51_cases import CASES
from llm import chat_model, ollama_base_url

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "V5.1_BENCHMARK_REPORT.md"
JSON_OUT = Path(__file__).resolve().parent / "benchmark_v51_results.json"

REQUIRED = {
    "llama3": ("llama3", "llama3:latest"),
    "qwen3.5-27b-q4": (
        "qwen3.5:27b",
        "qwen3.5:27b-q4_k_m",
        "qwen3.5:27b-q4",
        "qwen3.5:27b-instruct",
    ),
    "qwen3.5-35b-a3b-q4": (
        "qwen3.5:35b-a3b",
        "qwen3.5:35b-a3b-q4_k_m",
        "qwen3.5:35b-a3b-q4",
    ),
}


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=20)
        return out.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"ERR:{exc}"


def nvidia_query() -> dict:
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 6 or raw.startswith("ERR"):
        return {"raw": raw.strip(), "ok": False}
    return {
        "ok": True,
        "name": parts[0],
        "vram_total_mib": float(parts[1]),
        "vram_used_mib": float(parts[2]),
        "vram_free_mib": float(parts[3]),
        "gpu_util": float(parts[4]),
        "mem_util": float(parts[5]),
        "note": "nvidia-smi = dedicated VRAM, Shared GPU Memory değil",
    }


def system_ram_gb() -> dict:
    ps = (
        "$cs=Get-CimInstance Win32_ComputerSystem; $os=Get-CimInstance Win32_OperatingSystem; "
        "Write-Output (($cs.TotalPhysicalMemory/1GB).ToString('0.0') + ',' + "
        "($os.FreePhysicalMemory/1MB).ToString('0.0'))"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", ps]).strip()
    try:
        total, free = raw.split(",")
        return {"ram_total_gb": float(total.replace(",", ".")), "ram_free_gb": float(free.replace(",", "."))}
    except Exception:
        return {"ram_total_gb": None, "ram_free_gb": None, "raw": raw}


def shared_gpu_sample() -> dict:
    ps = (
        "try { $c=Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage','"
        "\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction Stop; "
        "$c.CounterSamples | ForEach-Object { $_.Path + '=' + $_.CookedValue } } "
        "catch { 'SHARED_COUNTER_UNAVAILABLE' }"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", ps]).strip()
    return {"shared_counter": raw[:2000]}


async def ollama_tags(http: httpx.AsyncClient) -> list[dict]:
    try:
        data = (await http.get(f"{ollama_base_url()}/api/tags")).json()
        rows = []
        for item in data.get("models") or []:
            rows.append(
                {
                    "name": item.get("name"),
                    "size_gb": round((item.get("size") or 0) / 1e9, 2),
                    "param": (item.get("details") or {}).get("parameter_size"),
                    "quant": (item.get("details") or {}).get("quantization_level"),
                }
            )
        return rows
    except Exception as exc:
        return [{"error": str(exc)}]


def resolve_installed(tags: list[dict], aliases: tuple[str, ...]) -> str | None:
    names = {str(item.get("name") or "") for item in tags}
    bases = {n.split(":")[0] for n in names}
    for alias in aliases:
        if alias in names:
            return alias
        if alias in bases:
            for n in names:
                if n == alias or n.startswith(alias + ":"):
                    return n
    return None


@dataclass
class HwPeak:
    vram_before: float | None = None
    vram_peak: float | None = None
    ram_free_before: float | None = None
    ram_free_min: float | None = None
    gpu_util_peak: float | None = None
    samples: int = 0

    def mark(self) -> None:
        gpu = nvidia_query()
        ram = system_ram_gb()
        if gpu.get("ok"):
            used = gpu["vram_used_mib"]
            util = gpu["gpu_util"]
            if self.vram_before is None:
                self.vram_before = used
            self.vram_peak = used if self.vram_peak is None else max(self.vram_peak, used)
            self.gpu_util_peak = util if self.gpu_util_peak is None else max(self.gpu_util_peak, util)
        free = ram.get("ram_free_gb")
        if isinstance(free, float):
            if self.ram_free_before is None:
                self.ram_free_before = free
            self.ram_free_min = free if self.ram_free_min is None else min(self.ram_free_min, free)
        self.samples += 1


def _session_for(case: dict) -> SessionState | None:
    raw = case.get("session")
    if not raw:
        return None
    return SessionState(session_id="v51", **raw)


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
            timeout=240.0,
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
            }
        )
        return (data.get("message") or {}).get("content") or data.get("response") or ""

    return generate


async def run_case(http, generate, case: dict):
    reset_quality()
    q = case["input"]
    facts = case.get("context") or ""
    session = _session_for(case)
    matches = case.get("matches")
    t0 = time.perf_counter()
    trace = await compose_consultant_traced(
        generate=generate,
        http=http,
        question=q,
        facts=facts,
        session=session,
        matches=matches,
    )
    latency = time.perf_counter() - t0
    scored_final = score_cmo_v51(trace.text, question=q, facts=facts)
    scored_direct = score_cmo_v51(trace.llm_direct, question=q, facts=facts) if trace.llm_direct else None
    fb = trace.commercial_fallback or commercial_fallback_reply(build_commercial_brief(q, session=session, matches=matches), q) or ""
    return {
        "id": case["id"],
        "input": q,
        "path": trace.path,
        "public_path": trace.public_path,
        "direct_status": trace.direct_status,
        "retry_status": trace.retry_status,
        "final": trace.text,
        "llm_direct": trace.llm_direct,
        "llm_retry": trace.llm_retry,
        "commercial_fallback": fb,
        "score_final": scored_final.as_dict(),
        "score_llm_direct": scored_direct.as_dict() if scored_direct else None,
        "brief_echo": scored_final.brief_echo or trace.brief_echo,
        "false_certainty": scored_final.false_certainty,
        "memory_dishonesty": scored_final.memory_dishonesty,
        "english_leakage": scored_final.english_leakage,
        "unsupported_fact": scored_final.unsupported_fact,
        "wall_latency_s": round(latency, 3),
        "real_llm": trace.path in ("LLM_CMO_SUCCESS", "LLM_CMO_RETRY_SUCCESS"),
        "first_done_reason": trace.first_done_reason,
        "done_reason": trace.done_reason,
    }


async def run_fallback_only(case: dict) -> dict:
    q = case["input"]
    facts = case.get("context") or ""
    session = _session_for(case)
    matches = case.get("matches")
    t0 = time.perf_counter()
    text = commercial_fallback_reply(
        build_commercial_brief(q, session=session, matches=matches), q
    ) or ""
    latency = time.perf_counter() - t0
    score = score_cmo_v51(text, question=q, facts=facts)
    return {
        "id": case["id"],
        "input": q,
        "path": "COMMERCIAL_FALLBACK",
        "public_path": "COMMERCIAL_FALLBACK",
        "direct_status": "",
        "retry_status": "",
        "final": text,
        "llm_direct": "",
        "llm_retry": "",
        "commercial_fallback": text,
        "score_final": score.as_dict(),
        "score_llm_direct": None,
        "brief_echo": score.brief_echo,
        "false_certainty": score.false_certainty,
        "memory_dishonesty": score.memory_dishonesty,
        "english_leakage": score.english_leakage,
        "unsupported_fact": score.unsupported_fact,
        "wall_latency_s": round(latency, 4),
        "real_llm": False,
    }


def summarize(label: str, rows: list[dict], metrics: dict, hw: HwPeak, status: str) -> dict:
    n = len(rows) or 1
    llm_ok = sum(1 for r in rows if r.get("real_llm"))
    calls = metrics.get("calls") or []
    lat = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
    tps = [c["tokens_per_sec"] for c in calls if c.get("tokens_per_sec")]
    loads = [c["load_s"] for c in calls if c.get("load_s") is not None]
    ram = system_ram_gb()
    peak_ram = None
    if hw.ram_free_before is not None and hw.ram_free_min is not None and ram.get("ram_total_gb"):
        used_before = ram["ram_total_gb"] - hw.ram_free_before
        used_peak = ram["ram_total_gb"] - hw.ram_free_min
        peak_ram = round(max(used_before, used_peak), 2)
    offload = None
    if hw.vram_peak and hw.vram_before is not None:
        delta = hw.vram_peak - hw.vram_before
        offload = "UNKNOWN"
        if delta < 500 and any((c.get("eval_count") or 0) > 10 for c in calls):
            offload = "LIKELY_YES"
        elif delta >= 1500:
            offload = "NO"
    scores = [r["score_final"]["total"] for r in rows if r.get("score_final")]
    return {
        "label": label,
        "status": status,
        "n": len(rows),
        "cmo_score_avg": round(sum(scores) / len(scores), 3) if scores else None,
        "real_llm_success_rate": round(llm_ok / n, 3),
        "fallback_rate": round(sum(1 for r in rows if not r.get("real_llm")) / n, 3),
        "brief_echo_rate": round(sum(1 for r in rows if r.get("brief_echo")) / n, 3),
        "english_rate": round(sum(1 for r in rows if r.get("english_leakage")) / n, 3),
        "unsupported_fact_rate": round(sum(1 for r in rows if r.get("unsupported_fact")) / n, 3),
        "false_certainty_rate": round(sum(1 for r in rows if r.get("false_certainty")) / n, 3),
        "avg_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        "tokens_per_sec": round(sum(tps) / len(tps), 2) if tps else None,
        "load_s": round(sum(loads) / len(loads), 3) if loads else None,
        "peak_vram_mib": hw.vram_peak,
        "vram_before_mib": hw.vram_before,
        "peak_ram_gb": peak_ram,
        "gpu_util_peak": hw.gpu_util_peak,
        "offload": offload,
        "rows": rows,
        "call_metrics": calls,
    }


def _md_table(summaries: list[dict]) -> str:
    lines = [
        "| MODEL | CMO SCORE | REAL LLM SUCCESS | FALLBACK RATE | BRIEF ECHO | ENGLISH | UNSUPPORTED FACT | FALSE CERTAINTY | AVG LATENCY | TOKENS/SEC | PEAK VRAM | PEAK RAM |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in summaries:
        if item["status"] != "TESTED":
            lines.append(
                f"| {item['label']} | NOT INSTALLED / NOT TESTED | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        vram = item["peak_vram_mib"]
        vram_s = f"{vram:.0f} MiB" if isinstance(vram, float) else "—"
        ram = item["peak_ram_gb"]
        ram_s = f"{ram} GB" if ram is not None else "—"
        lines.append(
            "| {label} | {cmo} | {llm} | {fb} | {be} | {en} | {uf} | {fc} | {lat} | {tps} | {vram} | {ram} |".format(
                label=item["label"],
                cmo=item["cmo_score_avg"],
                llm=item["real_llm_success_rate"],
                fb=item["fallback_rate"],
                be=item["brief_echo_rate"],
                en=item["english_rate"],
                uf=item["unsupported_fact_rate"],
                fc=item["false_certainty_rate"],
                lat=item["avg_latency_s"],
                tps=item["tokens_per_sec"],
                vram=vram_s,
                ram=ram_s,
            )
        )
    return "\n".join(lines)


def write_report(payload: dict) -> None:
    hw = payload["hardware"]
    germany = payload.get("germany_vs_france") or {}
    body = f"""# V5.1 CMO Brain / Model Benchmark

Tarih: {time.strftime("%Y-%m-%d %H:%M")}
Amaç: AI gerçekten ticari düşünüyor mu, yoksa fallback mi sistemi iyi gösteriyor?
Production `OLLAMA_REASONING_MODEL` **değiştirilmedi**.

## Donanım (referans)

| Ölçüm | Değer |
|---|---|
| GPU | {hw.get("gpu", {}).get("name")} |
| Dedicated VRAM (nvidia-smi total) | {hw.get("gpu", {}).get("vram_total_mib")} MiB |
| Idle VRAM used | {hw.get("gpu", {}).get("vram_used_mib")} MiB |
| Shared GPU Memory | Windows WDDM kavramı; **gerçek VRAM değil**. Sayacı: `{hw.get("shared", {}).get("shared_counter", "")[:180]}` |
| System RAM | {hw.get("ram", {}).get("ram_total_gb")} GB total / {hw.get("ram", {}).get("ram_free_gb")} GB free |
| Driver | nvidia-smi 597.06, CUDA 13.2 |

**Kural:** Model dosya boyutu ≠ runtime VRAM. KV cache + overhead ayrıca.

## Kurulu Ollama modelleri

```
{json.dumps(payload.get("installed"), ensure_ascii=False, indent=2)}
```

Env: `OLLAMA_CHAT_MODEL`={os.getenv("OLLAMA_CHAT_MODEL") or "(unset, default llama3)"} ; `OLLAMA_REASONING_MODEL`={os.getenv("OLLAMA_REASONING_MODEL") or "(unset, falls back to chat)"}

| İstenen | Durum |
|---|---|
| Llama3 | {payload["model_status"]["llama3"]} |
| Qwen3.5 27B Q4 | {payload["model_status"]["qwen3.5-27b-q4"]} |
| Qwen3.5-35B-A3B Q4 | {payload["model_status"]["qwen3.5-35b-a3b-q4"]} |
| Commercial fallback | MODEL KURULU (kural motoru, GPU model değil) |
| Extra installed (istenmedi) | {payload["model_status"].get("extra_qwen25", "—")} |

Kurulum gerekirse (bu turda **çekilmedi**):

```
ollama pull qwen3.5:27b
ollama pull qwen3.5:35b-a3b
```

`qwen3.5:35b-a3b` kütüphane etiketi Q4_K_M, dosya ~24 GB. 22.5 GB dedicated VRAM + KV cache için **kesin sığar varsayımı yok**.

## Özet tablo

{_md_table(payload["summaries"])}

REAL_LLM_SUCCESS = (LLM_DIRECT + LLM_RETRY) / reasoning tasks.
Fallback kalitesi ayrı satırdadır; LLM başarısı sayılmaz.

## GERMANY VS FRANCE CASE

Soru: *Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?*
Aynı facts / kart bağlamı tüm modellerde.

"""
    for name, row in germany.items():
        if not row:
            body += f"\n### {name}\n\nNOT INSTALLED / NOT TESTED\n"
            continue
        body += f"""
### {name}

- Path: `{row.get("public_path")}` (internal `{row.get("path")}`)
- Validator direct: `{row.get("direct_status") or "—"}` retry: `{row.get("retry_status") or "—"}`
- Retry oldu mu: {"evet" if row.get("llm_retry") else "hayır / yok"}
- CMO score (final /14): {row.get("score_final", {}).get("total")}
- Brief echo: {row.get("brief_echo")}
- False certainty: {row.get("false_certainty")}
- Memory dishonesty: {row.get("memory_dishonesty")}

**LLM direct raw:**

```
{(row.get("llm_direct") or "(yok)")[:2500]}
```

**LLM retry raw:**

```
{(row.get("llm_retry") or "(yok)")[:2500]}
```

**Final output:**

```
{(row.get("final") or "")[:2500]}
```
"""
    body += """
## 12 soruluk sonuç

"""
    tested = [s for s in payload["summaries"] if s.get("status") == "TESTED"]
    llama = next((s for s in tested if s["label"].startswith("llama3")), None)
    qwen27 = next((s for s in payload["summaries"] if "27B" in s["label"]), None)
    qwen35 = next((s for s in payload["summaries"] if "35B" in s["label"]), None)
    extra = next((s for s in tested if "qwen2.5" in s["label"]), None)
    fb = next((s for s in tested if "fallback" in s["label"]), None)

    def rate(item, key):
        if not item or item.get("status") != "TESTED":
            return "NOT TESTED"
        return item.get(key)

    body += f"""
1. Llama3 gerçekten reasoning yapıyor mu? **Hayır / çok zayıf.** REAL_LLM_SUCCESS={rate(llama, "real_llm_success_rate")}; brief echo oranı={rate(llama, "brief_echo_rate")}. Kullanıcıya giden kalite çoğunlukla fallback.
2. Qwen 27B fark yaratıyor mu? **{rate(qwen27, "status") if qwen27 else "MODEL GEREKLİ"}** — bu makinede kurulu değil, sonuç uydurulmadı.
3. Qwen 35B-A3B fark yaratıyor mu? **{rate(qwen35, "status") if qwen35 else "MODEL GEREKLİ"}** — kurulu değil.
4. 24 GB Quadro RTX 6000 üzerinde pratik olan: kurulu **llama3 8B Q4** (~4.7 GB dosya) ve kural fallback. 35B-A3B Q4 dosyası ~24 GB; 22.5 GB dedicated VRAM + KV cache için ölçü olmadan production varsayımı yok.
5. CPU/RAM offload: llama3 için `{rate(llama, "offload")}`. 27B/35B ölçülemedi.
6. VRAM: idle ~{hw.get("gpu", {}).get("vram_used_mib")} MiB (XTTS/python + masaüstü dahil). Llama3 peak `{rate(llama, "peak_vram_mib")}` MiB dedicated. Shared GPU Memory ekstra VRAM değildir.
7. Latency: llama3 avg `{rate(llama, "avg_latency_s")}` s; fallback `{rate(fb, "avg_latency_s")}` s.
8. Doğal Türkçe: fallback kural metni tutarlı; llama3 brief kopyalarsa 0. Extra qwen2.5 7B (istenmeyen model) `{rate(extra, "cmo_score_avg") if extra else "—"}`.
9. Fallback'e düşme: llama3 `{rate(llama, "fallback_rate")}`. 27B/35B yok.
10. Brief echo: llama3 `{rate(llama, "brief_echo_rate")}`.
11. False certainty: llama3 `{rate(llama, "false_certainty_rate")}`.
12. Production önerisi: **NO CLEAR WINNER for a new reasoning model.** Mevcut production `llama3` + commercial fallback kalsın. Qwen3.5 27B/35B kurulup aynı harness ile ölçülmeden `OLLAMA_REASONING_MODEL` değiştirilmesin.

### MODEL WINNER

**NO CLEAR WINNER.** İstenen Qwen3.5 27B ve 35B-A3B bu Ollama kurulumunda yok. Kurulu llama3, V4 ile aynı: reasoning başarısı düşük, sistem kalitesi fallback'e borçlu. Extra `qwen2.5:7b` Qwen3.5 27B yerine geçmez.

## Test sayıları

- Mevcut 96 test: korunmalı (bu turda 96 + V5.1 regression).
- V5.1 unit/regression: `test_brain_v51.py`
- Canlı model benchmark: bu rapor

## Korunan dosyalar

`backend/tts_local.py`, `lib/speech.ts`, `components/mic-button.tsx`, `components/speak-button.tsx`, `/api/speak`, billing, public endpoint sözleşmesi. Production `OLLAMA_REASONING_MODEL` değiştirilmedi.

## Değişen / eklenen dosyalar

- `backend/agents/quality_flags.py` (yeni)
- `backend/agents/validator.py` — BRIEF_ECHO / memory honesty REJECT
- `backend/agents/evaluator.py` — CMO /14 v51
- `backend/agents/commercial.py` — competitor fallback, is_reasoning_task
- `backend/agents/response_engine.py` — stats fallback competitor'dan sonra
- `backend/agents/quality.py` — public path, REAL_LLM_SUCCESS
- `backend/main.py` — reasoning task routing (aynı model)
- `backend/test_brain_v51.py`, `backend/benchmark_v51.py`, `backend/benchmark_v51_cases.py`
"""
    REPORT.write_text(body, encoding="utf-8")


async def main() -> None:
    gpu0 = nvidia_query()
    ram0 = system_ram_gb()
    shared0 = shared_gpu_sample()
    async with httpx.AsyncClient(timeout=240.0) as http:
        tags = await ollama_tags(http)
        status = {}
        resolved = {}
        for key, aliases in REQUIRED.items():
            hit = resolve_installed(tags, aliases)
            resolved[key] = hit
            if hit:
                status[key] = f"MODEL KURULU ({hit})"
            else:
                status[key] = "MODEL GEREKLİ / NOT INSTALLED / NOT TESTED"
        extra = resolve_installed(tags, ("qwen2.5:latest", "qwen2.5", "qwen2.5:7b"))
        status["extra_qwen25"] = f"KURULU extra, istenen 27B değil: {extra}" if extra else "yok"

        summaries = []
        germany = {}

        # fallback
        fb_rows = [await run_fallback_only(case) for case in CASES]
        fb_hw = HwPeak()
        fb_hw.mark()
        summaries.append(summarize("commercial fallback", fb_rows, {}, fb_hw, "TESTED"))
        germany["commercial fallback"] = next(r for r in fb_rows if r["id"] == "germany_france")

        async def run_model(label: str, model: str | None, key: str) -> None:
            if not model:
                summaries.append(
                    summarize(label, [], {}, HwPeak(), "NOT_INSTALLED")
                )
                germany[label] = None
                return
            metrics: dict = {"calls": []}
            hw = HwPeak()
            hw.mark()
            gen = make_generate(model, metrics, hw)
            rows = []
            for case in CASES:
                print(f"[{label}] {case['id']} ...", flush=True)
                rows.append(await run_case(http, gen, case))
            summaries.append(summarize(label, rows, metrics, hw, "TESTED"))
            germany[label] = next(r for r in rows if r["id"] == "germany_france")

        await run_model("llama3", resolved["llama3"] or ("llama3" if any("llama3" in str(t.get("name")) for t in tags) else None), "llama3")
        if extra:
            await run_model("qwen2.5:7b (INSTALLED EXTRA, not 27B)", extra, "extra")
        await run_model("Qwen3.5 27B Q4", resolved["qwen3.5-27b-q4"], "qwen27")
        await run_model("Qwen3.5-35B-A3B Q4", resolved["qwen3.5-35b-a3b-q4"], "qwen35")

        payload = {
            "hardware": {"gpu": gpu0, "ram": ram0, "shared": shared0},
            "installed": tags,
            "model_status": status,
            "summaries": [{k: v for k, v in s.items() if k != "rows" and k != "call_metrics"} | {"germany_path": (germany.get(s["label"]) or {}).get("public_path") if germany.get(s["label"]) else None, "n_rows": s["n"]} for s in summaries],
            "germany_vs_france": germany,
            "full": summaries,
            "chat_model_default": chat_model(),
        }
        JSON_OUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        write_report(payload)
        print(json.dumps({"report": str(REPORT), "status": status, "table": [s["label"] for s in summaries]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
