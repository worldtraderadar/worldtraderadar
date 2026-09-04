"""V5.11 stable n=32 Qwen CMO rerun.

Kept next to the V5.1–V5.10 harness (not a throwaway). Writes
`benchmark_v511_n32_results.json` and refreshes `V5.11_N32_QWEN_BASELINE_REPORT.md`.
Does not set OLLAMA_REASONING_MODEL or change canary percent.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from benchmark_v51 import nvidia_query
from benchmark_v510_compare import metrics
from benchmark_v57 import run_model
from llm import QWEN_27B, reasoning_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
V510_JSON = BACKEND / "benchmark_v510_compare_results.json"
JSON_OUT = BACKEND / "benchmark_v511_n32_results.json"
REPORT = ROOT / "V5.11_N32_QWEN_BASELINE_REPORT.md"


def v510_qwen() -> dict:
    if not V510_JSON.exists():
        return {}
    data = json.loads(V510_JSON.read_text(encoding="utf-8"))
    return (data.get("qwen") or {}).get("metrics") or {}


def delta(now: float | int | None, then: float | int | None) -> str:
    if now is None or then is None:
        return "—"
    diff = float(now) - float(then)
    sign = "+" if diff >= 0 else ""
    if isinstance(now, int) and isinstance(then, int):
        return f"{sign}{int(diff)}"
    return f"{sign}{diff:.3f}"


def write_report(payload: dict) -> None:
    now = payload["metrics"]
    old = payload.get("v510") or {}
    still = payload.get("still_zero") or []
    extra = payload.get("raw_extra") or {}
    summary = payload.get("summary") or {}

    def row(name: str, key: str) -> str:
        a, b = now.get(key), old.get(key)
        return f"| {name} | {b} | {a} | {delta(a, b)} |"

    REPORT.write_text(
        f"""# V5.11 n=32 Qwen baseline

Live anatomy suite, think=false, num_predict=280, V5.11 CMO_CONTRACT + brief order.
Evaluator `score_cmo_v51` unchanged. Production `OLLAMA_REASONING_MODEL` **not changed**
(`{payload.get("reasoning_env")}`). Canary routing unchanged.

## vs V5.10 Qwen (n=32)

| Metric | V5.10 | V5.11 | Δ |
|---|---:|---:|---:|
{row("DECISION mean", "decision")}
{row("DECISION pass (DEC≥1)", "decision_pass")}
{row("DECISION =2 rate", "decision_two")}
{row("ACTION mean", "action")}
{row("CMO total", "cmo")}
{row("REAL_DECISION_MISS", "real_miss")}
{row("FALSE_CERTAINTY n", "fc_n")}
{row("UNSUPPORTED_FACT n", "uf_n")}
{row("latency avg (s)", "latency")}
{row("REAL_LLM", "real_llm")}
{row("first-shot", "first_shot")}
{row("fallback", "fallback")}

## Safety

FALSE_CERTAINTY={now.get("fc_n")} UNSUPPORTED_FACT={now.get("uf_n")}. Target: 0 / 0.

## Remaining DEC=0

{chr(10).join(f"- `{item}`" for item in still) if still else "None."}

first-shot={extra.get("first_shot_pass")} reject={extra.get("reject_rate")}
peak VRAM={summary.get("peak_vram_mib")} MiB tokens/s={summary.get("tokens_per_sec")}

This run does not change production default or canary percent.
""",
        encoding="utf-8",
    )


async def main() -> None:
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    if "qwen" in reasoning_model().casefold():
        raise SystemExit("do not flip production default")
    async with httpx.AsyncClient(timeout=300.0) as http:
        print(f"[qwen-v511-n32] {QWEN_27B}", flush=True)
        raw = await run_model(http, QWEN_27B, "qwen-v511-n32")
    now = metrics(raw)
    still = [
        r.get("id")
        for r in now.get("rows") or []
        if int((r.get("score_final") or {}).get("DECISION") or 0) == 0
    ]
    payload = {
        "reasoning_env": reasoning_model(),
        "gpu": nvidia_query(),
        "evaluator": "score_cmo_v51 (V5.10 matcher, unchanged)",
        "contract": "V5.11 stance-before-questions",
        "v510": {k: v for k, v in v510_qwen().items() if k != "rows"},
        "metrics": {k: v for k, v in now.items() if k != "rows"},
        "raw_extra": raw.get("extra"),
        "summary": raw.get("summary"),
        "still_zero": still,
        "rows": raw.get("rows"),
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    m = payload["metrics"]
    print(
        "V5.11-n32 CMO", m.get("cmo"), "DEC", m.get("decision"),
        "pass", m.get("decision_pass"), "miss", m.get("real_miss"),
        "ACTION", m.get("action"), "FC", m.get("fc_n"), "UF", m.get("uf_n"),
        "lat", m.get("latency"), "still", still,
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
