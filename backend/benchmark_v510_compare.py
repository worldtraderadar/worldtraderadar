"""V5.10 Qwen vs Llama3 n=32 CMO, updated score_cmo_v51.

Production OLLAMA_REASONING_MODEL değiştirilmez. Canary/default/think/n=280 aynı.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from agents.evaluator import has_decision_stance
from benchmark_v51 import nvidia_query
from benchmark_v57 import DIMS, mean_dim, run_model
from llm import QWEN_27B, chat_model, reasoning_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
JSON_OUT = BACKEND / "benchmark_v510_compare_results.json"
REPORT = ROOT / "V5.10_QWEN_VS_LLAMA3_REPORT.md"


def attach(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        text = item.get("final") or ""
        item["has_stance"] = has_decision_stance(text)
        item["real_miss"] = not item["has_stance"]
        out.append(item)
    return out


def metrics(block: dict) -> dict:
    rows = attach(block.get("rows") or [])
    n = len(rows) or 1
    extra = dict(block.get("extra") or {})
    dims = extra.get("dim_means") or {k: mean_dim(rows, k) for k in DIMS}
    dec_vals = [int((r.get("score_final") or {}).get("DECISION") or 0) for r in rows]
    return {
        "n": len(rows),
        "cmo": extra.get("cmo_final_avg")
        or round(sum(int((r.get("score_final") or {}).get("total") or 0) for r in rows) / n, 3),
        "decision": dims.get("DECISION"),
        "decision_pass": round(sum(1 for v in dec_vals if v >= 1) / n, 3),
        "decision_two": round(sum(1 for v in dec_vals if v >= 2) / n, 3),
        "action": dims.get("ACTION"),
        "uncertainty": dims.get("UNCERTAINTY"),
        "natural": dims.get("NATURAL_TURKISH"),
        "real_miss": sum(1 for r in rows if r.get("real_miss")),
        "stance_n": sum(1 for r in rows if r.get("has_stance")),
        "fc": round(sum(1 for r in rows if r.get("fc")) / n, 3),
        "uf": round(sum(1 for r in rows if r.get("uf")) / n, 3),
        "echo": round(sum(1 for r in rows if r.get("echo")) / n, 3),
        "fc_n": sum(1 for r in rows if r.get("fc")),
        "uf_n": sum(1 for r in rows if r.get("uf")),
        "fallback": round(sum(1 for r in rows if not r.get("real_llm")) / n, 3),
        "real_llm": round(sum(1 for r in rows if r.get("real_llm")) / n, 3),
        "first_shot": extra.get("first_shot_pass") or extra.get("pass_rate"),
        "latency": (block.get("summary") or {}).get("avg_latency_s"),
        "dims": dims,
        "rows": rows,
    }


def recommend(q: dict, llama: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if (q.get("fc_n") or 0) or (q.get("uf_n") or 0):
        return "KEEP_LLAMA_DEFAULT", ["Qwen FALSE_CERTAINTY veya UNSUPPORTED_FACT > 0"]
    if (llama.get("fc_n") or 0) or (llama.get("uf_n") or 0):
        reasons.append("Llama safety ihlali (Qwen 0)")
    q_dec, l_dec = q.get("decision") or 0, llama.get("decision") or 0
    q_cmo, l_cmo = q.get("cmo") or 0, llama.get("cmo") or 0
    q_lat, l_lat = q.get("latency") or 0, llama.get("latency") or 0
    if q_dec + 0.05 < l_dec and q_cmo + 0.2 < l_cmo:
        return "KEEP_LLAMA_DEFAULT", ["Llama CMO/DECISION daha yüksek"]
    if (q.get("real_llm") or 0) < 0.9:
        return "KEEP_LLAMA_DEFAULT", [f"Qwen REAL_LLM {q.get('real_llm')} < 0.9"]
    reasons.append(f"Qwen CMO {q_cmo} vs Llama {l_cmo}")
    reasons.append(f"Qwen DECISION {q_dec} vs Llama {l_dec}")
    reasons.append(f"Qwen latency {q_lat}s vs Llama {l_lat}s")
    reasons.append("VRAM dual-resident + mevcut %10 canary; default flip operasyonel risk")
    # Quality win is not an automatic production default switch.
    return "KEEP_LLAMA_DEFAULT", reasons


def write_report(payload: dict) -> None:
    q, llama = payload["qwen"]["metrics"], payload["llama"]["metrics"]
    rec, reasons = payload["recommendation"], payload["reasons"]

    def row(name: str, key: str) -> str:
        return f"| {name} | {q.get(key)} | {llama.get(key)} |"

    REPORT.write_text(
        f"""# V5.10 Qwen vs Llama3

n=32 anatomy, think=false, num_predict=280, current CMO_CONTRACT + updated `score_cmo_v51`.
Production `OLLAMA_REASONING_MODEL` **değiştirilmedi** (`{payload.get("reasoning_env")}`).

## Comparison

| Metric | Qwen 27B | Llama3 |
|---|---:|---:|
{row("CMO total", "cmo")}
{row("DECISION mean", "decision")}
{row("DECISION pass (DEC≥1)", "decision_pass")}
{row("DECISION =2 rate", "decision_two")}
{row("ACTION mean", "action")}
{row("REAL_DECISION_MISS", "real_miss")}
{row("FALSE_CERTAINTY n", "fc_n")}
{row("UNSUPPORTED_FACT n", "uf_n")}
{row("BRIEF_ECHO rate", "echo")}
{row("REAL_LLM", "real_llm")}
{row("first-shot", "first_shot")}
{row("fallback", "fallback")}
{row("latency avg (s)", "latency")}
{row("NATURAL_TURKISH", "natural")}

## Safety

Qwen FC={q.get("fc_n")} UF={q.get("uf_n")}. Llama FC={llama.get("fc_n")} UF={llama.get("uf_n")}.
Hedef: her iki modelde 0.

## Recommendation

**{rec}**

{chr(10).join(f"- {r}" for r in reasons)}

Default llama3 + Qwen %10 canary korunur. Bu koşu production routing değiştirmez.
""",
        encoding="utf-8",
    )


async def main() -> None:
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    llama_name = chat_model()
    if "qwen" in reasoning_model().casefold():
        raise SystemExit("do not flip production default")
    payload = {
        "reasoning_env": reasoning_model(),
        "gpu": nvidia_query(),
        "evaluator": "score_cmo_v51 updated V5.10",
    }
    async with httpx.AsyncClient(timeout=300.0) as http:
        print(f"[qwen] {QWEN_27B}", flush=True)
        qwen_raw = await run_model(http, QWEN_27B, "qwen-v510")
        print(f"[llama] {llama_name}", flush=True)
        llama_raw = await run_model(http, llama_name, "llama3-v510")
    payload["qwen"] = {"raw_extra": qwen_raw.get("extra"), "summary": qwen_raw.get("summary"), "metrics": metrics(qwen_raw), "rows": qwen_raw.get("rows")}
    payload["llama"] = {"raw_extra": llama_raw.get("extra"), "summary": llama_raw.get("summary"), "metrics": metrics(llama_raw), "rows": llama_raw.get("rows")}
    rec, reasons = recommend(payload["qwen"]["metrics"], payload["llama"]["metrics"])
    payload["recommendation"] = rec
    payload["reasons"] = reasons
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    qm, lm = payload["qwen"]["metrics"], payload["llama"]["metrics"]
    print("RECOMMENDATION", rec, flush=True)
    print(
        "Qwen CMO", qm.get("cmo"), "DEC", qm.get("decision"), "miss", qm.get("real_miss"),
        "FC", qm.get("fc_n"), "UF", qm.get("uf_n"), "lat", qm.get("latency"),
        flush=True,
    )
    print(
        "Llama CMO", lm.get("cmo"), "DEC", lm.get("decision"), "miss", lm.get("real_miss"),
        "FC", lm.get("fc_n"), "UF", lm.get("uf_n"), "lat", lm.get("latency"),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
