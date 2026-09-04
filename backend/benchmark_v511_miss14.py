"""V5.11 focused rerun: the 14 V5.10 Qwen DECISION=0 keys.

Supporting diagnostic for V5.11_REAL_DECISION_MISS_REPORT.md.
Full-suite freeze is benchmark_v511_n32.py. Does not flip production default.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from agents.evaluator import has_decision_stance
from benchmark_v51 import HwPeak, nvidia_query, run_case
from benchmark_v53 import extra_stats, make_generate
from benchmark_v57 import DIMS, mean_dim, pack_row
from benchmark_v57_cases import ANATOMY_CASES
from llm import QWEN_27B, reasoning_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
V510_JSON = BACKEND / "benchmark_v510_compare_results.json"
JSON_OUT = BACKEND / "benchmark_v511_miss14_results.json"
REPORT = ROOT / "V5.11_REAL_DECISION_MISS_REPORT.md"


def miss_ids() -> list[str]:
    data = json.loads(V510_JSON.read_text(encoding="utf-8"))
    rows = ((data.get("qwen") or {}).get("metrics") or {}).get("rows") or []
    out = []
    for row in rows:
        if int((row.get("score_final") or {}).get("DECISION") or 0) == 0:
            out.append(str(row.get("id") or ""))
    return [item for item in out if item]


def v510_row(cid: str) -> dict:
    data = json.loads(V510_JSON.read_text(encoding="utf-8"))
    for row in ((data.get("qwen") or {}).get("metrics") or {}).get("rows") or []:
        if row.get("id") == cid:
            return row
    return {}


def write_report(payload: dict) -> None:
    before = payload.get("before") or {}
    after = payload.get("after") or {}
    lines = [
        "# V5.11 Qwen REAL_DECISION_MISS",
        "",
        "Production default llama3, Qwen %10 canary. Evaluator / validator / SHA-256 canary değişmedi.",
        "V5.7.1 `geçici bir ticari duruş` satırı duruyor.",
        "",
        "## Failure analysis (V5.10 live Qwen, DECISION=0)",
        "",
        f"Keys ({len(payload.get('ids') or [])}): `{', '.join(payload.get('ids') or [])}`",
        "",
        "Trend: kısmi veride soru listesi duruşun yerini alıyordu. CMO tetikleyicisi",
        "`Eksik kritik veri varsa en fazla 1–3 soru sor` + brief'te soruların duruştan önce basılması.",
        "",
        "## Prompt / contract",
        "",
        "- CMO: kritik metrik yoksa önce açık geçici duruş; sorular sonra.",
        "- Brief: `ÖNCE DURUŞ` → aksiyon → eksikler/sorular.",
        "- Kısmi seçim (MOQ, tedarikçi, FOB/CIF, kredi, dönüşüm, zam) için geçici hold.",
        "",
        "## Re-eval of the same 14",
        "",
        "| Metric | V5.10 Qwen | V5.11 Qwen |",
        "|---|---:|---:|",
        f"| n | {before.get('n')} | {after.get('n')} |",
        f"| DECISION mean | {before.get('decision')} | {after.get('decision')} |",
        f"| DECISION pass (DEC≥1) | {before.get('decision_pass')} | {after.get('decision_pass')} |",
        f"| REAL_DECISION_MISS | {before.get('real_miss')} | {after.get('real_miss')} |",
        f"| ACTION mean | {before.get('action')} | {after.get('action')} |",
        f"| FALSE_CERTAINTY | {before.get('fc_n')} | {after.get('fc_n')} |",
        f"| UNSUPPORTED_FACT | {before.get('uf_n')} | {after.get('uf_n')} |",
        f"| fallback | {before.get('fallback')} | {after.get('fallback')} |",
        f"| latency avg (s) | {before.get('latency')} | {after.get('latency')} |",
        "",
        "## Remaining DEC=0",
        "",
    ]
    remain = payload.get("still_zero") or []
    if remain:
        lines.extend(f"- `{item}`" for item in remain)
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "Routing değişmedi. Llama3 default + Qwen canary durur.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def metrics(rows: list[dict]) -> dict:
    n = len(rows) or 1
    dec = [int((r.get("score_final") or {}).get("DECISION") or 0) for r in rows]
    act = [int((r.get("score_final") or {}).get("ACTION") or 0) for r in rows]
    return {
        "n": len(rows),
        "decision": round(sum(dec) / n, 3),
        "decision_pass": round(sum(1 for v in dec if v >= 1) / n, 3),
        "action": round(sum(act) / n, 3),
        "real_miss": sum(1 for r in rows if not has_decision_stance(r.get("final") or "")),
        "fc_n": sum(1 for r in rows if r.get("fc")),
        "uf_n": sum(1 for r in rows if r.get("uf")),
        "fallback": round(sum(1 for r in rows if not r.get("real_llm")) / n, 3),
        "latency": round(sum(float(r.get("wall_s") or 0) for r in rows) / n, 3),
        "dims": {k: mean_dim(rows, k) for k in DIMS},
    }


async def main() -> None:
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    if "qwen" in reasoning_model().casefold():
        raise SystemExit("do not flip production default")
    ids = miss_ids()
    if len(ids) != 14:
        raise SystemExit(f"expected 14 V5.10 misses, got {ids}")
    wanted = {row["id"]: row for row in ANATOMY_CASES}
    cases = [wanted[cid] for cid in ids]
    before_rows = [v510_row(cid) for cid in ids]
    metrics_calls: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    packed = []
    async with httpx.AsyncClient(timeout=300.0) as http:
        gen = make_generate(QWEN_27B, metrics_calls, hw, think=False, num_predict=280)
        for case in cases:
            print(f"[qwen-v511] {case['id']} ...", flush=True)
            raw = await run_case(http, gen, case)
            packed.append(pack_row(case, raw))
    after = metrics(packed)
    before = metrics(before_rows)
    still = [
        r.get("id")
        for r in packed
        if int((r.get("score_final") or {}).get("DECISION") or 0) == 0
    ]
    payload = {
        "reasoning_env": reasoning_model(),
        "gpu": nvidia_query(),
        "ids": ids,
        "before": before,
        "after": after,
        "still_zero": still,
        "rows": packed,
        "extra": extra_stats(
            [
                {
                    "direct_status": p.get("direct"),
                    "llm_direct": p.get("llm_direct"),
                    "score_llm_direct": p.get("score_llm"),
                }
                for p in packed
            ],
            metrics_calls,
        ),
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(
        "V5.11",
        "DEC", after.get("decision"),
        "pass", after.get("decision_pass"),
        "miss", after.get("real_miss"),
        "FC", after.get("fc_n"),
        "UF", after.get("uf_n"),
        "still", still,
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
