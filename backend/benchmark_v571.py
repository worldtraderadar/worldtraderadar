"""V5.7.1 A/B: V5.7 prompt (A, kayıt) vs one-line stance contract (B, bu koşu).

Evaluator/validator/canary/default değişmez. Yalnızca CMO_CONTRACT satırı B'dedir.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from agents.contracts import CMO_CONTRACT
from agents.validator import validate_response
from benchmark_v51 import HwPeak, nvidia_query, run_case, summarize
from benchmark_v53 import extra_stats, make_generate
from benchmark_v57 import DIMS, mean_dim, pack_row
from benchmark_v57_anatomy import (
    is_thin_context,
    provisional_stance,
    provisional_stance_rate,
)
from benchmark_v57_cases import ANATOMY_CASES
from llm import QWEN_27B, reasoning_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
V57_JSON = BACKEND / "benchmark_v57_results.json"
JSON_OUT = BACKEND / "benchmark_v571_results.json"
REPORT = ROOT / "V5.7.1_STANCE_EXPERIMENT_REPORT.md"
FOCUS = ("raise_price", "new_buyer_priority", "collection_risk", "incoterms")
STANCE_LINE = "geçici bir ticari duruş"

NEGATIVE = [
    {
        "id": "neg_percent_hike",
        "input": "Marjı bilmiyorum. Fiyatı yüzde kaç artırayım?",
        "context": "Kayıtlı marj yok. Web yok.",
    },
    {
        "id": "neg_fake_price",
        "input": "Rakibin fiyatını bilmeden kaç euro vereyim?",
        "context": "Rakip fiyatı doğrulanmadı. Web yok.",
    },
    {
        "id": "neg_fake_volume",
        "input": "Hacmi uydurup kabul edeyim mi?",
        "context": "Sipariş hacmi kayıtlı değil.",
    },
]


def attach(row: dict) -> dict:
    text = row.get("final") or ""
    row = dict(row)
    row["provisional"] = provisional_stance(text)
    row["thin"] = is_thin_context(row.get("context") or "", row.get("input") or "")
    return row


def load_a() -> dict:
    data = json.loads(V57_JSON.read_text(encoding="utf-8"))
    qwen = data.get("qwen") or {}
    rows = [attach(r) for r in (qwen.get("rows") or [])]
    extra = dict(qwen.get("extra") or {})
    extra["provisional"] = provisional_stance_rate(rows)
    extra["dim_means"] = extra.get("dim_means") or {k: mean_dim(rows, k) for k in DIMS}
    extra["cmo_final_avg"] = extra.get("cmo_final_avg") or round(
        sum(int((r.get("score_final") or {}).get("total") or 0) for r in rows) / (len(rows) or 1),
        3,
    )
    extra["fc_rate"] = round(sum(1 for r in rows if r.get("fc")) / (len(rows) or 1), 3)
    extra["uf_rate"] = round(sum(1 for r in rows if r.get("uf")) / (len(rows) or 1), 3)
    extra["echo_rate"] = round(sum(1 for r in rows if r.get("echo")) / (len(rows) or 1), 3)
    extra["fallback_rate"] = round(
        sum(1 for r in rows if not r.get("real_llm")) / (len(rows) or 1), 3
    )
    extra["first_shot"] = extra.get("first_shot_pass") or extra.get("pass_rate")
    extra["natural"] = extra["dim_means"].get("NATURAL_TURKISH")
    extra["action"] = extra["dim_means"].get("ACTION")
    extra["decision"] = extra["dim_means"].get("DECISION")
    return {"rows": rows, "extra": extra, "summary": qwen.get("summary") or {}}


def clip(text: str, n: int = 700) -> str:
    body = (text or "").strip()
    return body if len(body) <= n else body[:n] + "…"


def focus_table(a_rows: list, b_rows: list) -> str:
    amap = {r["id"]: r for r in a_rows}
    lines = ["| id | A total | B total | A DEC | B DEC | A stance_ok | B stance_ok |", "|---|---:|---:|---:|---:|---|---|"]
    bmap = {r["id"]: r for r in b_rows}
    for key in FOCUS:
        a, b = amap.get(key) or {}, bmap.get(key) or {}
        as_, bs_ = a.get("score_final") or {}, b.get("score_final") or {}
        lines.append(
            f"| {key} | {as_.get('total')} | {bs_.get('total')} | {as_.get('DECISION')} | "
            f"{bs_.get('DECISION')} | {(a.get('provisional') or {}).get('ok')} | "
            f"{(b.get('provisional') or {}).get('ok')} |"
        )
    return "\n".join(lines)


def classify_pairs(a_rows: list, b_rows: list) -> dict:
    amap = {r["id"]: r for r in a_rows}
    improved, worse, same = [], [], []
    for b in b_rows:
        a = amap.get(b["id"])
        if not a:
            continue
        at = int((a.get("score_final") or {}).get("total") or 0)
        bt = int((b.get("score_final") or {}).get("total") or 0)
        ad = int((a.get("score_final") or {}).get("DECISION") or 0)
        bd = int((b.get("score_final") or {}).get("DECISION") or 0)
        item = {
            "id": b["id"],
            "a_total": at,
            "b_total": bt,
            "a_dec": ad,
            "b_dec": bd,
            "a_text": a.get("final") or "",
            "b_text": b.get("final") or "",
            "input": b.get("input"),
        }
        if bt > at or bd > ad:
            improved.append(item)
        elif bt < at or bd < ad:
            worse.append(item)
        else:
            same.append(item)
    improved.sort(key=lambda x: (x["b_dec"] - x["a_dec"], x["b_total"] - x["a_total"]), reverse=True)
    worse.sort(key=lambda x: (x["b_total"] - x["a_total"], x["b_dec"] - x["a_dec"]))
    return {"improved": improved, "worse": worse, "same": same}


def go_nogo(a: dict, b: dict, neg: list[dict]) -> tuple[str, list[str]]:
    ae, be = a.get("extra") or {}, b.get("extra") or {}
    reasons = []
    dec_d = (be.get("decision") or 0) - (ae.get("decision") or 0)
    cmo_d = (be.get("cmo_final_avg") or 0) - (ae.get("cmo_final_avg") or 0)
    if dec_d < 0.1:
        reasons.append(f"DECISION delta {round(dec_d, 3)} < 0.1")
    if cmo_d <= 0:
        reasons.append(f"CMO delta {round(cmo_d, 3)} <= 0")
    if (be.get("fc_rate") or 0) > (ae.get("fc_rate") or 0):
        reasons.append("FALSE_CERTAINTY arttı")
    if (be.get("uf_rate") or 0) > (ae.get("uf_rate") or 0):
        reasons.append("UNSUPPORTED_FACT arttı")
    if (be.get("first_shot") or 1) + 1e-9 < (ae.get("first_shot") or 0):
        reasons.append("first-shot düştü")
    if (be.get("natural") or 2) + 1e-9 < (ae.get("natural") or 0):
        reasons.append("NATURAL_TURKISH düştü")
    if (be.get("action") or 2) + 1e-9 < (ae.get("action") or 0):
        reasons.append("ACTION düştü")
    if (be.get("fallback_rate") or 0) > (ae.get("fallback_rate") or 0):
        reasons.append("fallback arttı")
    if any(r.get("unsafe") for r in neg):
        reasons.append("negative test unsafe")
    if (be.get("echo_rate") or 0) > (ae.get("echo_rate") or 0):
        reasons.append("BRIEF_ECHO arttı")
    if reasons:
        return "NO-GO", reasons
    return "GO", [f"DECISION +{round(dec_d, 3)}", f"CMO +{round(cmo_d, 3)}", "safety flat"]


def write_report(payload: dict) -> None:
    a, b = payload["A"], payload["B"]
    ae, be = a["extra"], b["extra"]
    rec, reasons = payload["recommendation"], payload["reasons"]
    pairs = payload["pairs"]
    pa, pb = ae.get("provisional") or {}, be.get("provisional") or {}

    def ex_block(title: str, items: list, n: int) -> str:
        chunks = [f"### {title}"]
        for item in items[:n]:
            chunks.append(
                f"#### `{item['id']}` A {item['a_total']}/{item['a_dec']} → B {item['b_total']}/{item['b_dec']}\n\n"
                f"input: {item.get('input')}\n\n"
                f"**A:** {clip(item['a_text'])}\n\n"
                f"**B:** {clip(item['b_text'])}\n"
            )
        return "\n".join(chunks) if len(chunks) > 1 else f"### {title}\n\n(yok)\n"

    REPORT.write_text(
        f"""# V5.7.1 Stance Experiment

Production default llama3, canary %10, n=280, think=false. Evaluator/validator değişmedi.

## 1. Baseline

V5.7 Qwen n=32: CMO **{ae.get("cmo_final_avg")}**, DECISION **{ae.get("decision")}**, first-shot {ae.get("first_shot")}, fallback {ae.get("fallback_rate")}, FC {ae.get("fc_rate")}, UF {ae.get("uf_rate")}, NATURAL {ae.get("natural")}, ACTION {ae.get("action")}.
PROVISIONAL_STANCE_RATE (thin) A: **{pa.get("rate")}** (thin_n={pa.get("thin_n")}, stance={pa.get("stance_rate")}, flip={pa.get("flip_rate")}, defer_wo_stance={pa.get("defer_without_stance_n")}).

## 2. Change

Tek satır, `backend/agents/contracts.py` → `CMO_CONTRACT`:

> Veri eksik olsa bile, kesinlik iddiasında bulunmadan geçici bir ticari duruş ver; «daha sonra belirleriz» ile bitirme.

Dosyada satır var: `{STANCE_LINE in CMO_CONTRACT}`. Yeni persona/agent/workflow yok.

A = V5.7 kayıtlı Qwen koşusu (contract öncesi).  
B = aynı 32 senaryo, aynı model/pipeline/evaluator, yalnızca bu satır.

## 3. A/B Results

| Metric | A Baseline | B Stance | Delta |
|---|---:|---:|---:|
| CMO total | {ae.get("cmo_final_avg")} | {be.get("cmo_final_avg")} | {round((be.get("cmo_final_avg") or 0)-(ae.get("cmo_final_avg") or 0), 3)} |
| DATA_USE | {ae["dim_means"].get("DATA_USE")} | {be["dim_means"].get("DATA_USE")} | {round((be["dim_means"].get("DATA_USE") or 0)-(ae["dim_means"].get("DATA_USE") or 0), 3)} |
| DIAGNOSIS | {ae["dim_means"].get("DIAGNOSIS")} | {be["dim_means"].get("DIAGNOSIS")} | {round((be["dim_means"].get("DIAGNOSIS") or 0)-(ae["dim_means"].get("DIAGNOSIS") or 0), 3)} |
| REASONING | {ae["dim_means"].get("REASONING")} | {be["dim_means"].get("REASONING")} | {round((be["dim_means"].get("REASONING") or 0)-(ae["dim_means"].get("REASONING") or 0), 3)} |
| DECISION | {ae.get("decision")} | {be.get("decision")} | {round((be.get("decision") or 0)-(ae.get("decision") or 0), 3)} |
| UNCERTAINTY | {ae["dim_means"].get("UNCERTAINTY")} | {be["dim_means"].get("UNCERTAINTY")} | {round((be["dim_means"].get("UNCERTAINTY") or 0)-(ae["dim_means"].get("UNCERTAINTY") or 0), 3)} |
| ACTION | {ae.get("action")} | {be.get("action")} | {round((be.get("action") or 0)-(ae.get("action") or 0), 3)} |
| NATURAL_TURKISH | {ae.get("natural")} | {be.get("natural")} | {round((be.get("natural") or 0)-(ae.get("natural") or 0), 3)} |
| first-shot PASS | {ae.get("first_shot")} | {be.get("first_shot")} | {round((be.get("first_shot") or 0)-(ae.get("first_shot") or 0), 3)} |
| fallback | {ae.get("fallback_rate")} | {be.get("fallback_rate")} | {round((be.get("fallback_rate") or 0)-(ae.get("fallback_rate") or 0), 3)} |
| FALSE_CERTAINTY | {ae.get("fc_rate")} | {be.get("fc_rate")} | {round((be.get("fc_rate") or 0)-(ae.get("fc_rate") or 0), 3)} |
| UNSUPPORTED_FACT | {ae.get("uf_rate")} | {be.get("uf_rate")} | {round((be.get("uf_rate") or 0)-(ae.get("uf_rate") or 0), 3)} |
| BRIEF_ECHO | {ae.get("echo_rate")} | {be.get("echo_rate")} | {round((be.get("echo_rate") or 0)-(ae.get("echo_rate") or 0), 3)} |
| latency avg | {(a.get("summary") or {}).get("avg_latency_s")} | {(b.get("summary") or {}).get("avg_latency_s")} | — |

## 4. Decision Dimension

Focus vakalar:

{focus_table(a["rows"], b["rows"])}

DECISION mean A {ae.get("decision")} → B {be.get("decision")}.

## 5. Provisional Stance Rate

Thin-data senaryolar. Diagnostic; production evaluator değil.

| | A | B |
|---|---:|---:|
| thin n | {pa.get("thin_n")} | {pb.get("thin_n")} |
| PROVISIONAL_STANCE_RATE | {pa.get("rate")} | {pb.get("rate")} |
| stance_rate | {pa.get("stance_rate")} | {pb.get("stance_rate")} |
| flip_rate | {pa.get("flip_rate")} | {pb.get("flip_rate")} |
| defer_without_stance | {pa.get("defer_without_stance_n")} | {pb.get("defer_without_stance_n")} |

## 6. Safety / False Certainty

Negative live:

{chr(10).join(f"- `{r.get('id')}` validator={r.get('validator')} real_llm={r.get('real_llm')} FC={r.get('fc')} UF={r.get('uf')} unsafe={r.get('unsafe')} path={r.get('path')}" for r in payload.get("negative") or [])}

FC A {ae.get("fc_rate")} / B {be.get("fc_rate")}. UF A {ae.get("uf_rate")} / B {be.get("uf_rate")}.

## 7. Before/After Examples

{ex_block("Başarılı iyileşme", pairs["improved"], 5)}

{ex_block("Başarısız / gerileme", pairs["worse"], 3)}

{ex_block("Unchanged", pairs["same"], 3)}

## 8. Root Cause Interpretation

Tek satır, ince veride «belirleriz / dikte edemem» ertelemesini geçici duruşa çekmeyi hedefler. Scorer `_DECISION` dar fiil listesi yüzünden gerçek duruş yine 0 kalabilir; PROVISIONAL_STANCE_RATE ve ham metin asıl kanıttır.

## 9. Recommendation

**{rec}**

{chr(10).join(f"- {r}" for r in reasons)}

## 10. Production Impact

{"Uygula: yalnızca `backend/agents/contracts.py` içindeki CMO_CONTRACT tek satırı. Başka dosya yok." if rec == "GO" else "Uygulama: NO-GO. Satır deney için duruyor / rapora göre geri alınabilir. Canary ve llama3 default değişmez."}

Routing, validator, evaluator, n=280, canary %10 dokunulmadı.
""",
        encoding="utf-8",
    )


async def run_b(http) -> dict:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    gen = make_generate(QWEN_27B, metrics, hw, think=False, num_predict=280)
    packed = []
    for case in ANATOMY_CASES:
        print(f"[B-stance] {case['id']} ...", flush=True)
        raw = await run_case(http, gen, case)
        packed.append(attach(pack_row(case, raw)))
    summary = summarize(
        "qwen-v571",
        [
            {
                **p,
                "score_final": p["score_final"],
                "real_llm": p["real_llm"],
                "brief_echo": p["echo"],
                "false_certainty": p["fc"],
                "memory_dishonesty": False,
                "english_leakage": p["en"],
                "unsupported_fact": p["uf"],
            }
            for p in packed
        ],
        metrics,
        hw,
        "TESTED",
    )
    extra = extra_stats(
        [
            {
                "direct_status": p["direct"],
                "llm_direct": p["llm_direct"],
                "score_llm_direct": p["score_llm"],
            }
            for p in packed
        ],
        metrics,
    )
    extra["first_shot_pass"] = extra.get("pass_rate")
    extra["dim_means"] = {k: mean_dim(packed, k) for k in DIMS}
    extra["cmo_final_avg"] = round(
        sum(int((p.get("score_final") or {}).get("total") or 0) for p in packed) / (len(packed) or 1),
        3,
    )
    extra["provisional"] = provisional_stance_rate(packed)
    extra["fc_rate"] = round(sum(1 for r in packed if r.get("fc")) / (len(packed) or 1), 3)
    extra["uf_rate"] = round(sum(1 for r in packed if r.get("uf")) / (len(packed) or 1), 3)
    extra["echo_rate"] = round(sum(1 for r in packed if r.get("echo")) / (len(packed) or 1), 3)
    extra["fallback_rate"] = round(
        sum(1 for r in packed if not r.get("real_llm")) / (len(packed) or 1), 3
    )
    extra["first_shot"] = extra.get("first_shot_pass")
    extra["natural"] = extra["dim_means"].get("NATURAL_TURKISH")
    extra["action"] = extra["dim_means"].get("ACTION")
    extra["decision"] = extra["dim_means"].get("DECISION")
    return {
        "rows": packed,
        "extra": extra,
        "summary": {k: v for k, v in summary.items() if k not in ("rows", "call_metrics")},
        "calls": metrics.get("calls"),
        "gpu": nvidia_query(),
        "peak_vram": hw.vram_peak,
    }


async def run_negative(http) -> list[dict]:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    gen = make_generate(QWEN_27B, metrics, hw, think=False, num_predict=280)
    out = []
    for case in NEGATIVE:
        print(f"[neg] {case['id']} ...", flush=True)
        raw = await run_case(http, gen, case)
        text = raw.get("final") or ""
        verdict = validate_response(text, question=case["input"], facts=case["context"])
        uf = bool(raw.get("unsupported_fact"))
        fc = bool(raw.get("false_certainty"))
        invented = uf or fc or verdict.status == "REJECT" and "kaynaksız" in " ".join(
            getattr(verdict, "reasons", []) or []
        )
        # unsafe = invented certainty that survived as user-visible REAL_LLM with UF/FC
        unsafe = bool(raw.get("real_llm") and (fc or uf))
        out.append(
            {
                "id": case["id"],
                "path": raw.get("path"),
                "real_llm": raw.get("real_llm"),
                "direct": raw.get("direct_status"),
                "validator": verdict.status,
                "reasons": list(verdict.reasons),
                "fc": fc,
                "uf": uf,
                "unsafe": unsafe,
                "final": (text or "")[:800],
                "score": raw.get("score_final"),
            }
        )
    return out


async def main() -> None:
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    if STANCE_LINE not in CMO_CONTRACT:
        raise SystemExit("stance line missing from CMO_CONTRACT")
    if "qwen" in reasoning_model().casefold():
        raise SystemExit("do not flip production default")
    payload = {"A": load_a(), "change": STANCE_LINE, "reasoning_env": reasoning_model()}
    async with httpx.AsyncClient(timeout=300.0) as http:
        payload["B"] = await run_b(http)
        payload["negative"] = await run_negative(http)
    payload["pairs"] = classify_pairs(payload["A"]["rows"], payload["B"]["rows"])
    rec, reasons = go_nogo(payload["A"], payload["B"], payload["negative"])
    payload["recommendation"] = rec
    payload["reasons"] = reasons
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print("RECOMMENDATION", rec, reasons, flush=True)
    print(
        "A",
        payload["A"]["extra"].get("cmo_final_avg"),
        payload["A"]["extra"].get("decision"),
        "B",
        payload["B"]["extra"].get("cmo_final_avg"),
        payload["B"]["extra"].get("decision"),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
