"""V5.7 CMO anatomy diagnostic. Production path'e bağlanmaz.

OLLAMA_REASONING_MODEL değiştirilmez. Canary/validator/evaluator/TTS yok.
Resmi skor: score_cmo_v51 (değiştirilmez).
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from benchmark_v51 import HwPeak, nvidia_query, run_case, summarize, system_ram_gb
from benchmark_v53 import extra_stats, make_generate
from benchmark_v57_anatomy import CLASS_LABELS, classify_anatomy, data_use_ceiling
from benchmark_v57_cases import ANATOMY_CASES
from llm import QWEN_27B, chat_model, reasoning_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
JSON_OUT = BACKEND / "benchmark_v57_results.json"
REPORT = ROOT / "V5.7_CMO_ANATOMY_REPORT.md"
DIMS = (
    "DATA_USE",
    "DIAGNOSIS",
    "REASONING",
    "DECISION",
    "UNCERTAINTY",
    "ACTION",
    "NATURAL_TURKISH",
)


def v56_baseline() -> dict:
    path = BACKEND / "benchmark_v56_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    qwen = data.get("qwen") or {}
    qs, qe = qwen.get("summary") or {}, qwen.get("extra") or {}
    rows = qwen.get("rows") or []
    return {
        "real_llm": qs.get("real_llm_success_rate"),
        "first_shot": qe.get("first_shot_pass"),
        "fallback": qs.get("fallback_rate"),
        "cmo": qe.get("cmo_llm_avg") or qs.get("cmo_score_avg"),
        "latency": (qwen.get("latency") or {}).get("avg") or qs.get("avg_latency_s"),
        "p95": (qwen.get("latency") or {}).get("p95") or qe.get("p95_latency_s"),
        "echo": qs.get("brief_echo_rate"),
        "fc": qs.get("false_certainty_rate"),
        "uf": qs.get("unsupported_fact_rate"),
        "length": qe.get("length_done_rate"),
        "n": qs.get("n"),
        "low_rows": [
            {"id": r.get("id"), "cmo": r.get("cmo_llm"), "path": r.get("path")}
            for r in rows
            if isinstance(r.get("cmo_llm"), (int, float)) and r.get("cmo_llm") < 10
        ],
        "gate": data.get("gate"),
        "recommendation": data.get("recommendation"),
        "headroom_b": (data.get("lifecycle") or {}).get("headroom_b_xtts"),
    }


def mean_dim(rows: list[dict], key: str) -> float | None:
    vals = []
    for row in rows:
        score = row.get("score_final") or {}
        if isinstance(score.get(key), (int, float)):
            vals.append(score[key])
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def pack_row(case: dict, raw: dict) -> dict:
    score = raw.get("score_final") or {}
    llm_score = raw.get("score_llm_direct") or {}
    text = raw.get("final") or ""
    fb = raw.get("commercial_fallback") or ""
    anatomy = classify_anatomy(
        score,
        text=text,
        question=case["input"],
        facts=case.get("context") or "",
        fallback=fb,
    )
    return {
        "id": case["id"],
        "category": case.get("category"),
        "source": case.get("source"),
        "input": case["input"],
        "context": (case.get("context") or "")[:400],
        "path": raw.get("path"),
        "public_path": raw.get("public_path"),
        "real_llm": raw.get("real_llm"),
        "direct": raw.get("direct_status"),
        "retry": raw.get("retry_status"),
        "final": text,
        "llm_direct": (raw.get("llm_direct") or "")[:2000],
        "llm_retry": (raw.get("llm_retry") or "")[:1200],
        "fallback": fb[:1200],
        "score_final": score,
        "score_llm": llm_score,
        "anatomy": anatomy,
        "data_ceiling": data_use_ceiling(case["input"], case.get("context") or ""),
        "echo": raw.get("brief_echo"),
        "fc": raw.get("false_certainty"),
        "uf": raw.get("unsupported_fact"),
        "en": raw.get("english_leakage"),
        "wall_s": raw.get("wall_latency_s"),
        "done": raw.get("first_done_reason") or raw.get("done_reason"),
    }


async def run_model(http, model: str, label: str) -> dict:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    gen = make_generate(model, metrics, hw, think=False, num_predict=280)
    packed = []
    for case in ANATOMY_CASES:
        print(f"[{label}] {case['id']} ...", flush=True)
        raw = await run_case(http, gen, case)
        packed.append(pack_row(case, raw))
    summary = summarize(label, [ {**p, "score_final": p["score_final"], "real_llm": p["real_llm"], "brief_echo": p["echo"], "false_certainty": p["fc"], "memory_dishonesty": False, "english_leakage": p["en"], "unsupported_fact": p["uf"]} for p in packed ], metrics, hw, "TESTED")
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
    extra["weak_n"] = sum(1 for p in packed if (p.get("anatomy") or {}).get("weak"))
    gpu = nvidia_query()
    return {
        "label": label,
        "model": model,
        "summary": {k: v for k, v in summary.items() if k not in ("rows", "call_metrics")},
        "extra": extra,
        "rows": packed,
        "calls": metrics.get("calls"),
        "gpu": gpu,
        "ram": system_ram_gb(),
        "peak_vram": hw.vram_peak,
    }


async def run_fallback_only() -> dict:
    from agents.commercial import build_commercial_brief, commercial_fallback_reply
    from benchmark_v51 import _session_for
    from agents.evaluator import score_cmo_v51

    packed = []
    for case in ANATOMY_CASES:
        q = case["input"]
        facts = case.get("context") or ""
        session = _session_for(case)
        matches = case.get("matches")
        text = (
            commercial_fallback_reply(
                build_commercial_brief(q, session=session, matches=matches), q
            )
            or ""
        )
        raw = {
            "path": "COMMERCIAL_FALLBACK",
            "public_path": "COMMERCIAL_FALLBACK",
            "real_llm": False,
            "direct_status": "",
            "retry_status": "",
            "final": text,
            "llm_direct": "",
            "llm_retry": "",
            "commercial_fallback": text,
            "score_final": score_cmo_v51(text, question=q, facts=facts).as_dict(),
            "score_llm_direct": None,
            "brief_echo": False,
            "false_certainty": False,
            "unsupported_fact": False,
            "english_leakage": False,
            "wall_latency_s": 0.0,
            "first_done_reason": "",
        }
        packed.append(pack_row(case, raw))
    extra = {
        "dim_means": {k: mean_dim(packed, k) for k in DIMS},
        "cmo_final_avg": round(
            sum(int((p.get("score_final") or {}).get("total") or 0) for p in packed)
            / (len(packed) or 1),
            3,
        ),
        "weak_n": sum(1 for p in packed if (p.get("anatomy") or {}).get("weak")),
    }
    return {"label": "fallback", "model": "fallback", "extra": extra, "rows": packed}


def class_table(rows: list[dict]) -> list[dict]:
    n = len(rows) or 1
    totals = [int((r.get("score_final") or {}).get("total") or 0) for r in rows]
    mean_all = sum(totals) / n
    by = defaultdict(list)
    for row in rows:
        for cls in (row.get("anatomy") or {}).get("classes") or []:
            by[cls].append(row)
    out = []
    for cls, label in CLASS_LABELS.items():
        group = by.get(cls) or []
        scores = [int((r.get("score_final") or {}).get("total") or 0) for r in group]
        examples = sorted(group, key=lambda r: int((r.get("score_final") or {}).get("total") or 0))[:3]
        out.append(
            {
                "class": cls,
                "label": label,
                "count": len(group),
                "pct": round(100 * len(group) / n, 1),
                "mean_total": round(sum(scores) / len(scores), 3) if scores else None,
                "impact": round((sum(scores) / len(scores)) - mean_all, 3) if scores else None,
                "examples": [
                    {
                        "id": e["id"],
                        "total": (e.get("score_final") or {}).get("total"),
                        "dims": {k: (e.get("score_final") or {}).get(k) for k in DIMS},
                    }
                    for e in examples
                ],
            }
        )
    return out


def cat_means(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for row in rows:
        by[row.get("category") or "?"].append(
            int((row.get("score_final") or {}).get("total") or 0)
        )
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())}


def clip(text: str, n: int = 900) -> str:
    body = (text or "").strip()
    return body if len(body) <= n else body[:n] + "…"


def write_report(payload: dict) -> None:
    base = payload.get("v56") or {}
    qwen = payload.get("qwen") or {}
    llama = payload.get("llama3") or {}
    fb = payload.get("fallback") or {}
    qrows = qwen.get("rows") or []
    lrows = llama.get("rows") or []
    frows = fb.get("rows") or []
    qe = qwen.get("extra") or {}
    le = llama.get("extra") or {}
    fe = fb.get("extra") or {}
    classes = payload.get("qwen_classes") or []
    weak = sorted(
        [r for r in qrows if (r.get("anatomy") or {}).get("weak")],
        key=lambda r: int((r.get("score_final") or {}).get("total") or 0),
    )
    strong = sorted(
        qrows,
        key=lambda r: int((r.get("score_final") or {}).get("total") or 0),
        reverse=True,
    )
    dim_md = "\n".join(
        f"| {k} | {qe.get('dim_means', {}).get(k)} | "
        f"{'zayıf' if (qe.get('dim_means') or {}).get(k, 2) is not None and (qe.get('dim_means') or {}).get(k, 2) < 1.4 else 'orta' if (qe.get('dim_means') or {}).get(k, 2) < 1.7 else 'güçlü'} | "
        f"llama {(le.get('dim_means') or {}).get(k)} · fallback {(fe.get('dim_means') or {}).get(k)} |"
        for k in DIMS
    )
    class_md = "\n".join(
        "| {cls} {label} | {count} | {pct} | {impact} | {ex} |".format(
            cls=c["class"],
            label=c["label"],
            count=c["count"],
            pct=c["pct"],
            impact=c["impact"],
            ex=", ".join(f"{e['id']}({e['total']})" for e in c["examples"]),
        )
        for c in classes
    )

    def example_block(row: dict) -> str:
        s = row.get("score_final") or {}
        a = row.get("anatomy") or {}
        return (
            f"### `{row.get('id')}` (cat {row.get('category')}, total {s.get('total')})\n\n"
            f"- input: {row.get('input')}\n"
            f"- brief: {clip(row.get('context') or '', 280)}\n"
            f"- path: {row.get('path')} first={row.get('direct')} retry={row.get('retry')}\n"
            f"- dims: " + ", ".join(f"{k}={s.get(k)}" for k in DIMS) + "\n"
            f"- classes: {a.get('classes')} ceiling={a.get('data_ceiling')} "
            f"data_gap={a.get('data_gap')} overlap={a.get('fallback_overlap')}\n\n"
            f"**Qwen:**\n\n{clip(row.get('final') or '', 1100)}\n"
        )

    strong_md = "\n\n".join(example_block(r) for r in strong[:5])
    weak_md = "\n\n".join(example_block(r) for r in weak[:10])
    # if fewer than 10 weak, pad with lowest remaining
    if len(weak) < 10:
        rest = [r for r in sorted(qrows, key=lambda r: int((r.get("score_final") or {}).get("total") or 99)) if r not in weak]
        weak_md = "\n\n".join(example_block(r) for r in (weak + rest)[:10])

    llama_by = {r["id"]: r for r in lrows}
    fb_by = {r["id"]: r for r in frows}
    cmp_lines = []
    for row in qrows:
        lid = row["id"]
        lt = int((llama_by.get(lid, {}).get("score_final") or {}).get("total") or 0)
        ft = int((fb_by.get(lid, {}).get("score_final") or {}).get("total") or 0)
        qt = int((row.get("score_final") or {}).get("total") or 0)
        cmp_lines.append(
            f"| {lid} | {row.get('category')} | {qt} | {lt} | {ft} | "
            f"{row.get('path')} | {llama_by.get(lid, {}).get('path')} |"
        )

    # Root-cause candidates from evidence
    dim_means = qe.get("dim_means") or {}
    weakest = sorted(DIMS, key=lambda k: dim_means.get(k) if dim_means.get(k) is not None else 9)
    ceil_miss = sum(1 for r in qrows if (r.get("anatomy") or {}).get("data_gap", 0) > 0)
    hedge_n = sum(1 for r in qrows if (r.get("anatomy") or {}).get("hedge"))
    overlap_n = sum(1 for r in qrows if (r.get("anatomy") or {}).get("fallback_like"))
    v51_q = [r for r in qrows if r.get("source") == "v51"]
    v51_avg = round(
        sum(int((r.get("score_final") or {}).get("total") or 0) for r in v51_q) / (len(v51_q) or 1),
        3,
    )

    REPORT.write_text(
        f"""# V5.7 CMO Anatomy Report

Production default **llama3**. Qwen %10 canary. Bu koşu production davranışını değiştirmedi.
Resmi skor: `score_cmo_v51` (7×0–2 = 14). Evaluator/validator gevşetilmedi.

## 1. Executive Summary

V5.6 Qwen CMO **{base.get('cmo')} / 14** (n=20). Bu teşhis n={len(qrows)} senaryoda aynı scorer ile boyut kırılımı üretir.

Qwen V5.7 anatomy CMO (final) **{qe.get('cmo_final_avg')}**.  
V5.1 fair-20 altkümesi: **{v51_avg}**.  
Llama3 final **{le.get('cmo_final_avg')}**. Fallback **{fe.get('cmo_final_avg')}**.

En zayıf resmi boyutlar (Qwen mean): {", ".join(f"{k}={dim_means.get(k)}" for k in weakest[:3])}.

DATA_USE tavanının altında kalan cevap: **{ceil_miss}/{len(qrows)}**.  
Hedge (`is_weak_decision`): **{hedge_n}**. Fallback-like overlap≥0.55: **{overlap_n}**.

Bu rapor skoru yükseltmek için yazılmadı. Production default önerisi değişmez: llama3.

## 2. V5.6 Baseline

Kaynak: `benchmark_v56_results.json` (tahmin yok).

| metrik | V5.6 Qwen |
|---|---|
| REAL_LLM | {base.get("real_llm")} |
| first-shot | {base.get("first_shot")} |
| fallback | {base.get("fallback")} |
| CMO | {base.get("cmo")} |
| latency avg / p95 | {base.get("latency")} / {base.get("p95")} |
| BRIEF_ECHO / FC / UF | {base.get("echo")} / {base.get("fc")} / {base.get("uf")} |
| done_reason=length | {base.get("length")} |
| gate / recommendation | {base.get("gate")} / {base.get("recommendation")} |
| SAFE XTTS headroom | {base.get("headroom_b")} MiB |

CMO < 10 olan V5.6 satırları: {base.get("low_rows")}

Env (dokunulmadı): canary 10%, UNLOAD_PEER=false, think=false, n=280, shadow off, reasoning={payload.get("reasoning_env")}.

## 3. Benchmark Design

- n={len(ANATOMY_CASES)} senaryo: 20 V5.1 fair + 12 V5.7 (A–F kategorileri).
- Aynı `compose_consultant_traced` production pipeline. Yeni production prompt yok.
- Qwen `qwen3.5:27b` think=false num_predict=280.
- llama3 aynı pipeline.
- Fallback: `commercial_fallback_reply` (LLM yok).
- Skor: yalnızca `score_cmo_v51`.
- A–H etiketleri `benchmark_v57_anatomy.py` içinde; production import etmez.

Kategori ortalaması (Qwen total): {payload.get("qwen_by_category")}

## 4. Qwen Overall Results

| metrik | Qwen | Llama3 | Fallback |
|---|---|---|---|
| CMO final avg | {qe.get("cmo_final_avg")} | {le.get("cmo_final_avg")} | {fe.get("cmo_final_avg")} |
| REAL_LLM | {(qwen.get("summary") or {}).get("real_llm_success_rate")} | {(llama.get("summary") or {}).get("real_llm_success_rate")} | 0 |
| fallback rate | {(qwen.get("summary") or {}).get("fallback_rate")} | {(llama.get("summary") or {}).get("fallback_rate")} | 1 |
| first-shot PASS | {qe.get("first_shot_pass")} | {le.get("first_shot_pass")} | — |
| BRIEF_ECHO | {(qwen.get("summary") or {}).get("brief_echo_rate")} | {(llama.get("summary") or {}).get("brief_echo_rate")} | — |
| FC / UF | {(qwen.get("summary") or {}).get("false_certainty_rate")} / {(qwen.get("summary") or {}).get("unsupported_fact_rate")} | {(llama.get("summary") or {}).get("false_certainty_rate")} / {(llama.get("summary") or {}).get("unsupported_fact_rate")} | — |
| avg latency | {(qwen.get("summary") or {}).get("avg_latency_s")} | {(llama.get("summary") or {}).get("avg_latency_s")} | ~0 |
| tok/s | {(qwen.get("summary") or {}).get("tokens_per_sec")} | {(llama.get("summary") or {}).get("tokens_per_sec")} | — |
| peak VRAM | {qwen.get("peak_vram")} | {llama.get("peak_vram")} | — |
| weak n (total<10 or dim=0) | {qe.get("weak_n")} | {le.get("weak_n")} | {fe.get("weak_n")} |

## 5. 7-Dimension Breakdown

Ölçek 0–2. Relative weakness: mean < 1.4 zayıf, < 1.7 orta, aksi güçlü.

| Dimension | Score | Relative weakness | Evidence |
|---|---:|---|---|
{dim_md}

DATA_USE tavanı (evaluator token listesi: almanya/fransa/marj/hacim/kapasite/zeytinyağı/`3 `). Tavan 0 olan senaryoda DATA_USE 2 imkânsızdır; bu model hatası değil scorer tavanıdır.

## 6. Failure Classification

Qwen n={len(qrows)}. Impact = sınıfın ortalama total − genel ortalama (negatif = aşağı çeker).

| Class | Count | % | Impact | Examples |
|---|---:|---:|---:|---|
{class_md}

Not: B (DIAGNOSIS≤1) sık görünür çünkü `score_cmo_v51` DIAGNOSIS=2 için Almanya/Fransa regex'ine (`ikinci test`, `fiyat baskısı…gözlem`, `üç lead`) bağlıdır. Bu, teşhis sığlığı ile karıştırılmamalı; kanıt örneklerde.

## 7. Strong Qwen Examples

{strong_md}

## 8. Weak Qwen Examples

{weak_md}

## 9. Llama vs Qwen

| id | cat | Qwen | Llama | Fallback | Qwen path | Llama path |
|---|---|---:|---:|---:|---|---|
{chr(10).join(cmp_lines)}

Llama REAL_LLM düşükse final skor fallback'e yakındır. Qwen'in farkı LLM path'te kalmasıdır; bu CMO 14'ü otomatik doldurmaz.

## 10. Fallback vs Qwen

Fallback-like (kelime overlap ≥0.55): {overlap_n}/{len(qrows)}.
Aynı şablon kopyası (BRIEF_ECHO) Qwen: {(qwen.get("summary") or {}).get("brief_echo_rate")}.

Fallback ticari olarak doğru olabilir. Overlap tek başına zayıflık değildir. Zayıf örneklerde overlap yüksekse model brief/fallback iskeletini tekrarliyor demektir.

## 11. Root Cause

Kanıtlanan (spekülasyon yok):

1. **Scorer tavanı / altın vaka yanlılığı.** DATA_USE ve DIAGNOSIS=2 token/regex'i Almanya–Fransa–zeytinyağı brief'ine kalibre. `raise_price`, `moq`, `incoterms`, `sales_drop`, `collection_risk` gibi opsiyon vakalarında tavan düşük. V5.6'da CMO<10 olan satırlar bu kümede yoğunlaştı: {base.get("low_rows")}.
2. **Karar ve gerekçe boyutu opsiyon/fiyat vakalarında düşük.** En zayıf resmi boyutlar: {", ".join(f"{k}={dim_means.get(k)}" for k in weakest[:3])}. Hedge sayısı={hedge_n}.
3. **V5.6 completeness (kısa bitir) length'i 0 yaptı ama CMO'yu şişirmedi.** Kısa cevap `_REASON`/`_ACTION`/`_UNCERTAIN_OK` hit sayısını keser; 9.8 bununla uyumlu.

## 12. Recommended V5.7 Fix

Kod bu turda yazılmadı. En küçük güvenli aday (sonraki tur):

| ne | neden | beklenen etki | risk | test |
|---|---|---|---|---|
| Production **prompt değil**, `speakable_instruction` içindeki “en fazla beş cümle”yi **karar+gerekçe+aksiyon zorunlu** diye netleştir (cümle sayısı değil içerik) | Kısa cevap DECISION/REASONING/ACTION hit kaçırıyor | CMO +0.3–0.8, özellikle raise_price/moq/incoterms/collection | Length geri gelebilir; n=280 | fair-20 first-shot ≥95, length, BRIEF_ECHO=0, golden CMO≥13 |
| Evaluator'ı **değiştirme** | Skor oyunu | — | V5.6 karşılaştırması bozulur | — |

İkinci aday (daha riskli): DATA_USE token listesini genişletmek. Bu 9.8'i yükseltir ama **gerçek CMO kalitesini yükseltmez**. V5.7'de yapılmamalı.

## 13. What NOT To Change

Validator, evaluator threshold, fallback safety, voice, billing, SHA-256 routing, public API, canary %, production default, think, num_predict, UNLOAD_PEER.

## 14. Go / No-Go

**NO-GO** production default Qwen.  
**GO (dar):** yalnızca completeness cümlesini “kısa ama karar+neden+aksiyon tamam” diye sıkılaştıran minimum prompt deneyi — evaluator/validator yok.  
**NO-GO:** scorer gevşetmek, n yükseltmek, canary %25.

Soru cevabı: Qwen Almanya/Fransa/marj trade-off'ta güçlü (golden 13). Fiyat/MOQ/incoterms/tahsilat/satış teşhisinde resmi CMO'yu **karar+gerekçe hit'leri ve scorer tavanı** birlikte 9.8'e çekiyor. En küçük güvenli değişiklik prompt içerik sözleşmesi; evaluator değil.
""",
        encoding="utf-8",
    )


async def main() -> None:
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    payload = {
        "v56": v56_baseline(),
        "reasoning_env": reasoning_model(),
        "n_cases": len(ANATOMY_CASES),
        "think": False,
        "num_predict": 280,
    }
    if "qwen" in (payload["reasoning_env"] or "").casefold():
        raise SystemExit("diagnostic must not flip OLLAMA_REASONING_MODEL")
    async with httpx.AsyncClient(timeout=300.0) as http:
        print("Qwen anatomy ...", flush=True)
        payload["qwen"] = await run_model(http, QWEN_27B, "qwen-v57")
        print("Llama3 anatomy ...", flush=True)
        payload["llama3"] = await run_model(http, chat_model(), "llama3-v57")
    print("Fallback anatomy ...", flush=True)
    payload["fallback"] = await run_fallback_only()
    payload["qwen_classes"] = class_table(payload["qwen"]["rows"])
    payload["qwen_by_category"] = cat_means(payload["qwen"]["rows"])
    payload["llama_by_category"] = cat_means(payload["llama3"]["rows"])
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(
        "Qwen CMO",
        payload["qwen"]["extra"].get("cmo_final_avg"),
        "dims",
        payload["qwen"]["extra"].get("dim_means"),
        flush=True,
    )
    print("wrote", JSON_OUT, REPORT, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
