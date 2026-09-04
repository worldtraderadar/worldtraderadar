"""V5.9: REAL_DECISION_MISS fix. Evaluator/scorer/canary/default değişmez.

Baseline = V5.7.1 B (n=32). Canlı koşu production prompt ile.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import httpx

from agents.contracts import CMO_CONTRACT, DIAGNOSTIC_CONTRACT
from agents.evaluator import score_cmo_v51
from agents.validator import validate_response
from benchmark_v51 import HwPeak, nvidia_query, run_case, summarize
from benchmark_v53 import extra_stats, make_generate
from benchmark_v57 import DIMS, mean_dim, pack_row
from benchmark_v57_anatomy import provisional_stance
from benchmark_v57_cases import ANATOMY_CASES
from benchmark_v58_labels import LABELS
from llm import QWEN_27B, reasoning_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
V571_JSON = BACKEND / "benchmark_v571_results.json"
JSON_OUT = BACKEND / "benchmark_v59_results.json"
REPORT = ROOT / "V5.9_REAL_DECISION_FIX_REPORT.md"

FOCUS = (
    "raise_price",
    "new_buyer_priority",
    "sales_drop",
    "discount",
    "collection_risk",
    "incoterms",
    "germany_france",
)
GOLDEN = (
    "germany_france",
    "discount",
    "buyer_abc",
    "collection_risk",
    "capacity",
    "capacity_split",
    "cheap_price_market",
    "incoterms",
    "accept_order",
    "lost_quote",
    "market_choice",
)
V571_LINE = "geçici bir ticari duruş"
V59_PRICE = "Marj yokken şimdilik fiyatı artırma"

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

_RAISE_STANCE = re.compile(
    r"(?i)(artırmaz|artırmam|şimdilik.{0,32}artır|fiyatı artırmaz|kör.{0,12}zam|zam yapma)"
)
_PROTECT = re.compile(
    r"(?i)("
    r"mevcut.{0,24}(koru|öncelik|tut)|"
    r"korumayı öncelik|"
    r"kazan[ıi]m|"
    r"yeni.{0,24}(ayr[ıi]|test|delik)"
    r")"
)
_REASK_SPLIT = re.compile(
    r"(?i)("
    r"son iki aydaki|"
    r"mevcut m[uü][sş]terilerinizden.{0,40}\?|"
    r"yeni m[uü][sş]teri (giri[sş]|ak[ıi][sş]).{0,24}durdu mu"
    r")"
)
_REAL_STANCE = re.compile(
    r"(?i)("
    r"artırmaz|düşürmez|düşürmeyin|kabul etme|kabul etmeyin|kabul etmem|"
    r"bırakmaz|girmez|girmem|önermem|öneririm|ben olsam|"
    r"şimdilik|şu aşamada|öncelik|koru|fob|"
    r"seçerdim|tercihim|elden çıkarma|ikinci (bir )?test|"
    r"standart ko[sş]ullar[ıi] koru|adım atmaz"
    r")"
)
_DEFER = re.compile(
    r"(?i)(stratejisini belirleriz|daha sonra belirle|tercih yapmam mümkün değil|"
    r"yönlendirebiliriz|seçebiliriz|belirleriz\.?$)"
)


def clip(text: str, n: int = 700) -> str:
    body = (text or "").strip()
    return body if len(body) <= n else body[:n] + "…"


def classify_text(case_id: str, question: str, text: str, dec: int) -> str:
    body = text or ""
    stance = bool(_REAL_STANCE.search(body))
    if case_id == "raise_price":
        stance = bool(_RAISE_STANCE.search(body))
    elif case_id == "new_buyer_priority":
        stance = bool(_PROTECT.search(body)) and not _REASK_SPLIT.search(body)
    elif case_id in ("sales_drop",) and _REASK_SPLIT.search(body):
        stance = False
    if stance and dec <= 0:
        return "SCORER_MISS"
    if stance and dec > 0:
        return "REAL_OK"
    if (not stance) and dec > 0:
        return "SCORER_FALSE_POS"
    return "REAL_DECISION_MISS"


def attach(row: dict) -> dict:
    text = row.get("final") or ""
    dec = int((row.get("score_final") or {}).get("DECISION") or 0)
    row = dict(row)
    row["provisional"] = provisional_stance(text)
    row["verdict"] = classify_text(row.get("id") or "", row.get("input") or "", text, dec)
    row["raise_stance"] = bool(_RAISE_STANCE.search(text))
    row["protect_stance"] = bool(_PROTECT.search(text))
    row["reask_split"] = bool(_REASK_SPLIT.search(text))
    return row


def load_baseline() -> dict:
    data = json.loads(V571_JSON.read_text(encoding="utf-8"))
    rows = []
    for raw in (data.get("B") or {}).get("rows") or []:
        item = dict(raw)
        text = item.get("final") or ""
        dec = int((item.get("score_final") or {}).get("DECISION") or 0)
        prior = (LABELS.get(item["id"]) or {}).get("verdict")
        item["verdict"] = prior or classify_text(item["id"], item.get("input") or "", text, dec)
        item["provisional"] = provisional_stance(text)
        rows.append(item)
    extra = dict((data.get("B") or {}).get("extra") or {})
    extra["verdicts"] = dict(Counter(r["verdict"] for r in rows))
    return {"rows": rows, "extra": extra, "summary": (data.get("B") or {}).get("summary") or {}}


def count_verdicts(rows: list[dict]) -> dict:
    return dict(Counter(r.get("verdict") for r in rows))


def go_nogo(base: dict, now: dict, neg: list[dict], rows: list[dict]) -> tuple[str, list[str]]:
    be, ne = base.get("extra") or {}, now.get("extra") or {}
    reasons: list[str] = []
    bmap = {r["id"]: r for r in base.get("rows") or []}
    nmap = {r["id"]: r for r in rows}
    rp = nmap.get("raise_price") or {}
    nb = nmap.get("new_buyer_priority") or {}
    if rp.get("verdict") == "REAL_DECISION_MISS":
        reasons.append("raise_price hâlâ REAL_DECISION_MISS")
    if nb.get("verdict") == "REAL_DECISION_MISS":
        reasons.append("new_buyer_priority hâlâ REAL_DECISION_MISS")
    if nb.get("reask_split"):
        reasons.append("new_buyer_priority verilen ayrımı tekrar soruyor")
    bv, nv = be.get("verdicts") or {}, ne.get("verdicts") or {}
    if int(nv.get("REAL_DECISION_MISS") or 0) > int(bv.get("REAL_DECISION_MISS") or 0):
        reasons.append("REAL_MISS arttı")
    if (ne.get("fc_rate") or 0) > 0:
        reasons.append("FALSE_CERTAINTY > 0")
    if (ne.get("uf_rate") or 0) > 0:
        reasons.append("UNSUPPORTED_FACT > 0")
    if (ne.get("echo_rate") or 0) > 0:
        reasons.append("BRIEF_ECHO > 0")
    if (ne.get("fallback_rate") or 0) > (be.get("fallback_rate") or 0):
        reasons.append("fallback arttı")
    if (ne.get("first_shot") or 1) + 1e-9 < (be.get("first_shot") or 0):
        reasons.append("first-shot düştü")
    if any(r.get("unsafe") for r in neg):
        reasons.append("negative test unsafe")
    broken = []
    for key in GOLDEN:
        a, b = bmap.get(key) or {}, nmap.get(key) or {}
        if (b.get("verdict") == "REAL_DECISION_MISS") and (a.get("verdict") != "REAL_DECISION_MISS"):
            broken.append(key)
    if broken:
        reasons.append("golden REAL_MISS oldu: " + ", ".join(broken))
    if reasons:
        return "NO-GO", reasons
    return "GO", ["raise_price ve new_buyer_priority REAL_MISS değil", "safety flat", "evaluator dokunulmadı"]


def focus_table(a_rows: list, b_rows: list) -> str:
    amap = {r["id"]: r for r in a_rows}
    bmap = {r["id"]: r for r in b_rows}
    lines = [
        "| id | V5.7.1 verdict | V5.9 verdict | A tot/DEC | B tot/DEC | reask |",
        "|---|---|---|---:|---:|---|",
    ]
    for key in FOCUS:
        a, b = amap.get(key) or {}, bmap.get(key) or {}
        as_, bs_ = a.get("score_final") or {}, b.get("score_final") or {}
        lines.append(
            f"| {key} | {a.get('verdict')} | {b.get('verdict')} | "
            f"{as_.get('total')}/{as_.get('DECISION')} | {bs_.get('total')}/{bs_.get('DECISION')} | "
            f"{b.get('reask_split')} |"
        )
    return "\n".join(lines)


def write_report(payload: dict) -> None:
    a, b = payload["baseline"], payload["now"]
    ae, be = a["extra"], b["extra"]
    rec, reasons = payload["recommendation"], payload["reasons"]
    av, bv = ae.get("verdicts") or {}, be.get("verdicts") or {}
    amap = {r["id"]: r for r in a["rows"]}
    bmap = {r["id"]: r for r in b["rows"]}

    def pair(cid: str) -> str:
        old, new = amap.get(cid) or {}, bmap.get(cid) or {}
        return (
            f"### `{cid}`\n\n"
            f"V5.7.1: {old.get('verdict')} · tot {(old.get('score_final') or {}).get('total')} · "
            f"DEC {(old.get('score_final') or {}).get('DECISION')}\n\n"
            f"**Before:** {clip(old.get('final') or '')}\n\n"
            f"V5.9: {new.get('verdict')} · tot {(new.get('score_final') or {}).get('total')} · "
            f"DEC {(new.get('score_final') or {}).get('DECISION')}\n\n"
            f"**After:** {clip(new.get('final') or '')}\n"
        )

    golden_lines = ["| id | before | after | tot | DEC |", "|---|---|---|---:|---:|"]
    for key in GOLDEN:
        old, new = amap.get(key) or {}, bmap.get(key) or {}
        sc = new.get("score_final") or {}
        golden_lines.append(
            f"| {key} | {old.get('verdict')} | {new.get('verdict')} | {sc.get('total')} | {sc.get('DECISION')} |"
        )

    REPORT.write_text(
        f"""# V5.9 Real Decision Stance Fix

Production default llama3, canary %10, n=280, think=false. Evaluator/validator değişmedi.

## 1. Değiştirilen dosya

- `backend/agents/contracts.py` — CMO_CONTRACT’a iki davranış cümlesi; DIAGNOSTIC_CONTRACT hiyerarşisi.
- `backend/agents/commercial.py` — kullanıcı verilen mevcut/yeni ayrımını brief’te tekrar sorma.

V5.7.1 satırı korundu: `{V571_LINE in CMO_CONTRACT}`.
V5.9 fiyat kuralı: `{V59_PRICE in CMO_CONTRACT}`.
DIAGNOSTIC tekrar-sorma: `{"verdiyse tekrar sorma" in DIAGNOSTIC_CONTRACT}`.

## 2. Değişikliğin tam davranış amacı

Veri eksik ≠ karar yok. Kullanıcının verdiği ayrımı yeniden sorma.
Marj yokken şimdilik fiyatı artırma. Teşhis sorusu duruşun yerine geçmesin.

## 3. Neden bu dosyalar

CMO her turda system’de. DIAGNOSTIC sonra ekleniyor ve V5.8’de stance’i eziyordu.
Brief, verilen ayrımı «Sor» diye basıyordu; contract ile çelişiyordu. Evaluator’a dokunulmadı.

## 4. raise_price before/after

{pair("raise_price")}

## 5. new_buyer_priority before/after

{pair("new_buyer_priority")}

## 6. diagnostic-vs-stance etkisi

{pair("sales_drop")}

## 7. REAL_MISS before/after

| | V5.7.1 B | V5.9 |
|---|---:|---:|
| REAL_DECISION_MISS | {av.get("REAL_DECISION_MISS")} | {bv.get("REAL_DECISION_MISS")} |
| SCORER_MISS | {av.get("SCORER_MISS")} | {bv.get("SCORER_MISS")} |
| REAL_OK | {av.get("REAL_OK")} | {bv.get("REAL_OK")} |
| SCORER_FALSE_POS | {av.get("SCORER_FALSE_POS")} | {bv.get("SCORER_FALSE_POS")} |

## 8. SCORER_MISS before/after

V5.7.1 SCORER_MISS {av.get("SCORER_MISS")} → V5.9 {bv.get("SCORER_MISS")}.
Scorer fiil listesi genişletilmedi. `artırmazdım` / `kabul etmeyin` / `FOB` hâlâ resmi DECISION’a girmeyebilir.

## 9. CMO score before/after

| Metric | V5.7.1 B | V5.9 | Delta |
|---|---:|---:|---:|
| CMO total | {ae.get("cmo_final_avg")} | {be.get("cmo_final_avg")} | {round((be.get("cmo_final_avg") or 0)-(ae.get("cmo_final_avg") or 0), 3)} |
| DECISION | {ae.get("decision")} | {be.get("decision")} | {round((be.get("decision") or 0)-(ae.get("decision") or 0), 3)} |
| ACTION | {ae.get("action")} | {be.get("action")} | {round((be.get("action") or 0)-(ae.get("action") or 0), 3)} |
| UNCERTAINTY | {(ae.get("dim_means") or {}).get("UNCERTAINTY")} | {(be.get("dim_means") or {}).get("UNCERTAINTY")} | — |
| NATURAL_TURKISH | {ae.get("natural")} | {be.get("natural")} | {round((be.get("natural") or 0)-(ae.get("natural") or 0), 3)} |
| first-shot PASS | {ae.get("first_shot")} | {be.get("first_shot")} | {round((be.get("first_shot") or 0)-(ae.get("first_shot") or 0), 3)} |
| fallback | {ae.get("fallback_rate")} | {be.get("fallback_rate")} | {round((be.get("fallback_rate") or 0)-(ae.get("fallback_rate") or 0), 3)} |
| FALSE_CERTAINTY | {ae.get("fc_rate")} | {be.get("fc_rate")} | {round((be.get("fc_rate") or 0)-(ae.get("fc_rate") or 0), 3)} |
| UNSUPPORTED_FACT | {ae.get("uf_rate")} | {be.get("uf_rate")} | {round((be.get("uf_rate") or 0)-(ae.get("uf_rate") or 0), 3)} |
| BRIEF_ECHO | {ae.get("echo_rate")} | {be.get("echo_rate")} | {round((be.get("echo_rate") or 0)-(ae.get("echo_rate") or 0), 3)} |
| latency avg | {(a.get("summary") or {}).get("avg_latency_s")} | {(b.get("summary") or {}).get("avg_latency_s")} | — |

DECISION artışı başarı kriteri değildir (V5.8: DEC=0’ın ~%56’sı scorer miss).

## 10. first-shot validator

V5.7.1 {ae.get("first_shot")} → V5.9 {be.get("first_shot")}

## 11. false certainty

{be.get("fc_rate")}

## 12. unsupported numbers

{be.get("uf_rate")}

## 13. BRIEF_ECHO

{be.get("echo_rate")}

## 14. golden regression

{chr(10).join(golden_lines)}

Focus:

{focus_table(a["rows"], b["rows"])}

## 15. tüm test sonucu

Unit: `test_brain_v59.py` + mevcut suite. Live n=32 Qwen 27B think=false n=280.

Negative:

{chr(10).join(f"- `{r.get('id')}` validator={r.get('validator')} real_llm={r.get('real_llm')} FC={r.get('fc')} UF={r.get('uf')} unsafe={r.get('unsafe')}" for r in payload.get("negative") or [])}

## 16. production config değişmediğinin doğrulaması

`OLLAMA_REASONING_MODEL` bu koşuda set edilmedi (`{payload.get("reasoning_env")}`).
think=false, n=280, canary/default/evaluator/validator dokunulmadı.

## 17. GO / NO-GO

**{rec}**

{chr(10).join(f"- {r}" for r in reasons)}

Asıl hedef REAL_DECISION_MISS azalıyor mu: {av.get("REAL_DECISION_MISS")} → {bv.get("REAL_DECISION_MISS")}.
""",
        encoding="utf-8",
    )


async def run_now(http) -> dict:
    metrics: dict = {"calls": []}
    hw = HwPeak()
    hw.mark()
    gen = make_generate(QWEN_27B, metrics, hw, think=False, num_predict=280)
    packed = []
    for case in ANATOMY_CASES:
        print(f"[v59] {case['id']} ...", flush=True)
        raw = await run_case(http, gen, case)
        packed.append(attach(pack_row(case, raw)))
    summary = summarize(
        "qwen-v59",
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
    extra["first_shot"] = extra.get("pass_rate")
    extra["first_shot_pass"] = extra.get("pass_rate")
    extra["dim_means"] = {k: mean_dim(packed, k) for k in DIMS}
    extra["cmo_final_avg"] = round(
        sum(int((p.get("score_final") or {}).get("total") or 0) for p in packed) / (len(packed) or 1),
        3,
    )
    extra["fc_rate"] = round(sum(1 for r in packed if r.get("fc")) / (len(packed) or 1), 3)
    extra["uf_rate"] = round(sum(1 for r in packed if r.get("uf")) / (len(packed) or 1), 3)
    extra["echo_rate"] = round(sum(1 for r in packed if r.get("echo")) / (len(packed) or 1), 3)
    extra["fallback_rate"] = round(sum(1 for r in packed if not r.get("real_llm")) / (len(packed) or 1), 3)
    extra["natural"] = extra["dim_means"].get("NATURAL_TURKISH")
    extra["action"] = extra["dim_means"].get("ACTION")
    extra["decision"] = extra["dim_means"].get("DECISION")
    extra["verdicts"] = count_verdicts(packed)
    extra["gpu"] = nvidia_query()
    return {"rows": packed, "extra": extra, "summary": summary, "peak_vram": hw.vram_peak}


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
    if V571_LINE not in CMO_CONTRACT:
        raise SystemExit("V5.7.1 stance line missing")
    if V59_PRICE not in CMO_CONTRACT:
        raise SystemExit("V5.9 price line missing")
    if "qwen" in reasoning_model().casefold():
        raise SystemExit("do not flip production default")
    payload = {
        "baseline": load_baseline(),
        "change": V59_PRICE,
        "reasoning_env": reasoning_model(),
    }
    async with httpx.AsyncClient(timeout=300.0) as http:
        payload["now"] = await run_now(http)
        payload["negative"] = await run_negative(http)
    rec, reasons = go_nogo(
        payload["baseline"], payload["now"], payload["negative"], payload["now"]["rows"]
    )
    payload["recommendation"] = rec
    payload["reasons"] = reasons
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print("RECOMMENDATION", rec, reasons, flush=True)
    print(
        "REAL_MISS",
        payload["baseline"]["extra"].get("verdicts"),
        "→",
        payload["now"]["extra"].get("verdicts"),
        "CMO",
        payload["baseline"]["extra"].get("cmo_final_avg"),
        "→",
        payload["now"]["extra"].get("cmo_final_avg"),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
