"""V5.11: Qwen REAL_DECISION_MISS — stance before questions. Evaluator/canary/default yok."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import build_commercial_brief
from agents.contracts import CMO_CONTRACT, DECISION_CONTRACT, DIAGNOSTIC_CONTRACT
from agents.evaluator import has_decision_stance, score_cmo_v51
from agents.quality_flags import is_false_certainty
from agents.validator import validate_response
from benchmark_v51 import _session_for
from benchmark_v57_cases import ANATOMY_CASES
from llm import QWEN_NUM_PREDICT, canary_unload_peer, chat_model, reasoning_model, reasoning_think

BACKEND = Path(__file__).resolve().parent
V510_JSON = BACKEND / "benchmark_v510_compare_results.json"
EVALUATOR = BACKEND / "agents" / "evaluator.py"
LLM = BACKEND / "llm.py"
V571_LINE = (
    "Veri eksik olsa bile, kesinlik iddiasında bulunmadan geçici bir ticari duruş ver"
)
STANCE_FIRST = "Kritik metrik (marj, kanal, kapasite, MOQ) yoksa bile önce açık geçici duruş"


def _qwen_dec_zero_ids() -> list[str]:
    data = json.loads(V510_JSON.read_text(encoding="utf-8"))
    rows = ((data.get("qwen") or {}).get("metrics") or {}).get("rows") or []
    out = []
    for row in rows:
        dec = int((row.get("score_final") or {}).get("DECISION") or 0)
        if dec == 0:
            out.append(str(row.get("id") or ""))
    return [item for item in out if item]


def _case_map() -> dict[str, dict]:
    return {str(row["id"]): row for row in ANATOMY_CASES}


class V511StanceBeforeQuestionsTests(unittest.TestCase):
    def test_v571_line_kept_and_trigger_flipped(self) -> None:
        self.assertIn(V571_LINE, CMO_CONTRACT)
        self.assertEqual(CMO_CONTRACT.count("geçici bir ticari duruş"), 1)
        self.assertIn(STANCE_FIRST, CMO_CONTRACT)
        self.assertNotIn("Eksik kritik veri varsa en fazla 1–3 soru sor.", CMO_CONTRACT)
        self.assertIn("soru listesiyle açma", CMO_CONTRACT)
        self.assertIn("Kısmi veride önce açık geçici duruş", DECISION_CONTRACT)
        self.assertIn("Cevabı soruyla açma", DIAGNOSTIC_CONTRACT)

    def test_production_routing_and_canary_untouched(self) -> None:
        os.environ.pop("OLLAMA_REASONING_MODEL", None)
        self.assertEqual(reasoning_model(), chat_model())
        self.assertFalse(reasoning_think())
        self.assertEqual(QWEN_NUM_PREDICT, 280)
        self.assertFalse(canary_unload_peer())
        llm_src = LLM.read_text(encoding="utf-8")
        self.assertIn("sha256", llm_src.casefold())
        self.assertIn("wtr-canary:", llm_src)

    def test_evaluator_not_widened_in_this_pass(self) -> None:
        src = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("yönelmeyin|önceliklendirmeyin", src)
        self.assertIn("bırakmak risk", src)
        self.assertNotIn("kilitlemez", src)

    def test_brief_puts_stance_before_questions(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        brief = build_commercial_brief(q)
        blob = brief.as_prompt()
        stance_at = blob.find("ÖNCE DURUŞ")
        ask_at = blob.find("Duruştan SONRA")
        self.assertGreaterEqual(stance_at, 0)
        self.assertGreaterEqual(ask_at, 0)
        self.assertLess(stance_at, ask_at)
        self.assertTrue(brief.critical_questions)
        self.assertTrue(has_decision_stance(brief.stance))

    def test_unknown_split_still_asks_after_stance(self) -> None:
        q = "Satışlarımız düştü, neden?"
        brief = build_commercial_brief(q)
        self.assertEqual(brief.mode, "diagnostic")
        self.assertTrue(brief.critical_questions)
        self.assertIn("mevcut", brief.critical_questions[0].casefold())
        self.assertTrue(has_decision_stance(brief.stance))
        self.assertLess(
            brief.as_prompt().find("ÖNCE DURUŞ"),
            brief.as_prompt().find("Duruştan SONRA"),
        )

    def test_given_split_still_skips_reask(self) -> None:
        q = "Yeni müşteri gelmiyor ama mevcut müşteriler sipariş veriyor. Nereye bakayım?"
        brief = build_commercial_brief(q)
        self.assertFalse(brief.critical_questions)
        self.assertIn("koru", (brief.stance or "").casefold())
        self.assertTrue(has_decision_stance(brief.stance))

    def test_v510_qwen_dec_zero_briefs_have_scored_stance(self) -> None:
        self.assertTrue(V510_JSON.exists())
        ids = _qwen_dec_zero_ids()
        self.assertEqual(len(ids), 14, ids)
        cases = _case_map()
        weak = []
        for cid in ids:
            case = cases[cid]
            brief = build_commercial_brief(
                case["input"],
                session=_session_for(case),
                matches=case.get("matches"),
            )
            text = brief.stance or ""
            scored = score_cmo_v51(
                text,
                question=case["input"],
                facts=case.get("context") or "",
            )
            if not has_decision_stance(text) or int(scored.decision) < 1:
                weak.append((cid, text, scored.decision))
            self.assertFalse(is_false_certainty(text), cid)
            self.assertNotEqual(
                validate_response(
                    text, question=case["input"], facts=case.get("context") or ""
                ).status,
                "REJECT",
                cid,
            )
        self.assertFalse(weak, weak)

    def test_partial_context_holds_are_not_false_certainty(self) -> None:
        samples = [
            "Marj yokken fiyat artırmayı önermem; kritik veri gelene kadar fiyatı sabit tut.",
            "MOQ'u hemen düşürme. Maliyet netleşene kadar mevcut şartı koru.",
            "Kredi kaydı yokken büyük vadeli siparişi kabul etmem.",
        ]
        for text in samples:
            self.assertTrue(has_decision_stance(text), text)
            self.assertGreaterEqual(score_cmo_v51(text).decision, 1, text)
            self.assertFalse(is_false_certainty(text), text)
            self.assertEqual(validate_response(text, facts="Kayıtlı marj yok.").status, "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
