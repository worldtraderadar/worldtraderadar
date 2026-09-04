"""V5.1 regression: brief echo, memory honesty, competitor fallback, inventory."""

from __future__ import annotations

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import (
    build_commercial_brief,
    commercial_fallback_reply,
    is_reasoning_task,
    prioritize_matches,
)
from agents.evaluator import score_cmo_v51
from agents.quality import ComposeTrace
from agents.quality_flags import is_brief_echo, is_memory_dishonest
from agents.response_engine import safe_consultant_fallback
from agents.validator import validate_response
from benchmark_v51_cases import BUYER_CARDS, CASES, FAIR_FACTS
from llm import chat_model, model_for_task


class BriefEchoAndMemoryTests(unittest.TestCase):
    def test_brief_echo_is_rejected(self) -> None:
        leaked = (
            "TİCARİ ZEKÂ BRİFİ (kopyalama, yorumla):\n"
            "Mod: decision\n"
            "BELLEK BOŞ: hatırlıyorum / geçen sefer deme.\n"
            "Doğal Konuş"
        )
        self.assertTrue(is_brief_echo(leaked))
        self.assertEqual(validate_response(leaked).status, "REJECT")

    def test_memory_dishonesty_without_memory(self) -> None:
        text = "Hatırlıyorum, geçen konuşmada Fransa'ya geçmiştik."
        self.assertTrue(is_memory_dishonest(text, memory_present=False))
        self.assertEqual(
            validate_response(text, facts="BELLEK BOŞ: hatırlıyorum deme.").status,
            "REJECT",
        )

    def test_memory_ok_when_present(self) -> None:
        text = "Geçen tur Almanya'da distribütör kararı kayıtlı; onu bırakmam."
        verdict = validate_response(
            text,
            facts="BELLEK (yalnızca kayıtlı): Almanya distribütör",
            memories=["Almanya'da distribütör modeliyle ilerleme kararı aldık."],
        )
        self.assertNotEqual(verdict.status, "REJECT")


class CompetitorFallbackRegressionTests(unittest.TestCase):
    def test_competitor_percent_is_not_market_stats(self) -> None:
        q = "Rakibim fiyatı benden %20 düşük."
        brief = build_commercial_brief(q)
        reply = commercial_fallback_reply(brief, q) or safe_consultant_fallback(
            question=q, facts=""
        )
        low = reply.casefold()
        self.assertNotIn("güncel pazar", low)
        self.assertTrue("iddia" in low or "aktardığın" in low or "fact" in low)
        self.assertTrue(is_reasoning_task(q))

    def test_stat_challenge_still_blocks_invented_market_size(self) -> None:
        q = "Almanya'da zeytinyağı pazarı 2026'da yüzde kaç büyüdü?"
        reply = safe_consultant_fallback(question=q, facts="")
        self.assertIn("doğrulanmış değil", reply.casefold())


class ScoreAndInventoryTests(unittest.TestCase):
    def test_v51_score_total_is_14_scale(self) -> None:
        q = "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
        reply = commercial_fallback_reply(build_commercial_brief(q), q) or ""
        score = score_cmo_v51(reply, question=q, facts=FAIR_FACTS)
        self.assertLessEqual(score.total, 14)
        self.assertFalse(score.brief_echo)
        self.assertGreaterEqual(score.total, 8)

    def test_twenty_reasoning_cases_exist(self) -> None:
        self.assertGreaterEqual(len(CASES), 20)
        ids = [row["id"] for row in CASES]
        self.assertIn("germany_france", ids)
        self.assertIn("competitor_pressure", ids)

    def test_buyer_high_score_alone_is_not_win_prob(self) -> None:
        ranked = prioritize_matches(BUYER_CARDS)
        generic = next(row for row in ranked if row.name == "Generic Shop")
        self.assertEqual(generic.band, "C")
        silentish = next(row for row in ranked if row.name == "Retail Mix GmbH")
        self.assertNotEqual(silentish.band, "A")

    def test_public_path_alias(self) -> None:
        row = ComposeTrace(text="x", path="LLM_CMO_SUCCESS")
        self.assertEqual(row.public_path, "LLM_DIRECT")
        self.assertEqual(
            ComposeTrace(text="x", path="REJECTED_NO_SAFE_RESPONSE").public_path,
            "SAFE_FALLBACK",
        )

    def test_model_routing_unchanged_default(self) -> None:
        self.assertEqual(model_for_task("decision"), chat_model())
        self.assertEqual(model_for_task("chat"), chat_model())


if __name__ == "__main__":
    unittest.main(verbosity=2)
