"""Türkçe, olgusallık ve ticari güvenilirlik testleri."""

from __future__ import annotations

import unittest

from agents.contracts import current_data_facts, memory_miss_facts
from agents.goals import classify_goal
from agents.intent import classify_intent
from agents.response_engine import _reject_llm, safe_consultant_fallback
from agents.session import SessionState
from agents.tools_base import ToolResult
from agents.validator import english_leak_score, unsourced_stats, validate_response
from prompts import (
    GREETING_REPLY,
    is_current_information,
    is_decision_question,
    is_memory_recall,
    is_stat_challenge,
    wants_web_research,
)


class ReliabilityContractTests(unittest.TestCase):
    def test_hello_stays_short_turkish(self) -> None:
        intent, _ = classify_intent("Merhaba")
        self.assertEqual(intent, "chat")
        self.assertFalse(_reject_llm(GREETING_REPLY))
        leak, _ = english_leak_score(GREETING_REPLY)
        self.assertFalse(leak)

    def test_olive_oil_germany_is_turkish_advisor_route(self) -> None:
        intent, _ = classify_intent("Almanya'ya zeytinyağı satmak istiyorum.")
        self.assertEqual(intent, "trade_advisor")
        text = "Olur. Ama önce distribütör mü, restoran mı, perakende mi netleştirelim."
        verdict = validate_response(text, question="Almanya'ya zeytinyağı satmak istiyorum.")
        self.assertEqual(verdict.status, "PASS")

    def test_2026_growth_does_not_invent_percent(self) -> None:
        q = "Almanya'da zeytinyağı pazarı 2026'da yüzde kaç büyüdü?"
        self.assertTrue(is_stat_challenge(q) or is_current_information(q))
        intent, _ = classify_intent(q)
        self.assertEqual(intent, "trade_advisor")
        invented = "Almanya pazarı 2026'da yüzde 12 büyüdü."
        verdict = validate_response(invented, question=q, facts=current_data_facts())
        self.assertEqual(verdict.status, "REJECT")
        safe = safe_consultant_fallback(
            question=q,
            facts=current_data_facts(),
            tools=[
                ToolResult(
                    name="web_search",
                    ok=False,
                    summary="yok",
                    provenance="TOOL_FAILED",
                )
            ],
        )
        self.assertNotRegex(safe, r"%\s*\d+|yüzde\s*\d+")
        self.assertIn("doğrula", safe.casefold())

    def test_germany_or_france_decision_admits_uncertainty(self) -> None:
        q = "Almanya mı Fransa mı?"
        self.assertTrue(is_decision_question(q))
        intent, _ = classify_intent(q)
        self.assertEqual(intent, "trade_advisor")
        self.assertEqual(classify_goal(q), "strategic_decision")
        safe = safe_consultant_fallback(question=q, facts="")
        self.assertRegex(safe.casefold(), r"veri|net|kilit")

    def test_memory_recall_does_not_bluff(self) -> None:
        q = "Geçen hafta neye karar vermiştik?"
        self.assertTrue(is_memory_recall(q))
        intent, _ = classify_intent(q)
        self.assertEqual(intent, "trade_advisor")
        missed = safe_consultant_fallback(
            question=q,
            facts="",
            tools=[
                ToolResult(
                    name="memory_search",
                    ok=False,
                    summary="yok",
                    provenance="TOOL_FAILED",
                )
            ],
        )
        self.assertEqual(missed, memory_miss_facts())
        remembered = "Almanya'da distribütör modeliyle ilerleme kararı aldık."
        verdict = validate_response(
            "Geçen hafta Almanya'da distribütörle ilerlemeye karar vermiştik.",
            question=q,
            facts=remembered,
            tools=[
                ToolResult(
                    name="memory_search",
                    ok=True,
                    summary=remembered,
                    data={"items": [remembered]},
                    provenance="SOURCE_MEMORY",
                )
            ],
        )
        self.assertEqual(verdict.status, "PASS")

    def test_today_market_news_opens_current_gate(self) -> None:
        q = "Bugün Almanya'daki zeytinyağı piyasasında ne oldu?"
        self.assertTrue(is_current_information(q))
        self.assertTrue(wants_web_research(q))
        intent, _ = classify_intent(q)
        self.assertEqual(intent, "trade_advisor")

    def test_claimed_market_size_is_not_confirmed_without_source(self) -> None:
        q = "123 milyon euro pazar büyüklüğü doğru mu?"
        self.assertTrue(is_stat_challenge(q) or is_decision_question(q))
        intent, _ = classify_intent(q)
        self.assertEqual(intent, "trade_advisor")
        fake = "Evet, Almanya zeytinyağı pazarı 123 milyon euro."
        verdict = validate_response(fake, question=q, facts=current_data_facts())
        self.assertEqual(verdict.status, "REJECT")

    def test_named_firm_decision_is_commercial_not_buyer_dump(self) -> None:
        q = "Şu firmayı müşterimiz yapalım mı?"
        self.assertTrue(is_decision_question(q))
        intent, _ = classify_intent(q)
        self.assertEqual(intent, "trade_advisor")


class LanguageAndStatGuardTests(unittest.TestCase):
    def test_trade_terms_are_allowed_in_turkish(self) -> None:
        text = "B2B ve MOQ netleşmeden FOB fiyatı kilitlemezdim."
        leak, _ = english_leak_score(text)
        self.assertFalse(leak)
        self.assertEqual(validate_response(text).status, "PASS")

    def test_english_paragraph_is_rejected(self) -> None:
        text = (
            "I understand that you want to export olive oil to Germany. "
            "As the director I will provide a strategy for this goal."
        )
        self.assertTrue(_reject_llm(text))
        leak, _ = english_leak_score(text)
        self.assertTrue(leak)

    def test_unsourced_percent_detected(self) -> None:
        hits = unsourced_stats("Pazar yüzde 12 büyüdü.", "doğrulanmış veri yok")
        self.assertTrue(hits)

    def test_sourced_percent_allowed(self) -> None:
        facts = "Kaynak: ithalat yüzde 12 arttı."
        hits = unsourced_stats("Verilere göre ithalat yüzde 12 arttı.", facts)
        self.assertFalse(hits)

    def test_failed_web_success_claim_rejected(self) -> None:
        verdict = validate_response(
            "Internette 17 firma buldum.",
            question="Rakiplerimiz ne yapıyor?",
            tools=[
                ToolResult(
                    name="web_search",
                    ok=False,
                    summary="yok",
                    provenance="TOOL_FAILED",
                )
            ],
        )
        self.assertEqual(verdict.status, "REJECT")

    def test_planning_today_is_not_current_market_gate(self) -> None:
        self.assertFalse(is_current_information("Bugün ne yapmalıyım?"))
        session = SessionState(session_id="p", product="zeytinyağı")
        intent, _ = classify_intent("Bugün ne yapmalıyım?", session=session)
        self.assertEqual(intent, "trade_advisor")


if __name__ == "__main__":
    unittest.main()
