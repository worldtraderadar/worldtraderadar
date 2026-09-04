"""AI danışman beyni: intent, bellek, araç ve persona testleri."""

from __future__ import annotations

import unittest

from agents.goals import classify_goal
from agents.intent import classify_intent
from agents.memory_store import should_persist
from agents.documents import analyze_documents, diagnostic_outline
from agents.response_engine import should_reason
from agents.session import SessionState
from agents.web_search import web_search_configured
from prompts import (
    CORE_PERSONA,
    GREETING_REPLY,
    VOICE_SYSTEM,
    compose_system_prompt,
    is_diagnostic_request,
    is_document_ask,
    is_memory_recall,
    is_planning_ask,
    wants_web_research,
)


class BrainRoutingTests(unittest.TestCase):
    def test_hello_is_chat(self) -> None:
        intent, _ = classify_intent("Merhaba")
        self.assertEqual(intent, "chat")

    def test_today_plan_uses_context_when_product_known(self) -> None:
        session = SessionState(session_id="plan", product="zeytinyağı", market="Almanya")
        self.assertTrue(is_planning_ask("Bugün ne yapmalıyım?"))
        intent, _ = classify_intent("Bugün ne yapmalıyım?", session=session)
        self.assertEqual(intent, "trade_advisor")
        self.assertEqual(classify_goal("Bugün ne yapmalıyım?", session), "planning")

    def test_olive_oil_germany_is_consultant_not_intake(self) -> None:
        intent, _ = classify_intent("Almanya'ya zeytinyağı satmak istiyorum.")
        self.assertEqual(intent, "trade_advisor")
        self.assertEqual(
            classify_goal("Almanya'ya zeytinyağı satmak istiyorum."),
            "export_strategy",
        )

    def test_hs_buyer_search(self) -> None:
        intent, _ = classify_intent("1509 HS koduyla Almanya'da müşteri bul.")
        self.assertEqual(intent, "buyer_finder")

    def test_memory_recall_routes_advisor(self) -> None:
        self.assertTrue(is_memory_recall("Geçen hafta ne konuşmuştuk?"))
        intent, _ = classify_intent("Geçen hafta ne konuşmuştuk?")
        self.assertEqual(intent, "trade_advisor")
        self.assertEqual(classify_goal("Geçen hafta ne konuşmuştuk?"), "memory_recall")

    def test_sales_drop_is_diagnostic(self) -> None:
        self.assertTrue(is_diagnostic_request("Satışlarımız düştü."))
        intent, _ = classify_intent("Satışlarımız düştü.")
        self.assertEqual(intent, "trade_advisor")
        outline = diagnostic_outline("Satışlarımız düştü.")
        self.assertIn("trafik", outline.casefold())
        self.assertIn("dönüşüm", outline.casefold())

    def test_competitors_want_web_research(self) -> None:
        self.assertTrue(wants_web_research("Rakiplerimiz ne yapıyor?"))
        intent, _ = classify_intent("Rakiplerimiz ne yapıyor?")
        self.assertEqual(intent, "trade_advisor")
        self.assertFalse(wants_web_research("İhracat nedir?"))

    def test_excel_without_file_is_honest(self) -> None:
        self.assertTrue(is_document_ask("Bu Excel'i analiz et."))
        result = analyze_documents(file_names=[], question="Bu Excel'i analiz et.")
        self.assertFalse(result.ok)
        self.assertIn("yok", result.error.casefold() if result.error else result.summary.casefold())


class BrainSafetyTests(unittest.TestCase):
    def test_persona_is_composed(self) -> None:
        self.assertIn("ticari danışman", CORE_PERSONA.casefold())
        self.assertIn("ticari danışman", VOICE_SYSTEM.casefold())
        block = compose_system_prompt(CORE_PERSONA, "Görev: test")
        self.assertIn("Görev: test", block)

    def test_mail_draft_skips_llm_rewrite(self) -> None:
        self.assertFalse(should_reason("bana mail taslağı hazırlarsın"))

    def test_memory_does_not_store_small_talk(self) -> None:
        self.assertFalse(should_persist("Merhaba"))
        self.assertFalse(should_persist("Bugün nasılsın?"))
        self.assertTrue(
            should_persist("Şirketimiz ağırlıklı olarak Avrupa'ya zeytinyağı ihraç ediyor.")
        )
        self.assertTrue(
            should_persist("Almanya'da distribütör modeliyle ilerleme kararı aldık.")
        )

    def test_web_search_not_faked_when_unconfigured(self) -> None:
        if web_search_configured():
            self.skipTest("web anahtarı tanımlı")
        self.assertFalse(web_search_configured())

    def test_english_llm_output_is_rejected(self) -> None:
        from agents.response_engine import _reject_llm

        self.assertTrue(_reject_llm("I understand that you want to export olive oil."))
        self.assertFalse(_reject_llm("Olur. Ama önce hedef segmenti netleştirelim."))


if __name__ == "__main__":
    unittest.main()
