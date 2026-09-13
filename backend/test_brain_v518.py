"""V5.18: keep last_advisor_kind + filled product on follow-up turns."""

from __future__ import annotations

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.intent import classify_intent
from agents.response_engine import should_reason
from agents.session import SessionState, advisor_answer, apply_utterance_slots
from prompts import advisor_focus, is_buyer_firm_hunt, is_sample_ask, is_stage_followup

REACH_START_A = (
    "Dokuma etiket üretiyorum ve Almanya'ya ihracata başlamak istiyorum. "
    "Müşteri bulmaya nereden başlayayım?"
)
DRAFT_T2 = "İlk müşteriye göndereceğim mesajı hazırla."
REACH_START_B = "Dokuma etiket üretiyorum, Almanya'da müşteri arıyorum."
FOLLOWUP_T2 = "Peki ilk adımım ne olsun?"
REACH_START_C = "Dokuma etiket üretiyorum, Almanya'ya satmak istiyorum."
FIRM_HUNT_T2 = "Almanya'da hangi firmaları aramalıyım?"
REACH_START_D = "Almanya'da müşteri bulmak istiyorum."
LANGUAGE_T2 = "Almanca bilmiyorum."
REACH_START_E = "Almanya'da müşteriye ulaşmak istiyorum."
SAMPLE_T2 = "Numuneyi nasıl göndereyim?"
OLIVE_START = "Zeytinyağı için Almanya'da müşteri arıyorum."
WOVEN_SWITCH = "Dokuma etiket üretiyorum."


def _olive(text: str) -> bool:
    low = (text or "").casefold()
    return "olive" in low or "zeytin" in low or "1509" in low


def _woven_slot(product: str | None) -> bool:
    low = (product or "").casefold()
    return "dokuma" in low or "woven" in low or "etiket" in low


def _germany(market: str | None) -> bool:
    low = (market or "").casefold()
    return "almanya" in low or low in {"de", "germany"}


class V518StageContinuityTests(unittest.TestCase):
    def test_a_reach_then_draft_keeps_product_market(self) -> None:
        session = SessionState(session_id="v518a")
        intent1, _ = classify_intent(REACH_START_A, session)
        self.assertEqual(intent1, "trade_advisor")
        self.assertEqual(advisor_focus(REACH_START_A, session), "reach")
        text1 = advisor_answer(REACH_START_A, session)
        self.assertEqual(session.last_advisor_kind, "reach")
        self.assertTrue(_woven_slot(session.product), session.product)
        self.assertTrue(_germany(session.market), session.market)
        self.assertIn("linkedin", text1.casefold())

        intent2, _ = classify_intent(DRAFT_T2, session)
        self.assertEqual(intent2, "trade_advisor")
        self.assertEqual(advisor_focus(DRAFT_T2, session), "draft")
        text2 = advisor_answer(DRAFT_T2, session)
        self.assertEqual(session.last_advisor_kind, "draft")
        self.assertTrue(_woven_slot(session.product), session.product)
        self.assertTrue(_germany(session.market), session.market)
        self.assertTrue(
            "subject:" in text2.casefold() or "mail" in text2.casefold() or "dear" in text2.casefold(),
            text2,
        )
        self.assertFalse(_olive(text2))

    def test_b_reach_then_generic_followup_keeps_reach(self) -> None:
        session = SessionState(session_id="v518b")
        advisor_answer(REACH_START_B, session)
        self.assertEqual(session.last_advisor_kind, "reach")
        self.assertTrue(is_stage_followup(FOLLOWUP_T2))
        intent2, _ = classify_intent(FOLLOWUP_T2, session)
        self.assertEqual(intent2, "trade_advisor")
        self.assertEqual(advisor_focus(FOLLOWUP_T2, session), "reach")
        self.assertFalse(should_reason(FOLLOWUP_T2))
        text2 = advisor_answer(FOLLOWUP_T2, session)
        self.assertEqual(session.last_advisor_kind, "reach")
        self.assertIn("linkedin", text2.casefold())
        self.assertTrue(_woven_slot(session.product), session.product)
        self.assertTrue(_germany(session.market), session.market)

    def test_c_reach_then_firm_hunt_allows_buyer_finder(self) -> None:
        session = SessionState(session_id="v518c")
        advisor_answer(REACH_START_C, session)
        self.assertTrue(_woven_slot(session.product), session.product)
        self.assertTrue(_germany(session.market), session.market)
        self.assertTrue(is_buyer_firm_hunt(FIRM_HUNT_T2))
        intent2, _ = classify_intent(FIRM_HUNT_T2, session)
        self.assertEqual(intent2, "buyer_finder")
        apply_utterance_slots(session, FIRM_HUNT_T2)
        self.assertTrue(_woven_slot(session.product), session.product)
        self.assertTrue(_germany(session.market), session.market)

    def test_d_reach_then_language_keeps_language_or_reach_bridge(self) -> None:
        session = SessionState(session_id="v518d")
        advisor_answer(REACH_START_D, session)
        intent2, _ = classify_intent(LANGUAGE_T2, session)
        self.assertEqual(intent2, "trade_advisor")
        focus = advisor_focus(LANGUAGE_T2, session)
        text2 = advisor_answer(LANGUAGE_T2, session)
        low = text2.casefold()
        self.assertTrue(
            focus == "language"
            or session.last_advisor_kind in {"language", "reach"},
            focus,
        )
        self.assertTrue(
            "translate" in low
            or "thank you" in low
            or "yabancı dil" in low
            or "almanca" in low
            or "kopyala" in low,
            text2,
        )

    def test_e_reach_then_sample_binds_outreach(self) -> None:
        session = SessionState(session_id="v518e")
        advisor_answer(REACH_START_E, session)
        self.assertTrue(is_sample_ask(SAMPLE_T2))
        intent2, _ = classify_intent(SAMPLE_T2, session)
        self.assertEqual(intent2, "trade_advisor")
        self.assertEqual(advisor_focus(SAMPLE_T2, session), "outreach")
        self.assertFalse(should_reason(SAMPLE_T2))
        text2 = advisor_answer(SAMPLE_T2, session)
        self.assertEqual(session.last_advisor_kind, "outreach")
        low = text2.casefold()
        self.assertTrue("numune" in low or "muster" in low or "sample" in low, text2)
        self.assertTrue("5-7" in text2 or "5–7" in text2 or "mail" in low)

    def test_f_explicit_product_switch_beats_olive_stage(self) -> None:
        session = SessionState(session_id="v518f")
        advisor_answer(OLIVE_START, session)
        apply_utterance_slots(session, WOVEN_SWITCH)
        self.assertTrue(_woven_slot(session.product), session.product)
        self.assertFalse(_olive(session.product or ""))
        self.assertIsNone(session.last_advisor_kind)
        text = advisor_answer(WOVEN_SWITCH, session)
        self.assertFalse(_olive(text), text)
        self.assertTrue(_woven_slot(session.product), session.product)

    def test_buyer_how_find_stays_buyer_finder(self) -> None:
        intent, _ = classify_intent("Müşteriyi nasıl bulurum?")
        self.assertEqual(intent, "buyer_finder")


if __name__ == "__main__":
    unittest.main()
