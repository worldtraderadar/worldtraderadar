"""V5.18.1: product-first advising + topic discipline."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.intent import classify_intent
from agents.retrieve import active_retrieval_product, retrieval_query
from agents.session import (
    SessionState,
    advisor_answer,
    apply_utterance_slots,
    stale_session_product_conflict,
)
from agents.validator import make_speakable
from prompts import advisor_focus, is_incoterm_ask, is_payment_ask, is_ratio_ask

BACKEND = Path(__file__).resolve().parent

GTM_HARD = (
    "Dokuma etiket üretiyorum ve Almanya'ya ihracata başlamak istiyorum. "
    "Müşteriyi nereden bulacağımı ve ilk müşteriye nasıl ulaşacağımı bilmiyorum."
)
GTM_SELL = "Dokuma etiket üretiyorum ve Almanya'ya satmak istiyorum."
FIND_T2 = "Almanya'da ilk müşteriyi nereden bulabilirim?"
REACH_T3 = "İlk müşteriye nasıl ulaşacağım?"
DRAFT_T4 = "İlk müşteriye göndereceğim mesajı hazırlar mısın?"
SAMPLE_T5 = "Numuneyi nasıl göndermeliyim?"
PAY_T6 = "Ödemeyi nasıl almalıyım?"
INCOTERM_T7 = "EXW ile DAP arasındaki fark nedir?"
RATIO_T8 = "Yüzde kaç peşin istemeliyim?"
FIRM_T9 = "Almanya'da hangi firmalara yazmalıyım?"
DRAFT_T10 = "Bu firmalara ilk mesajı da hazırlar mısın?"
FOLLOWUP = "Peki ilk adımım ne olsun?"
LANGUAGE_T2 = "Almanca bilmiyorum."
OLIVE_Q = "Zeytinyağı için Almanya'da müşteri arıyorum."
WOVEN_SWITCH = "Dokuma etiket üretiyorum."

_RATIO_LEAK = re.compile(r"%\s*\d+|\d+\s*/\s*\d+|yüzde\s*\d+", re.IGNORECASE)
_PAY_LEAK = re.compile(
    r"(?i)(pe[sş]in|vadeli|akreditif|letter of credit|[oö]deme\s*(plan|[sş]ekil|risk))"
)
_TERM_LEAK = re.compile(
    r"(?i)(\bexw\b|\bdap\b|\bfca\b|incoterm|gt[iı]p|\bhs\b|g[uü]mr[uü]k\s*beyanname)"
)


def _olive(text: str) -> bool:
    low = (text or "").casefold()
    return "olive" in low or "zeytin" in low or "1509" in low


def _woven(product: str | None) -> bool:
    low = (product or "").casefold()
    return "dokuma" in low or "woven" in low or "etiket" in low


def _germany(market: str | None) -> bool:
    low = (market or "").casefold()
    return "almanya" in low or low in {"de", "germany"}


def _buyer_tokens(text: str) -> bool:
    low = (text or "").casefold()
    return any(
        token in low
        for token in (
            "giyim",
            "konfeksiyon",
            "marka",
            "sourcing",
            "satın alma",
            "etiket",
        )
    )


def _sales_path(text: str) -> bool:
    low = (text or "").casefold()
    return "linkedin" in low and (
        "texworld" in low or "texprocess" in low or "fuar" in low or "sourcing" in low
    )


class V5181ProductFirstTests(unittest.TestCase):
    def test_a_product_first_gtm_hard(self) -> None:
        session = SessionState(session_id="v5181a")
        self.assertEqual(advisor_focus(GTM_HARD, session), "reach")
        text = advisor_answer(GTM_HARD, session)
        self.assertTrue(_buyer_tokens(text), text)
        self.assertIn("almanya", text.casefold())
        self.assertIsNone(_RATIO_LEAK.search(text), text)
        self.assertIsNone(_PAY_LEAK.search(text), text)
        self.assertIsNone(_TERM_LEAK.search(text), text)
        self.assertNotIn("1509", text)

    def test_b_product_specific_customers(self) -> None:
        session = SessionState(session_id="v5181b", product="dokuma etiket", market="Almanya")
        text = advisor_answer(GTM_SELL, session)
        self.assertTrue("giyim" in text.casefold() or "konfeksiyon" in text.casefold(), text)
        self.assertIn("sourcing", text.casefold())

    def test_c_product_market_sales_path(self) -> None:
        session = SessionState(session_id="v5181c")
        text = advisor_answer(GTM_SELL, session)
        self.assertTrue(_woven(session.product), session.product)
        self.assertTrue(_germany(session.market), session.market)
        self.assertTrue(_sales_path(text), text)

    def test_d_no_unsolicited_percentages(self) -> None:
        session = SessionState(session_id="v5181d")
        for question in (GTM_HARD, FIND_T2, REACH_T3):
            if "dokuma" not in question.casefold():
                advisor_answer(GTM_SELL, session)
            text = advisor_answer(question, session)
            self.assertIsNone(_RATIO_LEAK.search(text), question + "\n" + text)

    def test_e_no_unsolicited_payment(self) -> None:
        session = SessionState(session_id="v5181e")
        text = advisor_answer(FIND_T2, SessionState(session_id="v5181e2", product="dokuma etiket"))
        self.assertIsNone(_PAY_LEAK.search(text), text)
        advisor_answer(GTM_SELL, session)
        text2 = advisor_answer(REACH_T3, session)
        self.assertEqual(session.last_advisor_kind, "reach")
        self.assertIsNone(_PAY_LEAK.search(text2), text2)

    def test_f_no_unsolicited_technical_terms(self) -> None:
        session = SessionState(session_id="v5181f")
        text = advisor_answer(GTM_HARD, session)
        self.assertIsNone(_TERM_LEAK.search(text), text)

    def test_g_explicit_payment_still_works(self) -> None:
        session = SessionState(session_id="v5181g", product="dokuma etiket")
        advisor_answer(GTM_SELL, session)
        self.assertTrue(is_payment_ask(PAY_T6))
        self.assertEqual(advisor_focus(PAY_T6, session), "mediate")
        intent, _ = classify_intent(PAY_T6, session)
        self.assertEqual(intent, "trade_advisor")
        text = advisor_answer(PAY_T6, session)
        self.assertTrue(
            "peşin" in text.casefold() or "ödeme" in text.casefold() or "%" in text,
            text,
        )

    def test_h_explicit_incoterm_still_works(self) -> None:
        self.assertTrue(is_incoterm_ask(INCOTERM_T7))
        self.assertEqual(advisor_focus(INCOTERM_T7), "docs")
        text = advisor_answer(INCOTERM_T7)
        low = text.casefold()
        self.assertIn("exw", low)
        self.assertTrue("dap" in low or "ex works" in low)
        self.assertIn("free carrier", low)

    def test_i_explicit_percentage_still_works(self) -> None:
        session = SessionState(session_id="v5181i", product="dokuma etiket")
        self.assertTrue(is_ratio_ask(RATIO_T8))
        self.assertTrue(is_payment_ask(RATIO_T8))
        text = advisor_answer(RATIO_T8, session)
        self.assertIsNotNone(_RATIO_LEAK.search(text) or re.search(r"%", text), text)

    def test_j_reach_then_draft(self) -> None:
        session = SessionState(session_id="v5181j")
        advisor_answer(GTM_SELL, session)
        self.assertEqual(session.last_advisor_kind, "reach")
        intent, _ = classify_intent(DRAFT_T4, session)
        self.assertEqual(intent, "trade_advisor")
        self.assertEqual(advisor_focus(DRAFT_T4, session), "draft")
        text = advisor_answer(DRAFT_T4, session)
        self.assertEqual(session.last_advisor_kind, "draft")
        self.assertTrue("subject:" in text.casefold() or "dear" in text.casefold(), text)
        self.assertTrue(_woven(session.product), session.product)

    def test_k_reach_then_generic_followup(self) -> None:
        session = SessionState(session_id="v5181k")
        advisor_answer(GTM_SELL, session)
        intent, _ = classify_intent(FOLLOWUP, session)
        self.assertEqual(intent, "trade_advisor")
        self.assertEqual(advisor_focus(FOLLOWUP, session), "reach")
        text = advisor_answer(FOLLOWUP, session)
        self.assertEqual(session.last_advisor_kind, "reach")
        self.assertIn("linkedin", text.casefold())
        self.assertIsNone(_RATIO_LEAK.search(text), text)

    def test_l_reach_then_buyer_finder(self) -> None:
        session = SessionState(session_id="v5181l")
        advisor_answer(GTM_SELL, session)
        intent, _ = classify_intent(FIRM_T9, session)
        self.assertEqual(intent, "buyer_finder")
        apply_utterance_slots(session, FIRM_T9)
        self.assertTrue(_woven(session.product), session.product)
        self.assertTrue(_germany(session.market), session.market)

    def test_m_reach_then_language(self) -> None:
        session = SessionState(session_id="v5181m")
        advisor_answer(GTM_SELL, session)
        intent, _ = classify_intent(LANGUAGE_T2, session)
        self.assertEqual(intent, "trade_advisor")
        text = advisor_answer(LANGUAGE_T2, session)
        low = text.casefold()
        self.assertTrue("translate" in low or "thank you" in low or "yabancı dil" in low, text)

    def test_n_reach_then_sample(self) -> None:
        session = SessionState(session_id="v5181n")
        advisor_answer(GTM_SELL, session)
        self.assertEqual(advisor_focus(SAMPLE_T5, session), "outreach")
        text = advisor_answer(SAMPLE_T5, session)
        self.assertEqual(session.last_advisor_kind, "outreach")
        low = text.casefold()
        self.assertTrue("numune" in low or "sample" in low or "muster" in low, text)
        self.assertIsNone(_RATIO_LEAK.search(text), text)
        self.assertIsNone(_TERM_LEAK.search(text), text)

    def test_o_stale_olive_to_woven(self) -> None:
        session = SessionState(session_id="v5181o", product="zeytinyağı")
        self.assertTrue(stale_session_product_conflict(session.product, GTM_HARD))
        product = active_retrieval_product(session, GTM_HARD)
        query = retrieval_query(session, GTM_HARD, role="advisor")
        self.assertFalse(_olive(product), product)
        self.assertFalse(_olive(query), query)
        text = advisor_answer(GTM_HARD, session)
        self.assertFalse(_olive(text), text)
        self.assertTrue(_woven(session.product), session.product)

    def test_draft_after_firm_hunt_keeps_product(self) -> None:
        session = SessionState(session_id="v5181t10")
        advisor_answer(GTM_SELL, session)
        classify_intent(FIRM_T9, session)
        self.assertEqual(advisor_focus(DRAFT_T10, session), "draft")
        text = advisor_answer(DRAFT_T10, session)
        self.assertEqual(session.last_advisor_kind, "draft")
        self.assertTrue(_woven(session.product), session.product)
        self.assertTrue("subject:" in text.casefold() or "dear" in text.casefold(), text)

    def test_english_names_kept_in_text_tts_is_voice_layer(self) -> None:
        session = SessionState(session_id="v5181tts", product="dokuma etiket")
        text = advisor_answer(GTM_SELL, session)
        self.assertIn("LinkedIn", text)
        self.assertTrue("Texworld" in text or "Texprocess" in text, text)
        spoken = make_speakable(text)
        self.assertIn("LinkedIn", spoken)
        tts = (BACKEND / "tts_local.py").read_text(encoding="utf-8")
        self.assertIn('TTS_LANGUAGE = "tr"', tts)
        self.assertIn("_TTS_ABBREVIATIONS", tts)
        self.assertNotIn("linke din", tts.casefold())
        self.assertNotIn("teksvörld", tts.casefold())

    def test_v512_telemetry_source_untouched(self) -> None:
        quality = (BACKEND / "agents" / "quality.py").read_text(encoding="utf-8")
        self.assertIn("def persistable_consult_telemetry", quality)
        self.assertIn("reasoning_model", quality)
        self.assertIn("canary_qwen", quality)
        self.assertIn("session_id", quality)

    def test_retrieve_guards_not_rewritten(self) -> None:
        retrieve = (BACKEND / "agents" / "retrieve.py").read_text(encoding="utf-8")
        self.assertIn("stale_session_product_conflict", retrieve)
        self.assertNotIn("last_advisor_kind", retrieve)


if __name__ == "__main__":
    unittest.main()
