"""V5.18.5 Phase 2: commercial policy + user-facing NBA guidance."""

from __future__ import annotations

import re
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import (
    apply_commercial_guidance,
    build_commercial_policy,
    build_commercial_situation,
)
from agents.intent import classify_intent
from agents.session import (
    SessionState,
    advisor_answer,
    apply_utterance_slots,
    hydrate,
)
from prompts import advisor_focus, is_customer_find_ask

CHAIN_1 = (
    "Dokuma etiket üretiyorum ve Almanya'ya satmak istiyorum. "
    "Nereden başlamalıyım?"
)
CHAIN_2 = (
    "Almanya'da bu ürünü kimlere satabilirim ve müşterileri nereden bulabilirim?"
)
CHAIN_3 = "İlk müşteriye göndereceğim mesajı hazırlar mısın?"
CHAIN_4 = "Numuneyi nasıl göndereyim?"

_INTERNAL = (
    "identify_customer",
    "qualify_buyer",
    "prepare_outreach",
    "send_sample",
    "prepare_quotation",
    "next_best_action",
    "missing_critical",
    "sales_stage",
)

_EARLY_TOPIC = re.compile(
    r"(?i)("
    r"\bexw\b|\bdap\b|\bfca\b|\bfob\b|\bcif\b|"
    r"incoterm|"
    r"pe[sş]in|"
    r"akreditif|"
    r"ödeme\s*[sş]art|"
    r"g[uü]mr[uü]k|"
    r"\bnavlun\b|"
    r"\bfreight\b|"
    r"\bcustoms\b"
    r")"
)


def _woven(product: str | None) -> bool:
    low = (product or "").casefold()
    return "dokuma" in low and "etiket" in low


def _guided(question: str, session: SessionState, matches: list | None = None) -> tuple[str, object]:
    focus = advisor_focus(question, session)
    sit = build_commercial_situation(question, session=session, focus=focus)
    raw = advisor_answer(question, session, matches=matches or [])
    text = apply_commercial_guidance(
        raw, sit, question, session=session, focus=focus
    )
    return text, sit


def _no_italy_nl(text: str) -> None:
    low = (text or "").casefold()
    assert "italya" not in low and "italy" not in low, text
    assert "hollanda" not in low and "netherlands" not in low, text


def _no_internal(text: str) -> None:
    for token in _INTERNAL:
        assert token not in (text or ""), text


def _no_early_commercial(text: str) -> None:
    assert not _EARLY_TOPIC.search(text or ""), text


class V51852ProgressionTests(unittest.TestCase):
    def test_t1_t4_commercial_progression(self) -> None:
        sid = "v51852-chain"
        history: list[dict[str, str]] = []
        t1_text = ""
        for turn, question in enumerate((CHAIN_1, CHAIN_2, CHAIN_3, CHAIN_4), start=1):
            session = hydrate(sid, history)
            apply_utterance_slots(session, question)
            intent, _ = classify_intent(question, session)
            focus = advisor_focus(question, session)
            advice, sit = _guided(question, session)

            self.assertTrue(_woven(session.product), session.product)
            self.assertEqual(session.market, "Almanya")
            self.assertIsNone(session.capacity)
            _no_italy_nl(advice)
            self.assertNotIn("Capacity: Dokuma", advice)
            self.assertNotIn("alıcısı", advice.casefold())
            _no_internal(advice)
            self.assertLessEqual(advice.count("?"), 1)

            if turn == 1:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "reach")
                self.assertEqual(sit.sales_stage, "MARKET")
                self.assertEqual(sit.next_best_action, "identify_customer")
                self.assertIn("giyim", advice.casefold())
                self.assertIn("hedef müşteri", advice.casefold())
                self.assertTrue(
                    "marka" in advice.casefold() and "konfeksiyon" in advice.casefold(),
                    advice,
                )
                _no_early_commercial(advice)
                t1_text = advice
            elif turn == 2:
                self.assertTrue(is_customer_find_ask(question))
                self.assertEqual(sit.sales_stage, "CUSTOMER_IDENTIFICATION")
                self.assertEqual(sit.next_best_action, "prepare_outreach")
                self.assertNotEqual(advice.strip(), t1_text.strip())
                low = advice.casefold()
                self.assertIn("giyim", low)
                self.assertIn("nereden bulunur", low)
                self.assertIn("nasıl temas", low)
                self.assertTrue(
                    "linkedin" in low or "fuar" in low or "mail" in low, advice
                )
                self.assertTrue(
                    "mesaj" in low or "ilk" in low, advice
                )
                _no_early_commercial(advice)
            elif turn == 3:
                self.assertEqual(sit.sales_stage, "OUTREACH")
                self.assertEqual(focus, "draft")
                self.assertIn("purchasing team", advice.casefold())
                self.assertEqual(advice.count("Subject:"), 1)
                # Draft playbook unchanged — no Phase 2 append.
                self.assertNotIn("Bence sıradaki", advice)
            else:
                self.assertEqual(sit.sales_stage, "SAMPLE")
                self.assertEqual(focus, "outreach")
                self.assertIn("sample from turkey", advice.casefold())
                self.assertIn("numune", advice.casefold())
                # Sample playbook unchanged — no Phase 2 NBA append.
                self.assertNotIn("Bence sıradaki", advice)

            session.append("user", question)
            session.append("assistant", advice)
            history = session.history_dicts()


class V51852NbaTests(unittest.TestCase):
    def test_market_nba_customer(self) -> None:
        session = SessionState(session_id="v51852-nba-m")
        apply_utterance_slots(session, CHAIN_1)
        advice, sit = _guided(CHAIN_1, session)
        self.assertEqual(sit.next_best_action, "identify_customer")
        self.assertIn("hedef müşteri", advice.casefold())

    def test_customer_nba_outreach(self) -> None:
        session = SessionState(session_id="v51852-nba-c")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        advice, sit = _guided(CHAIN_2, session)
        self.assertEqual(sit.next_best_action, "prepare_outreach")
        self.assertTrue("mesaj" in advice.casefold() or "yazalım" in advice.casefold())

    def test_outreach_nba_internal_sample(self) -> None:
        session = SessionState(session_id="v51852-nba-o")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        _advice, sit = _guided(CHAIN_3, session)
        self.assertEqual(sit.next_best_action, "send_sample")

    def test_sample_nba_quotation(self) -> None:
        session = SessionState(session_id="v51852-nba-s")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        _advice, sit = _guided(CHAIN_4, session)
        self.assertEqual(sit.next_best_action, "prepare_quotation")


class V51852MissingTests(unittest.TestCase):
    def test_buyer_type_question_on_market_only(self) -> None:
        session = SessionState(session_id="v51852-miss-b")
        apply_utterance_slots(session, CHAIN_1)
        advice, sit = _guided(CHAIN_1, session)
        self.assertEqual(sit.missing_critical, ["buyer_type"])
        self.assertIn("marka mı", advice.casefold())
        self.assertLessEqual(advice.count("?"), 1)

    def test_no_buyer_reask_on_customer_find(self) -> None:
        session = SessionState(session_id="v51852-miss-t2")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        advice, sit = _guided(CHAIN_2, session)
        self.assertEqual(sit.missing_critical, ["buyer_type"])
        self.assertNotIn("marka mı", advice.casefold())

    def test_capacity_missing_not_asked_on_draft(self) -> None:
        session = SessionState(session_id="v51852-miss-c")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        advice, sit = _guided(CHAIN_3, session)
        self.assertEqual(sit.missing_critical, ["capacity"])
        self.assertNotIn("kapasiteniz", advice.casefold())
        self.assertIsNone(session.capacity)

    def test_market_missing_question(self) -> None:
        session = SessionState(session_id="v51852-miss-m")
        session.product = "Dokuma Etiket"
        sit = build_commercial_situation(
            "Bu ürünü satmak istiyorum, nereden başlamalıyım?",
            session=session,
            focus="reach",
        )
        policy = build_commercial_policy(
            sit,
            question="Bu ürünü satmak istiyorum, nereden başlamalıyım?",
            session=session,
            focus="reach",
        )
        self.assertEqual(sit.missing_critical, ["market"])
        self.assertTrue(policy.ask_missing)
        self.assertIn("pazar", (policy.missing_question or "").casefold())

    def test_product_missing(self) -> None:
        session = SessionState(session_id="v51852-miss-p")
        sit = build_commercial_situation(
            "İhracata nereden başlamalıyım?",
            session=session,
            focus="reach",
        )
        self.assertEqual(sit.missing_critical, ["product"])


class V51852SafetyTests(unittest.TestCase):
    def test_no_unsolicited_payment_incoterm_on_t1(self) -> None:
        session = SessionState(session_id="v51852-safe-pay")
        apply_utterance_slots(session, CHAIN_1)
        advice, _sit = _guided(CHAIN_1, session)
        _no_early_commercial(advice)

    def test_germany_capacity_fake_buyer(self) -> None:
        session = SessionState(session_id="v51852-safe-all")
        apply_utterance_slots(session, CHAIN_1)
        advice, _sit = _guided(CHAIN_1, session)
        self.assertEqual(session.market, "Almanya")
        self.assertIsNone(session.capacity)
        _no_italy_nl(advice)
        self.assertNotIn("alıcısı", advice.casefold())
        self.assertNotIn("@", advice)

    def test_buyer_type_known_skips_missing(self) -> None:
        session = SessionState(session_id="v51852-known")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        q = "Almanya'da konfeksiyon üreticilerini hedefliyorum, nereden başlamalıyım?"
        sit = build_commercial_situation(q, session=session, focus="reach")
        self.assertEqual(sit.sales_stage, "MARKET")
        self.assertEqual(sit.missing_critical, [])


if __name__ == "__main__":
    unittest.main()
