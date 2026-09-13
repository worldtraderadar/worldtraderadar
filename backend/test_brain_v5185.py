"""V5.18.5 Phase 1: CommercialSituation core — derived stage / missing / NBA."""

from __future__ import annotations

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import (
    build_commercial_situation,
    derive_missing_critical,
    derive_next_best_action,
    derive_sales_stage,
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


def _woven(product: str | None) -> bool:
    low = (product or "").casefold()
    return "dokuma" in low and "etiket" in low


def _situation_for(question: str, session: SessionState | None = None):
    focus = advisor_focus(question, session)
    return build_commercial_situation(question, session=session, focus=focus)


class V5185SalesStageTests(unittest.TestCase):
    def test_product_market_discovery_chain1(self) -> None:
        session = SessionState(session_id="v5185-t1")
        apply_utterance_slots(session, CHAIN_1)
        sit = _situation_for(CHAIN_1, session)
        self.assertEqual(sit.sales_stage, "MARKET")
        self.assertTrue(_woven(session.product))
        self.assertEqual(session.market, "Almanya")
        self.assertIsNone(session.capacity)

    def test_customer_identification_chain2(self) -> None:
        session = SessionState(session_id="v5185-t2")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        self.assertTrue(is_customer_find_ask(CHAIN_2))
        sit = _situation_for(CHAIN_2, session)
        self.assertEqual(sit.sales_stage, "CUSTOMER_IDENTIFICATION")

    def test_outreach_chain3(self) -> None:
        session = SessionState(session_id="v5185-t3")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for(CHAIN_3, session)
        self.assertEqual(sit.sales_stage, "OUTREACH")
        self.assertEqual(advisor_focus(CHAIN_3, session), "draft")

    def test_sample_chain4(self) -> None:
        session = SessionState(session_id="v5185-t4")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for(CHAIN_4, session)
        self.assertEqual(sit.sales_stage, "SAMPLE")
        self.assertEqual(advisor_focus(CHAIN_4, session), "outreach")

    def test_quotation_stage(self) -> None:
        session = SessionState(session_id="v5185-q")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for("Bu ürün için fiyat teklifi hazırlar mısın?", session)
        self.assertEqual(sit.sales_stage, "QUOTATION")

    def test_payment_stage(self) -> None:
        session = SessionState(session_id="v5185-pay")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for("Ödeme şartlarını nasıl belirlemeliyim?", session)
        self.assertEqual(sit.sales_stage, "PAYMENT")

    def test_logistics_stage(self) -> None:
        session = SessionState(session_id="v5185-log")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for("Teslim şekli EXW mi DAP mi olmalı?", session)
        self.assertEqual(sit.sales_stage, "LOGISTICS")

    def test_product_only_stage(self) -> None:
        session = SessionState(session_id="v5185-prod")
        session.product = "Dokuma Etiket"
        stage = derive_sales_stage(
            "Bu ürünü satmak istiyorum, nereden başlamalıyım?",
            session=session,
            focus="reach",
        )
        self.assertEqual(stage, "PRODUCT")


class V5185MissingCriticalTests(unittest.TestCase):
    def test_product_missing(self) -> None:
        session = SessionState(session_id="v5185-miss-p")
        missing = derive_missing_critical("DISCOVERY", session=session)
        self.assertEqual(missing, ["product"])

    def test_market_missing(self) -> None:
        session = SessionState(session_id="v5185-miss-m")
        session.product = "Dokuma Etiket"
        missing = derive_missing_critical("PRODUCT", session=session)
        self.assertEqual(missing, ["market"])

    def test_buyer_type_missing_on_customer_id(self) -> None:
        session = SessionState(session_id="v5185-miss-b")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        missing = derive_missing_critical(
            "CUSTOMER_IDENTIFICATION", session=session
        )
        self.assertEqual(missing, ["buyer_type"])

    def test_capacity_missing_on_outreach(self) -> None:
        session = SessionState(session_id="v5185-miss-c")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        session.capacity = None
        missing = derive_missing_critical("OUTREACH", session=session)
        self.assertEqual(missing, ["capacity"])

    def test_missing_at_most_one(self) -> None:
        session = SessionState(session_id="v5185-miss-1")
        apply_utterance_slots(session, CHAIN_1)
        sit = build_commercial_situation(CHAIN_1, session=session, focus="reach")
        self.assertLessEqual(len(sit.missing_critical), 1)


class V5185NextBestActionTests(unittest.TestCase):
    def test_market_to_identify_customer(self) -> None:
        session = SessionState(session_id="v5185-nba-m")
        apply_utterance_slots(session, CHAIN_1)
        sit = _situation_for(CHAIN_1, session)
        self.assertEqual(sit.sales_stage, "MARKET")
        self.assertEqual(sit.next_best_action, "identify_customer")

    def test_customer_id_to_qualify(self) -> None:
        session = SessionState(session_id="v5185-nba-c")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for(CHAIN_2, session)
        # Customer-find ask → next is outreach prep (not a rigid qualify-only FSM).
        self.assertEqual(sit.next_best_action, "prepare_outreach")

    def test_outreach_to_sample(self) -> None:
        session = SessionState(session_id="v5185-nba-o")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for(CHAIN_3, session)
        self.assertEqual(sit.next_best_action, "send_sample")

    def test_sample_to_quotation(self) -> None:
        session = SessionState(session_id="v5185-nba-s")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for(CHAIN_4, session)
        self.assertEqual(sit.next_best_action, "prepare_quotation")

    def test_quotation_to_negotiate(self) -> None:
        nba = derive_next_best_action("QUOTATION")
        self.assertEqual(nba, "negotiate")


class V5185SafetyTests(unittest.TestCase):
    def test_capacity_not_inferred_from_product(self) -> None:
        session = SessionState(session_id="v5185-safe-cap")
        apply_utterance_slots(session, CHAIN_1)
        before_cap = session.capacity
        sit = _situation_for(CHAIN_1, session)
        self.assertIsNone(before_cap)
        self.assertIsNone(session.capacity)
        self.assertNotEqual(session.capacity, session.product)
        self.assertIn("capacity_unset", sit.risk_flags)
        # Derived may note capacity gap later; never fills the slot.
        self.assertIsNone(session.capacity)

    def test_market_germany_only_no_italy_netherlands(self) -> None:
        session = SessionState(session_id="v5185-safe-mkt")
        apply_utterance_slots(session, CHAIN_1)
        sit = _situation_for(CHAIN_1, session)
        self.assertEqual(session.market, "Almanya")
        blob = " ".join(
            [
                sit.sales_stage,
                sit.next_best_action,
                *sit.missing_critical,
                *sit.risk_flags,
            ]
        ).casefold()
        self.assertNotIn("italya", blob)
        self.assertNotIn("italy", blob)
        self.assertNotIn("hollanda", blob)
        self.assertNotIn("netherlands", blob)
        advice = advisor_answer(CHAIN_1, session, matches=[])
        low = advice.casefold()
        self.assertNotIn("italya", low)
        self.assertNotIn("hollanda", low)

    def test_no_fake_buyer_in_situation(self) -> None:
        session = SessionState(session_id="v5185-safe-buyer")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = _situation_for(CHAIN_2, session)
        blob = str(sit.as_dict()).casefold()
        self.assertNotIn("gmbh", blob)
        self.assertNotIn("alıcısı", blob)
        self.assertNotIn("@", blob)


class V5185ChainBehavioralTests(unittest.TestCase):
    def test_germany_woven_chain_situation_and_slots(self) -> None:
        sid = "v5185-chain"
        history: list[dict[str, str]] = []
        t1_stage = ""
        for turn, question in enumerate((CHAIN_1, CHAIN_2, CHAIN_3, CHAIN_4), start=1):
            session = hydrate(sid, history)
            apply_utterance_slots(session, question)
            intent, _ = classify_intent(question, session)
            focus = advisor_focus(question, session)
            sit = build_commercial_situation(
                question, session=session, focus=focus
            )
            advice = advisor_answer(question, session, matches=[])

            self.assertTrue(_woven(session.product), session.product)
            self.assertEqual(session.market, "Almanya")
            self.assertIsNone(session.capacity)
            self.assertNotIn("italya", advice.casefold())
            self.assertNotIn("Capacity: Dokuma", advice)
            self.assertNotIn("alıcısı", advice.casefold())
            self.assertLessEqual(len(sit.missing_critical), 1)

            if turn == 1:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "reach")
                self.assertEqual(sit.sales_stage, "MARKET")
                self.assertEqual(sit.next_best_action, "identify_customer")
                t1_stage = sit.sales_stage
            elif turn == 2:
                self.assertEqual(sit.sales_stage, "CUSTOMER_IDENTIFICATION")
                self.assertNotEqual(sit.sales_stage, t1_stage)
                self.assertEqual(sit.next_best_action, "prepare_outreach")
                self.assertEqual(focus, "reach")
            elif turn == 3:
                self.assertEqual(sit.sales_stage, "OUTREACH")
                self.assertEqual(focus, "draft")
                self.assertEqual(sit.next_best_action, "send_sample")
            else:
                self.assertEqual(sit.sales_stage, "SAMPLE")
                self.assertEqual(focus, "outreach")
                self.assertEqual(sit.next_best_action, "prepare_quotation")

            session.append("user", question)
            session.append("assistant", advice)
            history = session.history_dicts()


if __name__ == "__main__":
    unittest.main()
