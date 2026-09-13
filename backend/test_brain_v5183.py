"""V5.18.3: deictic/capacity hard stop, customer-find reach, hydrate kind."""

from __future__ import annotations

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import build_commercial_brief
from agents.goals import classify_goal
from agents.intent import classify_intent
from agents.session import (
    SessionState,
    _product_en,
    advisor_answer,
    apply_utterance_slots,
    extract_product_slot,
    format_capacity,
    hydrate,
    is_product_switch,
    product_kind,
    session_notes,
    strip_query_noise,
    utterance_product,
)
from agents.response_engine import should_reason
from prompts import advisor_focus, is_customer_find_ask

WOVEN = "dokuma etiket"
CHAIN_1 = (
    "Dokuma etiket üretiyorum ve Almanya'ya satmak istiyorum. "
    "Nereden başlamalıyım?"
)
CHAIN_2 = (
    "Almanya'da bu ürünü kimlere satabilirim ve müşterileri nereden bulabilirim?"
)
CHAIN_3 = "İlk müşteriye göndereceğim mesajı hazırlar mısın?"
CHAIN_4 = "Numuneyi nasıl göndereyim?"
FRAG = "'da bu ürünü kimlere satabilirim ve"
LIST_Q = "Almanya'daki dokuma etiket alıcılarını listele"
SWITCH_Q = "dokuma etiket yerine karton etiket üretmeye başladım"
CAPACITY_OK = "aylık 100 bin adet üretebilirim"


def _woven(product: str | None) -> bool:
    low = (product or "").casefold()
    return "dokuma" in low and "etiket" in low


def _blob(*parts: str | None) -> str:
    return "\n".join(part or "" for part in parts).casefold()


def _assert_no_fragment(*parts: str | None) -> None:
    blob = _blob(*parts)
    assert FRAG.casefold() not in blob, blob
    assert "bu ürünü kimlere satabilirim" not in blob
    assert "'da bu" not in blob


class V5183ChainHydrateTests(unittest.TestCase):
    def test_chain_1_4_hydrate_prior_like_production(self) -> None:
        sid = "v5183-chain"
        history: list[dict[str, str]] = []
        rows: list[dict[str, str | None]] = []
        for turn, question in enumerate((CHAIN_1, CHAIN_2, CHAIN_3, CHAIN_4), start=1):
            session = hydrate(sid, history)
            apply_utterance_slots(session, question)
            intent, _ = classify_intent(question, session)
            goal = classify_goal(question, session)
            focus = advisor_focus(question, session)
            advice = advisor_answer(question, session)
            notes = session_notes(session, question=question)
            brief = build_commercial_brief(question, session=session)
            en_name = _product_en(
                session.product, product_kind(session.product) == "woven"
            )
            _assert_no_fragment(
                session.product,
                session.capacity,
                notes,
                brief.company_notes,
                brief.human_plan,
                en_name,
            )
            self.assertNotIn(FRAG.casefold(), (advice or "").casefold())
            self.assertTrue(_woven(session.product), session.product)
            self.assertEqual(session.market, "Almanya")
            self.assertIsNone(session.capacity, session.capacity)
            if turn == 1:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "reach")
                self.assertEqual(session.last_advisor_kind, "reach")
            elif turn == 2:
                self.assertTrue(is_customer_find_ask(question))
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "reach")
                self.assertEqual(session.last_advisor_kind, "reach")
                self.assertFalse(should_reason(question, focus))
            elif turn == 3:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "draft")
                self.assertEqual(session.last_advisor_kind, "draft")
                self.assertIn("woven clothing labels", advice.casefold())
                self.assertIn("offer and sample from turkey", advice.casefold())
            else:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "outreach")
                self.assertEqual(session.last_advisor_kind, "outreach")
                self.assertIn("woven clothing labels", advice.casefold())
                self.assertIn("sample from turkey", advice.casefold())
            rows.append(
                {
                    "turn": str(turn),
                    "product": session.product,
                    "market": session.market,
                    "capacity": session.capacity,
                    "kind": session.last_advisor_kind,
                    "intent": intent,
                    "focus": focus,
                    "goal": goal,
                }
            )
            session.append("user", question)
            session.append("assistant", advice)
            history = session.history_dicts()

        self.assertEqual(len(rows), 4)
        t2_prior = hydrate(sid, history[:2])
        self.assertEqual(t2_prior.last_advisor_kind, "reach")
        self.assertIsNone(t2_prior.capacity)

    def test_deictic_and_question_are_not_products(self) -> None:
        self.assertIsNone(extract_product_slot(FRAG))
        self.assertIsNone(utterance_product(FRAG))
        self.assertIsNone(extract_product_slot(CHAIN_2))
        self.assertIsNone(utterance_product(CHAIN_2))
        self.assertIsNone(extract_product_slot("bu ürünü kimlere satabilirim?"))
        self.assertIsNone(utterance_product("hangi ürünü satabilirim?"))
        self.assertIsNone(extract_product_slot("bu ürün hakkında bilgi"))
        self.assertIsNone(extract_product_slot("bu ürünü nasıl ihraç ederim?"))
        noise = strip_query_noise(CHAIN_2)
        self.assertIsNone(extract_product_slot(noise))
        self.assertIsNone(utterance_product(noise))

    def test_real_products_and_switch_still_work(self) -> None:
        self.assertIn("dokuma", (extract_product_slot(CHAIN_1) or "").casefold())
        self.assertIn("etiket", (utterance_product("karton etiket üretiyorum") or "").casefold())
        woven_label = utterance_product("woven label") or extract_product_slot("woven label")
        self.assertIsNotNone(woven_label)
        cotton = utterance_product("cotton label") or extract_product_slot("cotton label")
        self.assertIsNotNone(cotton)
        session = SessionState(session_id="v5183sw", product=WOVEN, market="Almanya")
        incoming = utterance_product(SWITCH_Q)
        self.assertIsNotNone(incoming)
        self.assertTrue(is_product_switch(WOVEN, incoming))
        apply_utterance_slots(session, SWITCH_Q)
        self.assertFalse(_woven(session.product), session.product)
        self.assertIn("etiket", (session.product or "").casefold())

    def test_capacity_only_from_quantity(self) -> None:
        self.assertIsNone(extract_product_slot(CHAIN_2))
        self.assertEqual(format_capacity(CHAIN_2), "")
        self.assertEqual(format_capacity("müşterileri nereden bulabilirim?"), "")
        self.assertEqual(format_capacity("Almanya'ya satmak istiyorum."), "")
        self.assertIn("100", format_capacity(CAPACITY_OK))
        session = hydrate(
            "v5183cap",
            [
                {"role": "user", "content": CHAIN_1},
                {
                    "role": "assistant",
                    "content": "Dokuma etiket. Numune, kapasite, teslim süresi yazın. "
                    "İngilizce için «mail taslağı» yazın.",
                },
                {"role": "user", "content": CHAIN_2},
            ],
        )
        self.assertTrue(_woven(session.product), session.product)
        self.assertIsNone(session.capacity)
        self.assertEqual(session.last_advisor_kind, "reach")
        _assert_no_fragment(session.product, session.capacity, session_notes(session))

        filled = SessionState(session_id="v5183capok")
        apply_utterance_slots(filled, CAPACITY_OK)
        self.assertIsNotNone(filled.capacity)

    def test_customer_find_vs_list_hunt(self) -> None:
        self.assertTrue(is_customer_find_ask(CHAIN_2))
        self.assertTrue(is_customer_find_ask("müşterileri nereden bulabilirim"))
        self.assertTrue(is_customer_find_ask("alıcıları nereden bulabilirim"))
        self.assertTrue(is_customer_find_ask("bu ürünü kimlere satabilirim"))
        session = SessionState(session_id="v5183cf", product=WOVEN, market="Almanya")
        intent, _ = classify_intent(CHAIN_2, session)
        self.assertEqual(intent, "trade_advisor")
        self.assertEqual(advisor_focus(CHAIN_2, session), "reach")
        list_intent, _ = classify_intent(LIST_Q, session)
        self.assertEqual(list_intent, "buyer_finder")
        bare, _ = classify_intent("Müşteriyi nasıl bulurum?")
        self.assertEqual(bare, "buyer_finder")


if __name__ == "__main__":
    unittest.main()
