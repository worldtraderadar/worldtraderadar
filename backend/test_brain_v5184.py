"""V5.18.4: response-quality — capacity omit, market discipline, org-only draft."""

from __future__ import annotations

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.intent import classify_intent
from agents.session import (
    SessionState,
    advisor_answer,
    advisor_draft_reply,
    advisor_outreach_reply,
    apply_utterance_slots,
    extract_product_slot,
    hydrate,
    is_product_switch,
    match_firm_addressee,
    match_firm_label,
    utterance_product,
)
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
SWITCH_Q = "dokuma etiket yerine karton etiket üretmeye başladım"
DEICTIC = "bu ürünü kimlere satabilirim?"


def _woven(product: str | None) -> bool:
    low = (product or "").casefold()
    return "dokuma" in low and "etiket" in low


def _no_italy(text: str) -> None:
    low = (text or "").casefold()
    assert "italya" not in low and "italy" not in low, text


def _no_ops_numbers(text: str) -> None:
    body = text or ""
    assert "8–12" not in body and "8-12" not in body, body
    assert "5–7" not in body and "5-7" not in body, body


class _DestOnly:
    organization_name = None
    destination_country = "DE"
    product_name = "Woven clothing labels (dokuma etiket)"


class _OrgRow:
    organization_name = "Acme Labels GmbH"
    destination_country = "DE"
    product_name = "woven labels"


class V5184ResponseQualityTests(unittest.TestCase):
    def test_chain_1_4_response_quality(self) -> None:
        sid = "v5184-chain"
        history: list[dict[str, str]] = []
        t1_text = ""
        for turn, question in enumerate((CHAIN_1, CHAIN_2, CHAIN_3, CHAIN_4), start=1):
            session = hydrate(sid, history)
            apply_utterance_slots(session, question)
            intent, _ = classify_intent(question, session)
            focus = advisor_focus(question, session)
            advice = advisor_answer(question, session, matches=[_DestOnly()])
            self.assertTrue(_woven(session.product), session.product)
            self.assertEqual(session.market, "Almanya")
            self.assertIsNone(session.capacity)
            _no_italy(advice)
            self.assertNotIn("Capacity: Dokuma", advice)
            self.assertNotIn("Capacity: dokuma", advice.casefold())
            self.assertNotIn("alıcısı", advice.casefold())
            if turn == 1:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "reach")
                self.assertEqual(session.last_advisor_kind, "reach")
                _no_ops_numbers(advice)
                self.assertIn("giyim", advice.casefold())
                t1_text = advice
            elif turn == 2:
                self.assertTrue(is_customer_find_ask(question))
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "reach")
                self.assertNotEqual(advice.strip(), t1_text.strip())
                low = advice.casefold()
                self.assertIn("giyim", low)
                self.assertIn("nereden bulunur", low)
                self.assertIn("nasıl temas", low)
                self.assertTrue(
                    "linkedin" in low or "fuar" in low or "mail" in low, advice
                )
                _no_ops_numbers(advice)
            elif turn == 3:
                self.assertEqual(focus, "draft")
                self.assertIn("woven clothing labels", advice.casefold())
                self.assertIn("purchasing team", advice.casefold())
                self.assertEqual(advice.count("Subject:"), 1)
            else:
                self.assertEqual(focus, "outreach")
                self.assertIn("sample from turkey", advice.casefold())
                _no_ops_numbers(advice)
            session.append("user", question)
            session.append("assistant", advice)
            history = session.history_dicts()

    def test_capacity_none_omits_capacity_line(self) -> None:
        draft = advisor_draft_reply("Dokuma etiket", None, [], session_market="Almanya")
        self.assertNotIn("Capacity:", draft)
        out = advisor_outreach_reply(
            "Dokuma etiket", None, [], session_market="Almanya"
        )
        self.assertNotIn("Capacity:", out)
        filled = advisor_draft_reply(
            "Dokuma etiket",
            "Aylık 2.5 milyon metre",
            [],
            session_market="Almanya",
        )
        self.assertIn("Capacity:", filled)

    def test_single_market_germany(self) -> None:
        session = SessionState(
            session_id="v5184m", product=WOVEN, market="Almanya"
        )
        text = advisor_answer(CHAIN_1, session, matches=[_DestOnly()])
        _no_italy(text)
        self.assertIn("Almanya", text)

    def test_empty_organization_name(self) -> None:
        self.assertEqual(match_firm_label(_DestOnly()), "")
        self.assertEqual(match_firm_addressee(_DestOnly()), "")
        draft = advisor_draft_reply(
            WOVEN, None, [_DestOnly(), _DestOnly()], session_market="Almanya"
        )
        self.assertNotIn("alıcısı", draft.casefold())
        self.assertIn("Purchasing Team", draft)
        self.assertEqual(draft.count("Subject:"), 1)
        named = advisor_draft_reply(
            WOVEN, None, [_OrgRow()], session_market="Almanya"
        )
        self.assertIn("Acme Labels GmbH", named)

    def test_deictic_and_switch(self) -> None:
        session = SessionState(
            session_id="v5184d", product=WOVEN, market="Almanya"
        )
        apply_utterance_slots(session, DEICTIC)
        self.assertEqual(session.product, WOVEN)
        self.assertIsNone(utterance_product(DEICTIC))
        self.assertIsNone(extract_product_slot(DEICTIC))
        incoming = utterance_product(SWITCH_Q)
        self.assertIsNotNone(incoming)
        self.assertTrue(is_product_switch(WOVEN, incoming))
        apply_utterance_slots(session, SWITCH_Q)
        self.assertFalse(_woven(session.product), session.product)


if __name__ == "__main__":
    unittest.main()
