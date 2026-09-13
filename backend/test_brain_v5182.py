"""V5.18.2: deictic «bu ürünü» is not a new product slot."""

from __future__ import annotations

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.session import (
    SessionState,
    advisor_answer,
    apply_utterance_slots,
    extract_product_slot,
    hydrate,
    is_deictic_product_reference,
    is_product_switch,
    stale_session_product_conflict,
    utterance_product,
)

WOVEN = "dokuma etiket"
REPRO_A = "bu ürünü kimlere satabilirim ve ilk müşteriye nasıl ulaşacağım"
DEICTIC_B = "Bu ürün için Almanya'da hangi müşterilere ulaşmalıyım?"
DEICTIC_C = "Bu ürünün müşterilerini nereden bulabilirim?"
DEICTIC_WITH = "İlk müşteriye bu ürünle nasıl ulaşırım?"
CARTON = "Yeni olarak karton etiket üretmeye başladım."
OLIVE = "Zeytinyağı satmak istiyorum."
CHAIN_1 = (
    "Dokuma etiket üretiyorum ve Almanya'ya satmak istiyorum. "
    "Nereden başlamalıyım?"
)
CHAIN_2 = (
    "Almanya'da bu ürünü kimlere satabilirim ve müşterileri nereden bulabilirim?"
)
CHAIN_3 = "İlk müşteriye göndereceğim mesajı hazırlar mısın?"
CHAIN_4 = "Numuneyi nasıl göndereyim?"

_BAD_FRAGMENTS = (
    "bu ürünü kimlere",
    "bu ürünü kimlere satabilirim",
    "bu ürün için",
    "bu ürünün müşterilerini",
)


def _woven(product: str | None) -> bool:
    low = (product or "").casefold()
    return "dokuma" in low and "etiket" in low


def _assert_clean_slot(product: str | None) -> None:
    low = (product or "").casefold()
    for fragment in _BAD_FRAGMENTS:
        assert fragment not in low, f"{fragment!r} in {product!r}"
    assert "kimlere satabilirim" not in low, product


class V5182DeicticProductTests(unittest.TestCase):
    def test_a_repro_keeps_woven_no_switch(self) -> None:
        session = SessionState(session_id="v5182a", product=WOVEN, market="Almanya")
        self.assertTrue(is_deictic_product_reference(REPRO_A))
        self.assertIsNone(extract_product_slot(REPRO_A))
        self.assertIsNone(utterance_product(REPRO_A))
        self.assertFalse(stale_session_product_conflict(session.product, REPRO_A))
        switched = apply_utterance_slots(session, REPRO_A)
        self.assertFalse(switched)
        self.assertTrue(_woven(session.product), session.product)
        self.assertEqual(session.market, "Almanya")
        self.assertIsNone(session.capacity)
        _assert_clean_slot(session.product)

    def test_b_bu_urun_icin_keeps_woven(self) -> None:
        session = SessionState(session_id="v5182b", product=WOVEN, market="Almanya")
        apply_utterance_slots(session, DEICTIC_B)
        self.assertTrue(_woven(session.product), session.product)
        self.assertEqual(session.market, "Almanya")
        _assert_clean_slot(session.product)

    def test_c_bu_urunun_keeps_woven(self) -> None:
        session = SessionState(session_id="v5182c", product=WOVEN, market="Almanya")
        apply_utterance_slots(session, DEICTIC_C)
        self.assertTrue(_woven(session.product), session.product)
        _assert_clean_slot(session.product)

    def test_deictic_inflections_are_not_incoming_products(self) -> None:
        phrases = (
            "bu ürün",
            "bu ürünü",
            "bu ürüne",
            "bu ürünün",
            "bu üründen",
            "bu üründe",
            "bu ürünle",
            "bu ürün için",
            "bu ürün hakkında",
            DEICTIC_WITH,
        )
        for phrase in phrases:
            session = SessionState(
                session_id="v5182inf",
                product=WOVEN,
                market="Almanya",
                capacity="Aylık 2.5 milyon metre",
            )
            self.assertTrue(is_deictic_product_reference(phrase), phrase)
            self.assertIsNone(utterance_product(phrase), phrase)
            apply_utterance_slots(session, phrase)
            self.assertEqual(session.product, WOVEN, phrase)
            self.assertEqual(session.capacity, "Aylık 2.5 milyon metre", phrase)
            self.assertEqual(session.market, "Almanya", phrase)

    def test_d_carton_label_still_switches(self) -> None:
        session = SessionState(session_id="v5182d", product=WOVEN, market="Almanya")
        incoming = utterance_product(CARTON)
        self.assertIsNotNone(incoming)
        self.assertTrue(is_product_switch(WOVEN, incoming))
        self.assertTrue(stale_session_product_conflict(session.product, CARTON))
        apply_utterance_slots(session, CARTON)
        self.assertFalse(_woven(session.product), session.product)
        self.assertIn("etiket", (session.product or "").casefold())
        _assert_clean_slot(session.product)

    def test_e_olive_still_switches(self) -> None:
        session = SessionState(session_id="v5182e", product=WOVEN, market="Almanya")
        self.assertTrue(stale_session_product_conflict(session.product, OLIVE))
        apply_utterance_slots(session, OLIVE)
        low = (session.product or "").casefold()
        self.assertIn("zeytin", low)
        self.assertFalse(_woven(session.product), session.product)

    def test_i_conversation_chain_keeps_woven(self) -> None:
        session = SessionState(session_id="v5182i")
        advisor_answer(CHAIN_1, session)
        self.assertTrue(_woven(session.product), session.product)
        self.assertEqual(session.market, "Almanya")
        market = session.market
        advisor_answer(CHAIN_2, session)
        self.assertTrue(_woven(session.product), session.product)
        self.assertEqual(session.market, market)
        _assert_clean_slot(session.product)
        advisor_answer(CHAIN_3, session)
        self.assertTrue(_woven(session.product), session.product)
        self.assertEqual(session.last_advisor_kind, "draft")
        advisor_answer(CHAIN_4, session)
        self.assertTrue(_woven(session.product), session.product)
        self.assertEqual(session.last_advisor_kind, "outreach")
        _assert_clean_slot(session.product)

    def test_hydrate_does_not_write_deictic_fragment(self) -> None:
        session = hydrate(
            "v5182h",
            [
                {"role": "user", "content": CHAIN_1},
                {
                    "role": "assistant",
                    "content": "Dokuma etiket için Almanya'da LinkedIn sourcing.",
                },
                {"role": "user", "content": REPRO_A},
            ],
        )
        self.assertTrue(_woven(session.product), session.product)
        self.assertEqual(session.market, "Almanya")
        _assert_clean_slot(session.product)


if __name__ == "__main__":
    unittest.main()
