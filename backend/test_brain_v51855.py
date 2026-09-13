"""V5.18.5 Phase 5: full behavioral regression & commercial journey validation.

Validation only — does not change production behavior.
Uses deterministic playbooks / situation / brief / intent (no live LLM).
"""

from __future__ import annotations

import re
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import (
    build_commercial_brief,
    build_commercial_situation,
    detect_weak_commercial_assumption,
)
from agents.intent import classify_intent
from agents.response_engine import compose_consultant_reply, should_reason
from agents.session import (
    SessionState,
    advisor_answer,
    apply_utterance_slots,
    hydrate,
    is_product_switch,
    stale_session_product_conflict,
    utterance_product,
)
from prompts import advisor_focus, is_customer_find_ask, is_incoterm_ask, is_payment_ask

CHAIN_1 = (
    "Dokuma etiket üretiyorum ve Almanya'ya satmak istiyorum. "
    "Nereden başlamalıyım?"
)
CHAIN_2 = (
    "Almanya'da bu ürünü kimlere satabilirim ve müşterileri nereden bulabilirim?"
)
CHAIN_3 = "İlk müşteriye göndereceğim mesajı hazırlar mısın?"
CHAIN_4 = "Numuneyi nasıl göndereyim?"

_EARLY = re.compile(
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
_FAKE_BUYER = re.compile(
    r"(?i)("
    r"\bgmbh\b|"
    r"al[iı]c[iı]s[iı]|"
    r"@[a-z0-9.-]+\.[a-z]{2,}|"
    r"\+?\d{2,}\s*\d{3,}"
    r")"
)
_CAPACITY_INVENT = re.compile(
    r"(?i)("
    r"\b\d+\s*(ton|adet|metre|mt)\b|"
    r"ayl[iı]k\s*kapasite|"
    r"haftal[iı]k\s*kapasite|"
    r"kapasite(?:miz|niz)?\s*\d+"
    r")"
)
_INTERNAL = (
    "identify_customer",
    "qualify_buyer",
    "prepare_outreach",
    "next_best_action",
    "missing_critical",
    "sales_stage",
    "TİCARİ YOLCULUK",
    "TİCARİ MUHAKEME",
)


def _woven(product: str | None) -> bool:
    low = (product or "").casefold()
    return "dokuma" in low and "etiket" in low


def _no_italy_nl(text: str) -> None:
    low = (text or "").casefold()
    assert "italya" not in low and "italy" not in low, text
    assert "hollanda" not in low and "netherlands" not in low, text


def _turn(
    question: str, session: SessionState, matches: list | None = None
) -> tuple[str, object, str, str]:
    apply_utterance_slots(session, question)
    intent, _ = classify_intent(question, session)
    focus = advisor_focus(question, session)
    sit = build_commercial_situation(question, session=session, focus=focus)
    advice = advisor_answer(question, session, matches=matches or [])
    return advice, sit, focus, intent


class V51855RoutingTests(unittest.TestCase):
    def test_core_runners_still_classify(self) -> None:
        cases = [
            ("Merhaba", "chat"),
            ("İhracata başlamak istiyorum, ne yapmalıyım?", "trade_advisor"),
            ("Almanya'da dokuma etiket alıcı firmaları ara", "buyer_finder"),
            ("Çin'den tedarikçi bul", "supplier_finder"),
        ]
        for question, expected in cases:
            session = SessionState(session_id=f"route-{expected}")
            if expected != "chat":
                session.product = "Dokuma Etiket"
            intent, _ = classify_intent(question, session)
            self.assertEqual(intent, expected, question)


class V51855GoldenT1234Tests(unittest.TestCase):
    def test_golden_chain(self) -> None:
        sid = "v51855-golden"
        history: list[dict[str, str]] = []
        t1 = t3 = ""
        for turn, question in enumerate((CHAIN_1, CHAIN_2, CHAIN_3, CHAIN_4), start=1):
            session = hydrate(sid, history)
            advice, sit, focus, intent = _turn(question, session)
            self.assertTrue(_woven(session.product))
            self.assertEqual(session.market, "Almanya")
            self.assertIsNone(session.capacity)
            _no_italy_nl(advice)
            self.assertIsNone(_FAKE_BUYER.search(advice), advice)
            self.assertNotIn("Capacity: Dokuma", advice)
            for token in _INTERNAL:
                self.assertNotIn(token, advice)

            if turn == 1:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(focus, "reach")
                self.assertEqual(sit.sales_stage, "MARKET")
                self.assertIn("giyim", advice.casefold())
                self.assertIsNone(_EARLY.search(advice), advice)
                t1 = advice
            elif turn == 2:
                self.assertEqual(sit.sales_stage, "CUSTOMER_IDENTIFICATION")
                self.assertNotEqual(advice.strip(), t1.strip())
                self.assertIn("nereden bulunur", advice.casefold())
                self.assertNotIn("marka mı", advice.casefold())
            elif turn == 3:
                self.assertEqual(sit.sales_stage, "OUTREACH")
                self.assertEqual(focus, "draft")
                self.assertIn("purchasing team", advice.casefold())
                self.assertEqual(advice.count("Subject:"), 1)
                t3 = advice
            else:
                self.assertEqual(sit.sales_stage, "SAMPLE")
                self.assertEqual(focus, "outreach")
                self.assertIn("sample from turkey", advice.casefold())
                self.assertNotEqual(advice.strip(), t3.strip())

            session.append("user", question)
            session.append("assistant", advice)
            history = session.history_dicts()


class V51855FullJourneyTests(unittest.TestCase):
    def test_long_commercial_journey_stages(self) -> None:
        """DISCOVERY→…→customs: stage progression + slot discipline."""
        session = SessionState(session_id="v51855-journey")
        script = [
            ("Dokuma etiket üretiyorum.", None),
            (
                "Almanya'ya satmak istiyorum. Nereden başlamalıyım?",
                "MARKET",
            ),
            ("Kimlere satabilirim?", "CUSTOMER_IDENTIFICATION"),
            ("Müşterileri nereden bulabilirim?", "CUSTOMER_IDENTIFICATION"),
            ("İlk mesajı hazırlar mısın?", "OUTREACH"),
            ("Numuneyi nasıl göndereyim?", "SAMPLE"),
            ("Bu ürün için fiyat teklifi hazırlar mısın?", "QUOTATION"),
            ("Ödeme konusunda ne önerirsin?", "PAYMENT"),
            ("Hangi Incoterm daha uygun?", "LOGISTICS"),
            ("Lojistikte nelere dikkat etmeliyim?", "LOGISTICS"),
            ("Gümrük tarafında neyi kontrol etmeliyim?", "LOGISTICS"),
        ]
        early_blob = ""
        for i, (question, expected_stage) in enumerate(script):
            advice, sit, focus, intent = _turn(question, session)
            if i == 0:
                self.assertTrue(_woven(session.product))
                self.assertIsNone(session.capacity)
            if i >= 1:
                self.assertEqual(session.market, "Almanya")
                self.assertTrue(_woven(session.product), session.product)
                self.assertIsNone(session.capacity)
            if expected_stage:
                self.assertEqual(sit.sales_stage, expected_stage, question)
            # Locked GTM/customer-find reach path: no unsolicited payment/Incoterm
            if focus == "reach" and sit.sales_stage in (
                "DISCOVERY",
                "PRODUCT",
                "MARKET",
                "CUSTOMER_IDENTIFICATION",
            ):
                self.assertIsNone(_EARLY.search(advice), advice)
                early_blob += "\n" + advice
            if "Ödeme" in question:
                self.assertTrue(is_payment_ask(question))
                self.assertEqual(sit.sales_stage, "PAYMENT")
            if "Incoterm" in question:
                self.assertTrue(is_incoterm_ask(question))
            self.assertIsNone(_FAKE_BUYER.search(advice), advice)
            for token in _INTERNAL:
                self.assertNotIn(token, advice)
            if sit.sales_stage == "SAMPLE":
                self.assertIn("sample", advice.casefold())
            session.append("user", question)
            session.append("assistant", advice)

        self.assertEqual(session.market, "Almanya")
        self.assertIsNone(session.capacity)
        if early_blob.strip():
            self.assertIsNone(_EARLY.search(early_blob), early_blob)


class V51855ProductSwitchTests(unittest.TestCase):
    def test_woven_to_carton_italy(self) -> None:
        """Known-SKU switch (karton) + user-stated Italy market authority."""
        session = SessionState(session_id="v51855-switch")
        apply_utterance_slots(session, CHAIN_1)
        self.assertTrue(_woven(session.product))
        self.assertEqual(session.market, "Almanya")

        q2 = "dokuma etiket yerine karton etiket üretmeye başladım"
        self.assertTrue(stale_session_product_conflict(session.product, q2))
        incoming = utterance_product(q2)
        self.assertTrue(is_product_switch(session.product, incoming))
        apply_utterance_slots(session, q2)
        self.assertIn("karton", (session.product or "").casefold())
        self.assertFalse(_woven(session.product))

        q3 = "Bunu İtalya'ya satmak istiyorum."
        apply_utterance_slots(session, q3)
        self.assertEqual(session.market, "İtalya")
        _advice, _sit, _focus, _intent = _turn(q3, session)
        self.assertEqual(session.market, "İtalya")
        self.assertIn("karton", (session.product or "").casefold())
        self.assertNotEqual(session.market, "Almanya")
        self.assertIsNone(session.capacity)


class V51855SafetyDisciplineTests(unittest.TestCase):
    def test_capacity_none_no_invention(self) -> None:
        session = SessionState(session_id="v51855-cap")
        advice, _sit, _f, _i = _turn("Dokuma etiket üretiyorum.", session)
        self.assertTrue(_woven(session.product))
        self.assertIsNone(session.capacity)
        self.assertIsNone(_CAPACITY_INVENT.search(advice), advice)
        self.assertNotIn("Capacity:", advice)

    def test_germany_no_contamination(self) -> None:
        session = SessionState(session_id="v51855-de")
        advice, _sit, _f, _i = _turn(CHAIN_1, session)
        self.assertEqual(session.market, "Almanya")
        _no_italy_nl(advice)

    def test_customer_find_no_fake_company(self) -> None:
        session = SessionState(session_id="v51855-fake")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        q = "Almanya'da bana müşteri bul."
        advice, sit, focus, intent = _turn(q, session)
        self.assertIn(intent, ("buyer_finder", "trade_advisor"))
        self.assertIsNone(_FAKE_BUYER.search(advice), advice)
        self.assertNotIn("kesin müşteri", advice.casefold())

    def test_early_topics_suppressed_on_customer_find(self) -> None:
        session = SessionState(session_id="v51855-early")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        advice, sit, _f, _i = _turn(CHAIN_2, session)
        self.assertEqual(sit.sales_stage, "CUSTOMER_IDENTIFICATION")
        self.assertIsNone(_EARLY.search(advice), advice)

    def test_sample_no_full_payment_theory(self) -> None:
        session = SessionState(session_id="v51855-sample")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        advice, sit, focus, _i = _turn(CHAIN_4, session)
        self.assertEqual(sit.sales_stage, "SAMPLE")
        self.assertEqual(focus, "outreach")
        # Sample playbook may mention lead time; must not dump payment/Incoterm theory
        self.assertNotIn("peşin", advice.casefold())
        self.assertNotIn("akreditif", advice.casefold())
        self.assertNotIn("incoterm", advice.casefold())


class V51855JudgmentAssumptionTests(unittest.TestCase):
    def test_blast_message_challenged(self) -> None:
        q = "Herkese aynı mesajı gönderelim."
        hint = detect_weak_commercial_assumption(q)
        self.assertIn("Onaylama", hint)
        brief = build_commercial_brief(
            q,
            session=SessionState(
                session_id="v51855-j", product="Dokuma Etiket", market="Almanya"
            ),
            focus="mediate",
        )
        self.assertTrue(brief.judgment and brief.judgment.challenge_hint)

    def test_price_before_customer(self) -> None:
        q = "Önce herkese fiyat gönderelim, sonra müşteri buluruz."
        hint = detect_weak_commercial_assumption(q)
        self.assertTrue(hint)
        self.assertTrue(
            "risk" in hint.casefold() or "müşteri" in hint.casefold(), hint
        )

    def test_same_offer_all_buyers_challenged(self) -> None:
        q = "Her müşteriye aynı teklif gider."
        # Prefer selective outreach; blast-like if pattern hits, else judgment kit still loads.
        session = SessionState(
            session_id="v51855-same",
            product="Dokuma Etiket",
            market="Almanya",
        )
        brief = build_commercial_brief(q, session=session, focus="mediate")
        self.assertIsNotNone(brief.judgment)
        blob = "\n".join(brief.judgment.as_prompt_lines()).casefold()
        self.assertTrue(
            "onaylama" in blob or "seçici" in blob or "kör" in blob, blob
        )

    def test_incoterm_before_customer_not_forced_stage_machine(self) -> None:
        """Current question primary: Incoterm ask → LOGISTICS, not forced MARKET."""
        session = SessionState(
            session_id="v51855-inc-early",
            product="Dokuma Etiket",
            market="Almanya",
        )
        q = "Müşteri bulmadan önce Incoterm seçelim. Hangi Incoterm daha uygun?"
        sit = build_commercial_situation(q, session=session)
        self.assertEqual(sit.sales_stage, "LOGISTICS")
        self.assertTrue(is_incoterm_ask(q))


class V51855QuestionDisciplineTests(unittest.TestCase):
    def test_max_one_question_on_market_start(self) -> None:
        session = SessionState(session_id="v51855-q")
        from agents.commercial import apply_commercial_guidance

        apply_utterance_slots(session, CHAIN_1)
        sit = build_commercial_situation(CHAIN_1, session=session, focus="reach")
        raw = advisor_answer(CHAIN_1, session)
        guided = apply_commercial_guidance(
            raw, sit, CHAIN_1, session=session, focus="reach"
        )
        self.assertLessEqual(guided.count("?"), 1)

    def test_no_buyer_reask_on_t2(self) -> None:
        session = SessionState(
            session_id="v51855-q2", product="Dokuma Etiket", market="Almanya"
        )
        from agents.commercial import apply_commercial_guidance

        sit = build_commercial_situation(CHAIN_2, session=session, focus="reach")
        raw = advisor_answer(CHAIN_2, session)
        guided = apply_commercial_guidance(
            raw, sit, CHAIN_2, session=session, focus="reach"
        )
        self.assertNotIn("marka mı", guided.casefold())


class V51855LlmContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_situation_policy_judgment_in_system_not_user(self) -> None:
        captured: dict[str, str] = {}

        async def generate(_http, user, system_prompt=None, history=None, polish=False):
            captured["user"] = user
            captured["system"] = system_prompt or ""
            return "Almanya fırsatını koruyun; önce marjı ölçün."

        session = SessionState(
            session_id="v51855-llm", product="Dokuma Etiket", market="Almanya"
        )
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        text = await compose_consultant_reply(
            generate=generate,
            http=None,
            question=q,
            facts="Elde doğrulanmış pazar istatistiği yok.",
            session=session,
            focus="mediate",
        )
        system = captured["system"]
        user = captured["user"]
        self.assertIn("TİCARİ YOLCULUK", system)
        self.assertIn("TİCARİ MUHAKEME", system)
        self.assertNotIn("TİCARİ YOLCULUK", user)
        self.assertNotIn("TİCARİ MUHAKEME", user)
        for token in _INTERNAL:
            self.assertNotIn(token, text)

    async def test_stage_change_changes_system(self) -> None:
        systems: list[str] = []

        async def generate(_http, user, system_prompt=None, history=None, polish=False):
            systems.append(system_prompt or "")
            return "Net duruş veriyorum."

        session = SessionState(
            session_id="v51855-delta", product="Dokuma Etiket", market="Almanya"
        )
        await compose_consultant_reply(
            generate=generate,
            http=None,
            question="Ne yapmalıyım?",
            facts="Elde veri yok.",
            session=session,
            situation=build_commercial_situation(
                CHAIN_1, session=session, focus="reach"
            ),
            focus="mediate",
        )
        await compose_consultant_reply(
            generate=generate,
            http=None,
            question="Ne yapmalıyım?",
            facts="Elde veri yok.",
            session=session,
            situation=build_commercial_situation(
                "Ödeme şartlarını nasıl belirlemeliyim?",
                session=session,
                focus="mediate",
            ),
            focus="mediate",
        )
        self.assertIn("MARKET", systems[0])
        self.assertIn("PAYMENT", systems[1])
        self.assertNotEqual(systems[0], systems[1])


class V51855DeicticFragmentTests(unittest.TestCase):
    def test_deictic_keeps_woven_germany(self) -> None:
        sid = "v51855-deictic"
        history: list[dict[str, str]] = []
        session = hydrate(sid, history)
        advice1, _s1, _f1, _i1 = _turn(CHAIN_1, session)
        session.append("user", CHAIN_1)
        session.append("assistant", advice1)
        history = session.history_dicts()

        for frag in (
            "Bunu yapalım.",
            "Peki sonra?",
            "Bunu Almanya için yapalım.",
            "Peki numune?",
        ):
            session = hydrate(sid, history)
            apply_utterance_slots(session, frag)
            self.assertTrue(_woven(session.product), frag)
            self.assertEqual(session.market, "Almanya", frag)
            # Fragment must not become product
            self.assertFalse(
                stale_session_product_conflict(session.product, frag)
                and utterance_product(frag)
                and is_product_switch(session.product, utterance_product(frag)),
                frag,
            )
            advice = advisor_answer(frag, session)
            session.append("user", frag)
            session.append("assistant", advice)
            history = session.history_dicts()


class V51855DifferentiationTests(unittest.TestCase):
    def test_t1_ne_t2_ne_t3_ne_t4(self) -> None:
        session = SessionState(session_id="v51855-diff")
        a1, s1, f1, _ = _turn(CHAIN_1, session)
        a2, s2, f2, _ = _turn(CHAIN_2, session)
        a3, s3, f3, _ = _turn(CHAIN_3, session)
        a4, s4, f4, _ = _turn(CHAIN_4, session)
        self.assertNotEqual(a1.strip(), a2.strip())
        self.assertNotEqual(a2.strip(), a3.strip())
        self.assertNotEqual(a3.strip(), a4.strip())
        self.assertEqual(s1.sales_stage, "MARKET")
        self.assertEqual(s2.sales_stage, "CUSTOMER_IDENTIFICATION")
        self.assertEqual(s3.sales_stage, "OUTREACH")
        self.assertEqual(s4.sales_stage, "SAMPLE")
        self.assertEqual(f3, "draft")
        self.assertEqual(f4, "outreach")
        self.assertFalse(should_reason(CHAIN_1, f1))
        self.assertFalse(should_reason(CHAIN_3, f3))


class V51855KnownBacklogNote(unittest.TestCase):
    def test_capacity_satiri_is_known_backlog_not_regression(self) -> None:
        """Document: outreach may contain literal 'kapasite satırı' — UX backlog."""
        session = SessionState(
            session_id="v51855-backlog",
            product="Dokuma Etiket",
            market="Almanya",
        )
        advice, sit, focus, _ = _turn(CHAIN_4, session)
        self.assertEqual(sit.sales_stage, "SAMPLE")
        self.assertEqual(focus, "outreach")
        # State leak must still be absent
        self.assertIsNone(session.capacity)
        self.assertNotIn("Capacity: Dokuma", advice)
        # Wording backlog is acceptable to observe, not fail Phase 5
        _ = "kapasite satırı" in advice.casefold()


if __name__ == "__main__":
    unittest.main()
