"""V5.18.5 Phase 3: CommercialSituation → CommercialBrief → LLM system context."""

from __future__ import annotations

import re
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import build_commercial_brief, build_commercial_situation
from agents.intent import classify_intent
from agents.response_engine import compose_consultant_reply, should_reason
from agents.session import (
    SessionState,
    advisor_answer,
    apply_utterance_slots,
    hydrate,
)
from prompts import advisor_focus

CHAIN_1 = (
    "Dokuma etiket üretiyorum ve Almanya'ya satmak istiyorum. "
    "Nereden başlamalıyım?"
)
CHAIN_2 = (
    "Almanya'da bu ürünü kimlere satabilirim ve müşterileri nereden bulabilirim?"
)
CHAIN_3 = "İlk müşteriye göndereceğim mesajı hazırlar mısın?"
CHAIN_4 = "Numuneyi nasıl göndereyim?"

DECISION_Q = (
    "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
    "Fransa'ya mı yönelmeliyim?"
)

_INTERNAL = (
    "identify_customer",
    "qualify_buyer",
    "prepare_outreach",
    "send_sample",
    "prepare_quotation",
    "next_best_action",
    "missing_critical",
    "sales_stage",
    "TİCARİ YOLCULUK",
    "TİCARİ ZEKÂ BRİFİ",
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


class V51853BriefIntegrationTests(unittest.TestCase):
    def test_situation_attached_to_brief(self) -> None:
        session = SessionState(session_id="v51853-brief")
        apply_utterance_slots(session, CHAIN_1)
        sit = build_commercial_situation(CHAIN_1, session=session, focus="reach")
        brief = build_commercial_brief(
            CHAIN_1, session=session, situation=sit, focus="reach"
        )
        self.assertIs(brief.situation, sit)
        self.assertEqual(brief.situation.sales_stage, "MARKET")
        self.assertEqual(brief.situation.missing_critical, ["buyer_type"])
        self.assertEqual(brief.situation.next_best_action, "identify_customer")
        self.assertIn("capacity_unset", brief.situation.risk_flags)
        self.assertIsNotNone(brief.policy)

    def test_situation_in_as_prompt_not_user_facing_enums_required(self) -> None:
        session = SessionState(session_id="v51853-prompt")
        apply_utterance_slots(session, CHAIN_1)
        brief = build_commercial_brief(CHAIN_1, session=session, focus="reach")
        prompt = brief.as_prompt()
        self.assertIn("TİCARİ YOLCULUK", prompt)
        self.assertIn("MARKET", prompt)
        self.assertIn("buyer_type", prompt)
        self.assertIn("identify_customer", prompt)
        self.assertIn("Sorulmadıkça açma", prompt)
        self.assertIn("Uydurma yasak", prompt)
        # Natural NBA guidance present for the model
        self.assertIn("hedef müşteri", prompt.casefold())

    def test_reuse_situation_no_second_independent_state(self) -> None:
        session = SessionState(session_id="v51853-reuse")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = build_commercial_situation(CHAIN_2, session=session, focus="reach")
        brief = build_commercial_brief(
            CHAIN_2, session=session, situation=sit, focus="reach"
        )
        self.assertIs(brief.situation, sit)
        self.assertEqual(brief.situation.sales_stage, "CUSTOMER_IDENTIFICATION")
        self.assertEqual(brief.situation.next_best_action, "prepare_outreach")


class V51853LlmContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_situation_enters_system_not_user(self) -> None:
        captured: dict[str, str] = {}

        async def generate(_http, user, system_prompt=None, history=None, polish=False):
            captured["user"] = user
            captured["system"] = system_prompt or ""
            return (
                "Almanya'daki fırsatları bırakmazdım. Önce marjı ölçün; "
                "Fransa ikinci test olsun."
            )

        session = SessionState(session_id="v51853-llm")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = build_commercial_situation(DECISION_Q, session=session)
        text = await compose_consultant_reply(
            generate=generate,
            http=None,
            question=DECISION_Q,
            facts="Elde doğrulanmış pazar istatistiği yok.",
            session=session,
            situation=sit,
            focus="mediate",
        )
        system = captured.get("system") or ""
        user = captured.get("user") or ""
        self.assertTrue(should_reason(DECISION_Q, "mediate"))
        self.assertIn("TİCARİ YOLCULUK", system)
        self.assertIn(sit.sales_stage, system)
        self.assertNotIn("TİCARİ YOLCULUK", user)
        self.assertNotIn("TİCARİ ZEKÂ BRİFİ", user)
        for token in _INTERNAL:
            self.assertNotIn(token, text)

    async def test_situation_changes_system_guidance(self) -> None:
        """Same question facts; different situation must alter system brief."""
        systems: list[str] = []

        async def generate(_http, user, system_prompt=None, history=None, polish=False):
            systems.append(system_prompt or "")
            return "Net duruş: mevcut pazarı koruyun ve marjı ölçün."

        session = SessionState(session_id="v51853-delta")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        q = "Fiyat baskısı yüksek; ne yapmalıyım?"
        sit_market = build_commercial_situation(
            "Almanya'ya satmak istiyorum, nereden başlamalıyım?",
            session=session,
            focus="reach",
        )
        await compose_consultant_reply(
            generate=generate,
            http=None,
            question=q,
            facts="Elde doğrulanmış veri yok.",
            session=session,
            situation=sit_market,
            focus="mediate",
        )
        sit_pay = build_commercial_situation(
            "Ödeme şartlarını nasıl belirlemeliyim?",
            session=session,
            focus="mediate",
        )
        await compose_consultant_reply(
            generate=generate,
            http=None,
            question=q,
            facts="Elde doğrulanmış veri yok.",
            session=session,
            situation=sit_pay,
            focus="mediate",
        )
        self.assertEqual(len(systems), 2)
        self.assertIn("MARKET", systems[0])
        self.assertIn("PAYMENT", systems[1])
        self.assertNotEqual(systems[0], systems[1])

    async def test_locked_reach_skips_llm_unchanged(self) -> None:
        called: list[int] = []

        async def forbidden(_http, user, system_prompt=None, history=None, polish=False):
            called.append(1)
            return "should not run"

        session = SessionState(session_id="v51853-locked")
        apply_utterance_slots(session, CHAIN_1)
        facts = advisor_answer(CHAIN_1, session)
        text = await compose_consultant_reply(
            generate=forbidden,
            http=None,
            question=CHAIN_1,
            facts=facts,
            session=session,
            focus="reach",
        )
        self.assertEqual(called, [])
        self.assertIn("giyim", text.casefold())


class V51853PolicySafetyTests(unittest.TestCase):
    def test_brief_policy_blocks_early_topics(self) -> None:
        session = SessionState(session_id="v51853-pol")
        apply_utterance_slots(session, CHAIN_1)
        brief = build_commercial_brief(CHAIN_1, session=session, focus="reach")
        prompt = brief.as_prompt()
        self.assertIn("Sorulmadıkça açma", prompt)
        topics = brief.policy.allowed_topics if brief.policy else ()
        for banned in ("payment_terms", "incoterm", "customs", "freight"):
            self.assertNotIn(banned, topics)

    def test_no_hallucination_instructions(self) -> None:
        session = SessionState(session_id="v51853-hal")
        apply_utterance_slots(session, CHAIN_1)
        prompt = build_commercial_brief(
            CHAIN_1, session=session, focus="reach"
        ).as_prompt()
        low = prompt.casefold()
        self.assertIn("uydurma yasak", low)
        self.assertIn("firma", low)
        self.assertIn("kapasite", low)


class V51853ChainRegressionTests(unittest.TestCase):
    def test_t1_t4_still_playbook_safe(self) -> None:
        sid = "v51853-chain"
        history: list[dict[str, str]] = []
        t1 = ""
        for turn, question in enumerate((CHAIN_1, CHAIN_2, CHAIN_3, CHAIN_4), start=1):
            session = hydrate(sid, history)
            apply_utterance_slots(session, question)
            intent, _ = classify_intent(question, session)
            focus = advisor_focus(question, session)
            sit = build_commercial_situation(
                question, session=session, focus=focus
            )
            advice = advisor_answer(question, session, matches=[])
            brief = build_commercial_brief(
                question, session=session, situation=sit, focus=focus
            )

            self.assertTrue(_woven(session.product))
            self.assertEqual(session.market, "Almanya")
            self.assertIsNone(session.capacity)
            self.assertNotIn("italya", advice.casefold())
            self.assertNotIn("alıcısı", advice.casefold())
            self.assertIs(brief.situation, sit)
            self.assertFalse(should_reason(question, focus))

            if turn == 1:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(sit.sales_stage, "MARKET")
                self.assertIn("giyim", advice.casefold())
                self.assertIsNone(_EARLY_TOPIC.search(advice))
                t1 = advice
            elif turn == 2:
                self.assertEqual(sit.sales_stage, "CUSTOMER_IDENTIFICATION")
                self.assertNotEqual(advice.strip(), t1.strip())
                self.assertIn("nereden bulunur", advice.casefold())
            elif turn == 3:
                self.assertEqual(sit.sales_stage, "OUTREACH")
                self.assertEqual(focus, "draft")
                self.assertIn("purchasing team", advice.casefold())
            else:
                self.assertEqual(sit.sales_stage, "SAMPLE")
                self.assertEqual(focus, "outreach")
                self.assertIn("sample from turkey", advice.casefold())

            session.append("user", question)
            session.append("assistant", advice)
            history = session.history_dicts()


if __name__ == "__main__":
    unittest.main()
