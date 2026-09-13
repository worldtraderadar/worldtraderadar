"""V5.18.5 Phase 4: CommercialJudgmentKit persona / judgment (LLM brief)."""

from __future__ import annotations

import re
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import (
    build_commercial_brief,
    build_commercial_judgment_kit,
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


class V51854JudgmentKitTests(unittest.TestCase):
    def test_judgment_in_brief_prompt(self) -> None:
        session = SessionState(session_id="v51854-kit")
        apply_utterance_slots(session, CHAIN_1)
        brief = build_commercial_brief(CHAIN_1, session=session, focus="reach")
        self.assertIsNotNone(brief.judgment)
        prompt = brief.as_prompt()
        self.assertIn("TİCARİ MUHAKEME", prompt)
        self.assertIn("TİCARİ YOLCULUK", prompt)
        self.assertIn("sıcak", prompt.casefold())
        self.assertIn("kanıtsız", prompt.casefold())
        self.assertTrue(any("Ton:" in ln for ln in prompt.splitlines()))
        self.assertLess(len(brief.judgment.as_prompt_lines()), 16)

    def test_stage_maturity_differs(self) -> None:
        session = SessionState(session_id="v51854-mat")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        m = build_commercial_judgment_kit(
            build_commercial_situation(CHAIN_1, session=session, focus="reach"),
            question=CHAIN_1,
        )
        c = build_commercial_judgment_kit(
            build_commercial_situation(CHAIN_2, session=session, focus="reach"),
            question=CHAIN_2,
        )
        s = build_commercial_judgment_kit(
            build_commercial_situation(CHAIN_4, session=session, focus="outreach"),
            question=CHAIN_4,
        )
        self.assertNotEqual(m.maturity_hint, c.maturity_hint)
        self.assertNotEqual(c.maturity_hint, s.maturity_hint)
        self.assertIn("Numune", s.maturity_hint)

    def test_blast_assumption_detected(self) -> None:
        hint = detect_weak_commercial_assumption("Herkese aynı mesajı gönderelim.")
        self.assertIn("kişiselleştirilmiş", hint.casefold())
        session = SessionState(session_id="v51854-blast")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        sit = build_commercial_situation(
            "Herkese aynı mesajı gönderelim.", session=session, focus="mediate"
        )
        kit = build_commercial_judgment_kit(
            sit, question="Herkese aynı mesajı gönderelim."
        )
        self.assertTrue(kit.challenge_hint)
        self.assertIn("Onaylama", kit.challenge_hint)


class V51854LlmContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_judgment_enters_system_context(self) -> None:
        captured: dict[str, str] = {}

        async def generate(_http, user, system_prompt=None, history=None, polish=False):
            captured["user"] = user
            captured["system"] = system_prompt or ""
            return (
                "Almanya fırsatını bırakmazdım. Önce marjı ölçün; "
                "Fransa ikinci test olsun."
            )

        session = SessionState(session_id="v51854-llm")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
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
        system = captured.get("system") or ""
        user = captured.get("user") or ""
        self.assertIn("TİCARİ MUHAKEME", system)
        self.assertIn("TİCARİ YOLCULUK", system)
        self.assertNotIn("TİCARİ MUHAKEME", user)
        self.assertNotIn("TİCARİ MUHAKEME", text)
        self.assertNotIn("maturity_hint", text)

    async def test_different_stages_different_judgment_context(self) -> None:
        systems: list[str] = []

        async def generate(_http, user, system_prompt=None, history=None, polish=False):
            systems.append(system_prompt or "")
            return "Net duruş: seçici ilerleyin."

        session = SessionState(session_id="v51854-delta")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        await compose_consultant_reply(
            generate=generate,
            http=None,
            question="Fiyat baskısı yüksek; ne yapmalıyım?",
            facts="Elde doğrulanmış veri yok.",
            session=session,
            situation=build_commercial_situation(
                CHAIN_1, session=session, focus="reach"
            ),
            focus="mediate",
        )
        await compose_consultant_reply(
            generate=generate,
            http=None,
            question="Fiyat baskısı yüksek; ne yapmalıyım?",
            facts="Elde doğrulanmış veri yok.",
            session=session,
            situation=build_commercial_situation(
                CHAIN_4, session=session, focus="outreach"
            ),
            focus="mediate",
        )
        self.assertEqual(len(systems), 2)
        self.assertIn("MARKET", systems[0])
        self.assertIn("SAMPLE", systems[1])
        self.assertIn("Numune", systems[1])
        self.assertNotEqual(systems[0], systems[1])


class V51854MultiTurnTests(unittest.TestCase):
    def test_product_then_market(self) -> None:
        session = SessionState(session_id="v51854-mt-a")
        apply_utterance_slots(session, "Dokuma etiket üretiyorum.")
        self.assertTrue(_woven(session.product))
        apply_utterance_slots(session, "Almanya'ya satmak istiyorum.")
        self.assertEqual(session.market, "Almanya")
        sit = build_commercial_situation(
            "Almanya'ya satmak istiyorum.", session=session, focus="reach"
        )
        brief = build_commercial_brief(
            "Almanya'ya satmak istiyorum.",
            session=session,
            situation=sit,
            focus="reach",
        )
        self.assertIn(sit.sales_stage, ("MARKET", "PRODUCT"))
        self.assertIsNotNone(brief.judgment)
        self.assertIn("Pazar", brief.judgment.maturity_hint)

    def test_market_then_customer(self) -> None:
        session = SessionState(session_id="v51854-mt-b")
        session.product = "Dokuma Etiket"
        apply_utterance_slots(session, "Almanya'ya satmak istiyorum.")
        q2 = "Kimlere satabilirim?"
        sit = build_commercial_situation(q2, session=session, focus="reach")
        self.assertEqual(sit.sales_stage, "CUSTOMER_IDENTIFICATION")
        kit = build_commercial_judgment_kit(sit, question=q2)
        self.assertIn("Müşteri keşfi", kit.maturity_hint)

    def test_customer_then_outreach(self) -> None:
        session = SessionState(session_id="v51854-mt-c")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        q1 = "Almanya'da müşterileri bulmak istiyorum."
        sit1 = build_commercial_situation(q1, session=session, focus="reach")
        self.assertEqual(sit1.sales_stage, "CUSTOMER_IDENTIFICATION")
        q2 = "İlk mesajı nasıl yazalım?"
        # Draft-shaped outreach
        sit2 = build_commercial_situation(
            "İlk müşteriye göndereceğim mesajı hazırlar mısın?",
            session=session,
            focus="draft",
        )
        self.assertEqual(sit2.sales_stage, "OUTREACH")
        self.assertNotEqual(sit1.sales_stage, sit2.sales_stage)

    def test_assumption_not_rubber_stamp(self) -> None:
        q = "Herkese aynı mesajı gönderelim."
        hint = detect_weak_commercial_assumption(q)
        self.assertTrue(hint)
        self.assertIn("Onaylama", hint)
        self.assertIn("kişiselleştirilmiş", hint.casefold())

    def test_no_early_commercial_checklist_in_customer_brief(self) -> None:
        session = SessionState(session_id="v51854-mt-e")
        session.product = "Dokuma Etiket"
        session.market = "Almanya"
        brief = build_commercial_brief(CHAIN_2, session=session, focus="reach")
        topics = brief.policy.allowed_topics if brief.policy else ()
        for banned in ("payment_terms", "incoterm", "customs", "freight"):
            self.assertNotIn(banned, topics)
        self.assertIsNone(_EARLY_TOPIC.search(brief.judgment.maturity_hint))


class V51854ChainRegressionTests(unittest.TestCase):
    def test_t1_t4_locked_untouched(self) -> None:
        sid = "v51854-chain"
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
            self.assertFalse(should_reason(question, focus))
            self.assertIsNotNone(brief.judgment)

            if turn == 1:
                self.assertEqual(intent, "trade_advisor")
                self.assertEqual(sit.sales_stage, "MARKET")
                self.assertIn("giyim", advice.casefold())
                t1 = advice
            elif turn == 2:
                self.assertEqual(sit.sales_stage, "CUSTOMER_IDENTIFICATION")
                self.assertNotEqual(advice.strip(), t1.strip())
            elif turn == 3:
                self.assertEqual(focus, "draft")
                self.assertIn("purchasing team", advice.casefold())
            else:
                self.assertEqual(focus, "outreach")
                self.assertIn("sample from turkey", advice.casefold())

            session.append("user", question)
            session.append("assistant", advice)
            history = session.history_dicts()


if __name__ == "__main__":
    unittest.main()
