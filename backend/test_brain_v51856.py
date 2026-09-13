"""V5.18.5 targeted fix: persona plan leak + LIVE T3 capacity hallucination."""

from __future__ import annotations

import re
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import build_commercial_brief, _human_plan
from agents.goals import classify_goal, plan_summary
from agents.response_engine import should_reason
from agents.session import SessionState, advisor_answer, apply_utterance_slots
from agents.validator import validate_response
from prompts import advisor_focus, is_draft_request

LIVE_T3 = "Almanya için kısa bir outreach maili yaz. Purchasing Team için olsun."
DRAFT_CLASSIC = "İlk müşteriye göndereceğim mesajı hazırlar mısın?"
DRAFT_MAILI = "Almanya için kısa bir maili yaz."
NON_DRAFT_MAIL = "mail hakkında ne düşünüyorsun?"

_INTERNAL_STANCE = re.compile(
    r"(?i)("
    r"ge[cç]ici\s*duru[sş]|"
    r"k[oö]r\s*kilit|"
    r"mevcut\s*tahsis|"
    r"tahsisi\s+ve\s+fiyat[iı]\s+koru"
    r")"
)
_CAPACITY_INVENT = re.compile(
    r"(?i)("
    r"kapasite(?:miz|niz)?\s*\d+|"
    r"capacity\s*[:=]?\s*\d+|"
    r"\b\d+\s*adet\b"
    r")"
)
_EARLY_PAY = re.compile(
    r"(?i)(\bexw\b|\bcif\b|\bfob\b|pe[sş]in|akreditif|ödeme\s*[sş]art)"
)


def _woven_de_session() -> SessionState:
    session = SessionState(session_id="v51856")
    session.product = "Dokuma etiket"
    session.market = "Almanya"
    session.capacity = None
    session.last_advisor_kind = "reach"
    return session


class PersonaPlanLeakTests(unittest.TestCase):
    def test_human_plan_omits_internal_stance(self) -> None:
        session = _woven_de_session()
        brief = build_commercial_brief(LIVE_T3, session=session)
        self.assertTrue((brief.stance or "").strip())
        self.assertRegex(brief.stance, _INTERNAL_STANCE)
        self.assertIn("geçici duruş", brief.as_prompt().casefold())

        plan = _human_plan(brief.mode, LIVE_T3, session, brief.stance)
        self.assertNotRegex(plan, _INTERNAL_STANCE)
        self.assertIn("Dokuma etiket", plan)
        self.assertIn("Almanya", plan)

    def test_plan_summary_omits_internal_stance(self) -> None:
        session = _woven_de_session()
        plan = plan_summary(classify_goal(LIVE_T3), LIVE_T3, session)
        self.assertNotRegex(plan, _INTERNAL_STANCE)
        brief = build_commercial_brief(LIVE_T3, session=session)
        self.assertRegex(brief.stance, _INTERNAL_STANCE)


class LiveT3DraftRoutingTests(unittest.TestCase):
    def test_live_phrasing_is_draft(self) -> None:
        self.assertTrue(is_draft_request(LIVE_T3))
        self.assertEqual(advisor_focus(LIVE_T3), "draft")
        self.assertFalse(should_reason(LIVE_T3, "draft"))

    def test_maili_yaz_is_draft(self) -> None:
        self.assertTrue(is_draft_request(DRAFT_MAILI))
        self.assertEqual(advisor_focus(DRAFT_MAILI), "draft")

    def test_classic_draft_still_works(self) -> None:
        self.assertTrue(is_draft_request(DRAFT_CLASSIC))
        self.assertEqual(advisor_focus(DRAFT_CLASSIC), "draft")

    def test_non_draft_mail_question_not_locked(self) -> None:
        self.assertFalse(is_draft_request(NON_DRAFT_MAIL))
        self.assertNotEqual(advisor_focus(NON_DRAFT_MAIL), "draft")

    def test_session_sticky_reach_preserved_when_not_draft(self) -> None:
        session = _woven_de_session()
        q = "Peki sonra?"
        self.assertFalse(is_draft_request(q))
        self.assertEqual(advisor_focus(q, session), "reach")

    def test_live_t3_playbook_no_capacity_invent(self) -> None:
        session = _woven_de_session()
        apply_utterance_slots(session, LIVE_T3)
        self.assertIsNone(session.capacity)
        self.assertEqual(advisor_focus(LIVE_T3, session), "draft")
        advice = advisor_answer(LIVE_T3, session, [])
        self.assertIsNone(session.capacity)
        self.assertNotRegex(advice, _CAPACITY_INVENT)
        self.assertIn("Purchasing Team", advice)
        self.assertRegex(advice, re.compile(r"(?i)(dokuma|woven)"))
        self.assertNotRegex(advice, _EARLY_PAY)
        # Deterministic draft omits Capacity line when capacity is None.
        self.assertNotRegex(advice, re.compile(r"(?i)capacity\s*:"))

    def test_known_capacity_still_allowed_in_draft(self) -> None:
        session = _woven_de_session()
        session.capacity = "Aylık 5000 adet"
        advice = advisor_answer(LIVE_T3, session, [])
        self.assertRegex(advice, re.compile(r"(?i)capacity\s*:"))
        self.assertNotRegex(advice, re.compile(r"(?i)kapasitemiz\s*5000"))
        # Canonical foreign capacity note is present (may be "5 bin" style).
        self.assertTrue(
            re.search(r"(?i)(5\s*bin|5000|adet)", advice) is not None,
            advice[:240],
        )


class CapacitySafetyNetTests(unittest.TestCase):
    def test_reject_invented_capacity_when_unset(self) -> None:
        fake = (
            "Almanya'daki purchasing ekibine yazın. "
            "Kapasitemiz 5000 adet ve teslim süresi 2-3 hafta kadardır."
        )
        verdict = validate_response(fake, question=LIVE_T3, capacity=None)
        self.assertEqual(verdict.status, "REJECT")
        self.assertTrue(any("kapasite" in r.casefold() for r in verdict.reasons))

    def test_known_capacity_not_rejected(self) -> None:
        text = "Kapasitemiz 5000 adet. Almanya için numune gönderebiliriz."
        verdict = validate_response(
            text, question=LIVE_T3, capacity="Aylık 5000 adet"
        )
        self.assertNotEqual(verdict.status, "REJECT")


if __name__ == "__main__":
    unittest.main()
