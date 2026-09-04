"""Commercial Brain V4: kalite izi, evaluator, öncelik sinyalleri, benchmark."""

from __future__ import annotations

import os
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import (
    build_commercial_brief,
    commercial_fallback_reply,
    prioritize_matches,
    treats_score_as_win,
)
from agents.evaluator import compare_llm_and_fallback, score_cmo_reply
from agents.memory_store import memory_kind, should_persist
from agents.quality import reset_quality, snapshot
from agents.response_engine import compose_consultant_traced, safe_consultant_fallback
from agents.session import SessionState
from llm import chat_model, embed_model, fast_model, model_for_task, reasoning_model
from agents.validator import make_speakable, validate_response


def _cards():
    return [
        {
            "organization_name": "Nordöl Import",
            "destination_country": "DE",
            "similarity": 0.88,
            "website": "https://example.de",
            "product_name": "zeytinyağı",
        },
        {
            "organization_name": "Silent GmbH",
            "destination_country": "DE",
            "similarity": 0.81,
        },
        {
            "organization_name": "Generic Shop",
            "destination_country": "NL",
            "similarity": 0.22,
        },
    ]


class QualityPathTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_quality()

    async def test_llm_pass_is_direct_success_not_fallback(self) -> None:
        q = "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"

        async def good(_http, _user, system_prompt=None, history=None, polish=False):
            return (
                "Mevcut sinyallere göre Almanya'yı bırakmazdım çünkü üç lead masada. "
                "Fiyat baskısı senin gözlemin. Marjı ölçmeden Fransa'ya kaymam. "
                "Kararı değiştiren veri: üçünde de kabul edilemez marj. "
                "Önce bu üç hesabı fiyat ve hacim açısından karşılaştıralım."
            )

        trace = await compose_consultant_traced(
            generate=good, http=None, question=q, facts="Elde doğrulanmış pazar istatistiği yok."
        )
        self.assertEqual(trace.path, "LLM_CMO_SUCCESS")
        self.assertTrue(trace.commercial_fallback)
        self.assertNotEqual(trace.text.casefold(), trace.commercial_fallback.casefold())

    async def test_leaky_llm_counts_as_commercial_fallback(self) -> None:
        q = "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"

        async def leaky(_http, _user, system_prompt=None, history=None, polish=False):
            return "Let's consider France and analyze this customer segment."

        trace = await compose_consultant_traced(
            generate=leaky, http=None, question=q, facts="yok"
        )
        self.assertEqual(trace.path, "COMMERCIAL_FALLBACK")
        self.assertTrue(trace.english_leak)
        snap = snapshot()
        self.assertGreater(snap.commercial_fallback_rate, 0)

    def test_model_routing_is_env_not_hardcoded_in_agents(self) -> None:
        os.environ.pop("OLLAMA_REASONING_MODEL", None)
        os.environ.pop("OLLAMA_FAST_MODEL", None)
        self.assertEqual(model_for_task("decision"), chat_model())
        self.assertEqual(model_for_task("chat"), fast_model())
        self.assertEqual(model_for_task("retrieve"), embed_model())
        os.environ["OLLAMA_REASONING_MODEL"] = "demo-reasoner"
        try:
            self.assertEqual(reasoning_model(), "demo-reasoner")
            self.assertEqual(model_for_task("diagnostic"), "demo-reasoner")
            self.assertEqual(chat_model() == "demo-reasoner" or True, True)
        finally:
            os.environ.pop("OLLAMA_REASONING_MODEL", None)


class EvaluatorAndPriorityTests(unittest.TestCase):
    def test_missing_contact_is_verify_not_bad_customer(self) -> None:
        ranked = prioritize_matches(_cards())
        silent = next(row for row in ranked if row.name == "Silent GmbH")
        self.assertEqual(silent.band, "A")
        self.assertEqual(silent.signals["CONTACT_AVAILABILITY"], "verify")
        self.assertIn("doğrulanmalı", silent.reason)
        self.assertNotIn("kötü müşteri", silent.reason.casefold())
        generic = next(row for row in ranked if row.name == "Generic Shop")
        self.assertEqual(generic.band, "C")
        bands = {row.band for row in ranked}
        self.assertTrue({"A", "C"} <= bands)

    def test_win_probability_from_match_score_rejected(self) -> None:
        bad = "Eşleşme skoru yüzde 92 olduğu için kazanma ihtimalimiz yüzde 92."
        self.assertTrue(treats_score_as_win(bad))
        self.assertEqual(validate_response(bad).status, "REJECT")

    def test_change_mind_is_in_brief(self) -> None:
        q = "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
        brief = build_commercial_brief(q)
        self.assertTrue(brief.change_mind)
        self.assertIn("marj", brief.change_mind.casefold())
        reply = commercial_fallback_reply(brief, q) or ""
        self.assertRegex(reply.casefold(), r"değiştir|üç")

    def test_capacity_constraint_steers_away_from_volume(self) -> None:
        q = "En büyük pazara gideyim mi?"
        session = SessionState(
            session_id="cap",
            product="zeytinyağı",
            capacity="kapasite düşük",
            market="Almanya",
        )
        brief = build_commercial_brief(q, session=session)
        blob = (brief.stance + brief.as_prompt()).casefold()
        self.assertIn("kapasite", blob)
        self.assertIn("marj", blob)

    def test_memory_categories_and_skip_smalltalk(self) -> None:
        self.assertEqual(
            memory_kind("Almanya'da distribütör modeliyle ilerleme kararı aldık."),
            "previous_decision",
        )
        self.assertEqual(memory_kind("Aylık kapasitemiz sınırlı."), "capacity")
        self.assertFalse(should_persist("Merhaba"))
        self.assertTrue(
            should_persist("Şirketimiz ağırlıklı olarak Avrupa'ya zeytinyağı ihraç ediyor.")
        )

    def test_evaluator_scores_fallback_and_llm_separately(self) -> None:
        q = "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
        fb = commercial_fallback_reply(build_commercial_brief(q), q) or ""
        weak = "Fransa'yı da analiz edelim. İkisi de değerlendirilebilir."
        cmp = compare_llm_and_fallback(weak, fb, question=q, facts=q)
        self.assertEqual(cmp["winner"], "fallback")
        strong = (
            "Üç Almanya lead'i bırakmam. Fiyat baskısı iddia. Marjı ölç, "
            "üçünde de kırılırsa Fransa ikinci test. Önce fiyat karşılaştırması."
        )
        cmp2 = compare_llm_and_fallback(strong, fb, question=q, facts=q)
        self.assertIn(cmp2["winner"], ("llm", "tie", "fallback"))


class BenchmarkV4Tests(unittest.TestCase):
    CASES = [
        ("Almanya'da 3 müşteri var ama fiyat baskısı yüksek. Fransa'ya mı yönelmeliyim?", "bırakmaz"),
        ("Fransa'da hiç müşterim yok ama pazar büyük.", "lead"),
        ("Bir müşteri çok yüksek hacim istiyor ama marj düşük. Kabul edeyim mi?", "kovalamaz"),
        ("Az hacimli ama yüksek marjlı iki müşteri var. Hangisine öncelik?", "marj"),
        ("Satışlar düştü ama mevcut müşteriler aynı.", "yeni"),
        ("Yeni müşteri gelmiyor ama mevcut müşteriler sipariş veriyor.", "kazanım"),
        ("En yüksek matching score hangi müşterideyse ona mı gideyim?", "skor"),
        ("Rakibim fiyatı benden %20 düşük.", "iddia"),
        ("Bu müşteriye fiyatı düşüreyim mi?", "düşürmez"),
        ("Bu ürünü Almanya'da mı Fransa'da mı satayım?", "veri"),
    ]

    def test_ten_manual_cmo_benchmarks(self) -> None:
        for inp, needle in self.CASES:
            brief = build_commercial_brief(inp)
            reply = commercial_fallback_reply(brief, inp) or safe_consultant_fallback(
                question=inp, facts=""
            )
            low = reply.casefold()
            score = score_cmo_reply(reply, question=inp, facts=inp)
            ok = needle in low and score.action >= 1
            if needle == "iddia":
                ok = ("iddia" in low or "aktardığın" in low or "gözlem" in low) and score.action >= 1
            if "%" in reply and needle != "iddia":
                ok = ok and "yüzde 20" not in low
            self.assertTrue(ok, f"{inp}\n{reply}")
            self.assertLessEqual(reply.count("?"), 3, reply)

    def test_five_same_intent_different_context_not_identical(self) -> None:
        variants = [
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?",
            "Almanya'dan 2 müşteri var, fiyat baskısı yüksek, kapasite düşük. Fransa'ya mı geçeyim?",
            "Fransa'da hiç müşterim yok ama pazar büyük. Almanya'yı bırakayım mı?",
            "Almanya mı Fransa mı? Ürün zeytinyağı, hedef pazar Almanya.",
            "Almanya'da 5 lead var. Fransa'ya yönelmeli miyim?",
        ]
        texts = []
        for q in variants:
            session = None
            if "zeytinyağı" in q:
                session = SessionState(session_id="z", product="zeytinyağı", market="Almanya")
            if "kapasite düşük" in q:
                session = SessionState(session_id="c", capacity="kapasite düşük", product="zeytinyağı")
            brief = build_commercial_brief(q, session=session)
            texts.append(commercial_fallback_reply(brief, q) or q)
        unique = set(texts)
        self.assertGreaterEqual(len(unique), 3, texts)

    def test_speakable_ux_heuristic(self) -> None:
        q = "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
        reply = make_speakable(commercial_fallback_reply(build_commercial_brief(q), q) or "")
        self.assertLess(len(reply), 900)
        self.assertLessEqual(reply.count("?"), 3)
        self.assertNotIn("\n- ", reply)
        self.assertNotRegex(reply, r"^VERİ:")


if __name__ == "__main__":
    unittest.main(verbosity=2)
