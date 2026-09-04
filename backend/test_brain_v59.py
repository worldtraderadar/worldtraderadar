"""V5.9 real-decision stance. Evaluator/validator/routing değiştirmez."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import build_commercial_brief
from agents.contracts import CMO_CONTRACT, DIAGNOSTIC_CONTRACT
from agents.evaluator import score_cmo_v51
from agents.validator import validate_response
from benchmark_v57_anatomy import provisional_stance
from llm import QWEN_NUM_PREDICT, chat_model, reasoning_model, reasoning_think

V571_LINE = (
    "Veri eksik olsa bile, kesinlik iddiasında bulunmadan geçici bir ticari duruş ver"
)
V59_REASK = "Kullanıcının verdiği ticari ayrımı"
V59_PRICE = "Marj yokken şimdilik fiyatı artırma"
EVALUATOR = Path(__file__).resolve().parent / "agents" / "evaluator.py"

GERMANY_Q = (
    "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
)
GERMANY_GOLDEN = (
    "Almanya'daki üç potansiyel müşteriyi hemen bırakmazdım. "
    "Fiyat baskısı senin gözlemin; henüz ölçülmüş marj verisi değil. "
    "Önce bu üç lead'de fiyatın gerçekten kabul edilemez marja inip inmediğini ölçelim. "
    "Üçünde de aynı tablo çıkarsa Fransa'yı ikinci test pazarı olarak açarız. "
    "Kararı değiştiren veri: üçünde de marj eşiğinin altına inmesi. "
    "Sonraki adım: Nordöl Import ve Gastro Nord için teklif ve hacmi netleştirmek."
)


class V59RealDecisionTests(unittest.TestCase):
    def test_v571_line_kept_and_v59_added(self) -> None:
        self.assertIn(V571_LINE, CMO_CONTRACT)
        self.assertIn("daha sonra belirleriz", CMO_CONTRACT)
        self.assertEqual(CMO_CONTRACT.count("geçici bir ticari duruş"), 1)
        self.assertIn(V59_REASK, CMO_CONTRACT)
        self.assertIn(V59_PRICE, CMO_CONTRACT)
        self.assertIn("verdiyse tekrar sorma", DIAGNOSTIC_CONTRACT)
        self.assertIn("yalnız soruyla bitirme", DIAGNOSTIC_CONTRACT)

    def test_evaluator_keeps_legacy_decision_verbs(self) -> None:
        src = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("bırakmaz", src)
        self.assertIn("düşürmez", src)
        self.assertIn("öncelik|önceliğ", src)
        self.assertIn("ikinci (bir )?test", src)
        self.assertIn("hemen temas", src)

    def test_production_routing_untouched(self) -> None:
        os.environ.pop("OLLAMA_REASONING_MODEL", None)
        self.assertEqual(reasoning_model(), chat_model())
        self.assertFalse(reasoning_think())
        self.assertEqual(QWEN_NUM_PREDICT, 280)

    def test_given_split_is_not_reasked_in_brief(self) -> None:
        q = "Yeni müşteri gelmiyor ama mevcut müşteriler sipariş veriyor. Nereye bakayım?"
        brief = build_commercial_brief(q)
        blob = brief.as_prompt().casefold()
        self.assertEqual(brief.mode, "diagnostic")
        self.assertFalse(brief.critical_questions)
        self.assertNotIn("son iki aydaki düşüş", blob)
        self.assertIn("tekrar sorma", blob)
        self.assertIn("koru", (brief.stance or "").casefold())

    def test_unknown_split_still_asks(self) -> None:
        q = "Satışlarımız düştü, neden?"
        brief = build_commercial_brief(q)
        self.assertEqual(brief.mode, "diagnostic")
        self.assertTrue(brief.critical_questions)
        self.assertIn("mevcut", brief.critical_questions[0].casefold())

    def test_raise_price_provisional_example_is_stance(self) -> None:
        good = (
            "Şu aşamada fiyatı artırmazdım. Önce minimum marjı ölçerdim; "
            "marj hedefin altındaysa artış tarafına geçerdim."
        )
        defer = (
            "Fiyatı artırmak için veri yok. En doğru fiyat stratejisini belirleriz."
        )
        g = provisional_stance(good)
        d = provisional_stance(defer)
        self.assertTrue(g["stance"] and g["ok"])
        self.assertTrue(d["defer"] or not d["ok"])

    def test_validator_still_rejects_invented_numbers(self) -> None:
        self.assertEqual(
            validate_response(
                "Marj yok ama fiyatı %12 artırın.",
                facts="Kayıtlı marj yok.",
            ).status,
            "REJECT",
        )
        self.assertEqual(
            validate_response(
                "Hacmi 400 ton varsayıp CIF 3.2 euro verin.",
                facts="Hacim ve fiyat kayıtlı değil. Web yok.",
            ).status,
            "REJECT",
        )

    def test_germany_golden_and_false_certainty(self) -> None:
        self.assertEqual(validate_response(GERMANY_GOLDEN, question=GERMANY_Q).status, "PASS")
        scored = score_cmo_v51(GERMANY_GOLDEN, question=GERMANY_Q, facts="Almanya 3 müşteri")
        self.assertGreaterEqual(scored.total, 10)
        self.assertEqual(
            validate_response("Fransa kesin daha kârlı.", question=GERMANY_Q).status,
            "REJECT",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
