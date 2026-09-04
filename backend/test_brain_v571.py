"""V5.7.1 stance contract. Production evaluator/validator/routing değiştirmez."""

from __future__ import annotations

import os
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.contracts import CMO_CONTRACT
from agents.evaluator import score_cmo_v51
from agents.validator import validate_response
from benchmark_v57_anatomy import provisional_stance
from llm import QWEN_NUM_PREDICT, chat_model, reasoning_model, reasoning_think

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
STANCE_LINE = (
    "Veri eksik olsa bile, kesinlik iddiasında bulunmadan geçici bir ticari duruş ver"
)


class V571StanceContractTests(unittest.TestCase):
    def test_one_line_in_cmo_contract_only(self) -> None:
        self.assertIn(STANCE_LINE, CMO_CONTRACT)
        self.assertIn("daha sonra belirleriz", CMO_CONTRACT)
        self.assertEqual(CMO_CONTRACT.count("geçici bir ticari duruş"), 1)

    def test_production_routing_untouched(self) -> None:
        os.environ.pop("OLLAMA_REASONING_MODEL", None)
        self.assertEqual(reasoning_model(), chat_model())
        self.assertFalse(reasoning_think())
        self.assertEqual(QWEN_NUM_PREDICT, 280)

    def test_validator_rejects_invented_percent_hike(self) -> None:
        bad = "Marj verisini bilmiyoruz ama fiyatı %12 artırın."
        self.assertEqual(validate_response(bad, facts="Kayıtlı marj yok.").status, "REJECT")

    def test_validator_rejects_invented_volume_and_price(self) -> None:
        self.assertEqual(
            validate_response(
                "Hacmi 400 ton varsayıp CIF 3.2 euro verin.",
                facts="Hacim ve fiyat kayıtlı değil. Web yok.",
            ).status,
            "REJECT",
        )

    def test_provisional_stance_good_vs_defer(self) -> None:
        good = (
            "Şu aşamada fiyatı artırmazdım. Önce minimum marjı ölçerdim; "
            "marj hedefin altındaysa artış tarafına geçerdim."
        )
        defer = "Fiyat artışının doğru olup olmadığını bilmek için veri gerekir. Stratejisini daha sonra belirleriz."
        g = provisional_stance(good)
        d = provisional_stance(defer)
        self.assertTrue(g["stance"] and g["flip_condition"] and g["ok"])
        self.assertTrue(d["defer"] or not d["ok"])

    def test_germany_golden_still_pass(self) -> None:
        self.assertEqual(validate_response(GERMANY_GOLDEN, question=GERMANY_Q).status, "PASS")
        scored = score_cmo_v51(GERMANY_GOLDEN, question=GERMANY_Q, facts="Almanya 3 müşteri")
        self.assertGreaterEqual(scored.total, 10)

    def test_false_certainty_still_rejected(self) -> None:
        self.assertEqual(
            validate_response("Fransa kesin daha kârlı.", question=GERMANY_Q).status,
            "REJECT",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
