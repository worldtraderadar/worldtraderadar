"""V5.10 evaluator: geçici ticari duruş matchers. Routing/prompt/canary değişmez."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.evaluator import has_decision_stance, score_cmo_v51

BACKEND = Path(__file__).resolve().parent
V59_JSON = BACKEND / "benchmark_v59_results.json"

# V5.8/V5.9 elle SCORER_MISS sayılan duruşlar (DEC=0 olmamalı).
STANCE_HITS = [
    "Kritik veri gelene kadar fiyatı sabit tutmak en güvenli duruştur.",
    "Mevcut fiyatı koru; marj netleşmeden zam yapma.",
    "Durumunu koru; netleşene kadar mevcut tahsisi tut.",
    "Sabitleme kararı: şimdilik fiyatı sabit tut.",
    "Geçici olarak erteleme duruşu alırım; kanıt gelince açarım.",
    "Bu bir geçici duruş: mevcut müşteri önceliğini koru.",
    "Netleşene kadar koru; marj gelince artışı test ederiz.",
    "Mevcut dengede kal; yeni talebi ikinci sıraya al.",
    "Bu kanalı korumak önceliğiniz olmalı.",
    "Fiyatı hemen düşürmeyin; marj net değil.",
    "Hacmi riske atmanızı önermem.",
    "Fransa'yı ikinci bir test pazarı olarak açarız.",
    "Şu anki verilerle FOB şartını seçin.",
    "Kredi kaydı olmadan bu teklifi kabul etmem.",
    "Bu siparişi kabul etmeyin; kapasite yetmez.",
    "Bu siparişi kabul etmemek en güvenli hamledir.",
    "Fiyat baskısını indirmek yerine marjı korumaya odaklanmalıyız.",
    "Fiyatı hemen düşürmek marjınızı riske atar.",
    "Şu anki kapasitenizle büyük perakende zincirlerine yönelmeyin.",
    "Doğrulanmış marj olmadan mevcut müşteriyi bırakmak risklidir.",
    "Beklemek yerine bu potansiyel müşterileri bu hafta test edin.",
    "CRM kurmadan yeni müşteriye yönelmeyin.",
    "Marj eşiğinin altındaysa kapasiteyi stratejik müşteriye ayırmak daha güvenli olur.",
    "Bu hesabı şimdilik önceliklendirmeyin.",
]

EVASIONS = [
    "Daha sonra bakalım, bilmiyorum.",
    "En doğru fiyat stratejisini belirleriz.",
    "Fransa'yı da analiz edelim. İkisi de değerlendirilebilir.",
    "Veri gelince karar veririz.",
    "Almanya'ya odaklanmak daha güvenli görünse de ürünü bilmem gerekiyor.",
    "İsterseniz o pazara yönelmek isteyebilirsiniz.",
]

FALSE_KAL = [
    "Kalite düşerse marj erir; rakamı uydurmam.",
    "Bu bilgiler gelmeden kesin bir karar vermek ticari olarak temelsiz kalır.",
]


def _dec(text: str, question: str = "", facts: str = "Marj kayıtlı değil.") -> int:
    return int(score_cmo_v51(text, question=question, facts=facts).decision)


class V510EvaluatorStanceTests(unittest.TestCase):
    def test_provisional_price_and_hold_patterns_score(self) -> None:
        for text in STANCE_HITS:
            self.assertTrue(has_decision_stance(text), text)
            self.assertGreaterEqual(_dec(text), 1, text)

    def test_hold_with_reason_is_decision_two(self) -> None:
        text = (
            "Elimde marj yok; bu yüzden kritik veri gelene kadar "
            "fiyatı sabit tutmak en güvenli duruştur."
        )
        self.assertEqual(_dec(text), 2)

    def test_evasions_are_not_decisions(self) -> None:
        for text in EVASIONS:
            self.assertFalse(has_decision_stance(text), text)
            self.assertEqual(_dec(text), 0, text)

    def test_kalite_and_kalir_are_not_false_positives(self) -> None:
        for text in FALSE_KAL:
            self.assertFalse(has_decision_stance(text), text)
            self.assertEqual(_dec(text), 0, text)

    def test_kabul_etmemiz_is_not_reject_stance(self) -> None:
        text = "Bu şartları kabul etmemiz gerekir; marjı sonra konuşuruz."
        self.assertFalse(has_decision_stance(text), text)

    def test_v59_logs_scorer_miss_falls(self) -> None:
        if not V59_JSON.exists():
            self.skipTest("benchmark_v59_results.json yok")
        data = json.loads(V59_JSON.read_text(encoding="utf-8"))
        rows = (data.get("now") or {}).get("rows") or []
        self.assertEqual(len(rows), 32)
        # Elle SCORER_MISS / REAL_OK duruş metinleri DEC>=1 olmalı.
        expect_hit = {
            "raise_price",
            "new_buyer_priority",
            "sales_drop",
            "discount",
            "collection_risk",
            "incoterms",
            "germany_france",
            "lost_quote",
            "accept_order",
            "fair_leads",
            "capacity_split",
            "cheap_price_market",
            "capacity",
            "keep_customer",
            "today_plan",
            "repeat_orders",
            "mix_priority",
        }
        # Hâlâ duruş yok sayılanlar DEC şişmemeli.
        expect_zero = {
            "uncertain_data",
            "supplier_choice",
            "distributor_pick",
            "market_choice",
            "moq",
        }
        misses = []
        inflated = []
        for row in rows:
            cid = row.get("id")
            text = row.get("final") or ""
            dec = _dec(text, question=row.get("input") or "", facts=row.get("context") or "")
            if cid in expect_hit and dec < 1:
                misses.append((cid, dec, text[:120]))
            if cid in expect_zero and dec > 0:
                inflated.append((cid, dec, text[:120]))
        self.assertFalse(misses, misses)
        self.assertFalse(inflated, inflated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
