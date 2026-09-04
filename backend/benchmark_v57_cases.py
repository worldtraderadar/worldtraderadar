"""V5.7 diagnostic scenarios. Production CASES (20) değişmez. Yeni dünya verisi yok."""

from __future__ import annotations

from benchmark_v51_cases import BUYER_CARDS, CASES, FAIR_FACTS

COMPANY = (
    "ŞİRKET PROFİLİ: ürün zeytinyağı; kapasite düşük / 20 ton ay. "
    "Kayıtlı marj eşiği yok. Web bu turda yok."
)
CRM_THIN = "CRM yok. Hacim/fiyat kırılımı yok. Web yok."
CREDIT_THIN = "Kredi/tahsilat kaydı yok. Marj eşiği kayıtlı değil. Web yok."

_CAT_20 = {
    "market_choice": "B",
    "germany_france": "B",
    "price_pressure": "C",
    "raise_price": "C",
    "discount": "C",
    "margin_drop": "C",
    "sales_drop": "A",
    "new_buyer_priority": "A",
    "buyer_abc": "D",
    "supplier_choice": "B",
    "capacity": "E",
    "volume_vs_margin": "C",
    "competitor_pressure": "C",
    "new_country": "B",
    "keep_customer": "B",
    "collection_risk": "D",
    "moq": "C",
    "incoterms": "E",
    "today_plan": "F",
    "uncertain_data": "B",
}

NEW_CASES: list[dict] = [
    {
        "id": "quote_conversion",
        "category": "A",
        "input": "Teklif sayısı aynı ama dönüşüm düştü. Neden?",
        "context": CRM_THIN,
        "expected": "fiyat mı hedef mi takip; uydurma funnel yok",
    },
    {
        "id": "churn_risk",
        "category": "A",
        "input": "Almanya'daki mevcut müşteri siparişi kesti. Yeni ülkeye mi geçeyim?",
        "context": FAIR_FACTS,
        "expected": "kaybı teşhis et; hemen pazar değiştirme",
    },
    {
        "id": "repeat_orders",
        "category": "A",
        "input": "İlk sipariş geliyor ama tekrar sipariş düşük. Ne yapayım?",
        "context": CRM_THIN + " ŞİRKET PROFİLİ: ürün zeytinyağı.",
        "expected": "tekrar sipariş deliği; reklam dayatma yok",
    },
    {
        "id": "distributor_pick",
        "category": "B",
        "input": "Almanya'da iki distribütör var, hangisini seçeyim?",
        "context": (
            FAIR_FACTS + " İki aday isim doğrulanmış değil; MOQ ve münhasırlık kaydı yok."
        ),
        "expected": "kör seçim yok; kapsama/MOQ sor",
    },
    {
        "id": "cheap_price_market",
        "category": "B",
        "input": "Düşük fiyat isteyen bir pazara gireyim mi? Hacim yüksek görünüyor.",
        "context": COMPANY + " Hacim iddiası kullanıcıdan; doğrulanmış sipariş yok.",
        "expected": "hacim iddia; kapasite/marj; kör giriş yok",
    },
    {
        "id": "strategic_small",
        "category": "D",
        "input": "Küçük hacimli ama referans olabilecek bir müşteri. Öncelik vereyim mi?",
        "context": FAIR_FACTS + " " + COMPANY,
        "expected": "kapasiteyi büyük siparişe kilitleme; referans iddia",
    },
    {
        "id": "mix_priority",
        "category": "D",
        "input": "Biri yüksek hacim düşük marj, diğeri düşük hacim stratejik. Hangisine kapasite vereyim?",
        "context": COMPANY + " Marj rakamı kayıtlı değil. İki müşteri kartı yok.",
        "expected": "kör hacim yok; marj/kapasite trade-off",
    },
    {
        "id": "capacity_split",
        "category": "E",
        "input": "Kapasitem yetmiyor. Mevcut Almanya müşterisi mi yeni talep mi?",
        "context": FAIR_FACTS + " " + COMPANY,
        "expected": "mevcutı bırakma; kapasite tahsisi",
    },
    {
        "id": "accept_order",
        "category": "E",
        "input": "Bu siparişi kabul mü reddedeyim? Hacim büyük, teslim tarihi sıkışık.",
        "context": COMPANY + " Teslim tarihi ve kapasite takvimi kayıtlı değil.",
        "expected": "kör kabul yok; kapasite/teslimat",
    },
    {
        "id": "fair_leads",
        "category": "F",
        "input": "Fuar sonrası kart vizitleri aldım. Bugün hangilerine gideyim?",
        "context": FAIR_FACTS + " SOURCE_USER: Kullanıcı fuar sonrası 12 kart viziti aldığını söylüyor.",
        "expected": "A öncelik; skor ≠ win; 12 sayı kullanıcı iddiası",
    },
    {
        "id": "lost_quote",
        "category": "F",
        "input": "Teklif kaybettik. Fiyatı düşürüp tekrar mı gideyim?",
        "context": "Marj kırılımı yok. Kaybetme nedeni kayıtlı değil. Rakip fiyatı doğrulanmadı.",
        "expected": "hemen indirim yok; kayıp nedeni",
    },
    {
        "id": "say_no",
        "category": "F",
        "input": "Bu müşteriye ne zaman hayır demeliyim?",
        "context": CREDIT_THIN + " " + COMPANY,
        "expected": "eşik; kör evet yok; uydurma yüzde yok",
    },
]

ANATOMY_CASES: list[dict] = [
    {**row, "category": _CAT_20[row["id"]], "source": "v51"} for row in CASES
] + [{**row, "source": "v57"} for row in NEW_CASES]
