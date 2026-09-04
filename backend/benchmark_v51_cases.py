"""V5.1 adil ticari reasoning senaryoları. Kelime ezberi yok; mantık beklenir."""

from __future__ import annotations

FAIR_FACTS = (
    "SOURCE_USER: Kullanıcı Almanya'da 3 potansiyel müşteri bulunduğunu söylüyor. "
    "Doğrulanmış gelir, sipariş veya marj yok. "
    "Kartlar: Nordöl Import DE similarity=0.88 website=var; "
    "Gastro Nord DE similarity=0.79 contact=var; "
    "Retail Mix GmbH DE similarity=0.41. "
    "Fransa için doğrulanmış müşteri, gelir veya fiyat yok. "
    "Eşleşme skoru kazanma ihtimali değildir. "
    "Web bu turda yok. Company capacity kayıtlı değil."
)

BUYER_CARDS = [
    {
        "organization_name": "Nordöl Import",
        "destination_country": "DE",
        "similarity": 0.88,
        "website": "https://example.de",
        "product_name": "zeytinyağı",
    },
    {
        "organization_name": "Gastro Nord",
        "destination_country": "DE",
        "similarity": 0.79,
        "has_contact": True,
        "product_name": "zeytinyağı",
    },
    {
        "organization_name": "Retail Mix GmbH",
        "destination_country": "DE",
        "similarity": 0.41,
    },
    {
        "organization_name": "Generic Shop",
        "destination_country": "NL",
        "similarity": 0.22,
    },
]

# expected: ticari mantık anahtarları (hepsi olmak zorunda değil; test/heuristic)
CASES: list[dict] = [
    {
        "id": "market_choice",
        "input": "Bu ürünü Almanya'da mı Fransa'da mı satayım?",
        "context": FAIR_FACTS,
        "expected": "veri sınırlı; kilitleme; sonraki adım",
    },
    {
        "id": "germany_france",
        "input": "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?",
        "context": FAIR_FACTS,
        "expected": "Almanya'yı bırakma; fiyat baskısı iddia; Fransa ikinci test; 1 soru",
    },
    {
        "id": "price_pressure",
        "input": "Almanya'da 3 müşteri var ama fiyat baskısı yüksek.",
        "context": FAIR_FACTS,
        "expected": "gözlem ≠ marj; ölç",
    },
    {
        "id": "raise_price",
        "input": "Fiyatı artırayım mı?",
        "context": "Kayıtlı marj yok. Web yok.",
        "expected": "kör zam yok; marj ve rekabet verisi sor",
    },
    {
        "id": "discount",
        "input": "Bu müşteriye fiyatı düşüreyim mi?",
        "context": "Marj kırılımı yok. Rakip fiyatı doğrulanmadı.",
        "expected": "hemen düşürme; marj ölç",
    },
    {
        "id": "margin_drop",
        "input": "Marjımız düştü, ne yapalım?",
        "context": "CRM yok. Hacim/fiyat kırılımı yok.",
        "expected": "teşhis; fiyat mı hacim mi; uydurma yok",
    },
    {
        "id": "sales_drop",
        "input": "Satışlar düştü ama mevcut müşteriler aynı.",
        "context": "CRM yok.",
        "expected": "yeni kazanım deliği; reklam dayatma yok",
    },
    {
        "id": "new_buyer_priority",
        "input": "Yeni müşteri gelmiyor ama mevcut müşteriler sipariş veriyor. Nereye bakayım?",
        "context": "CRM yok.",
        "expected": "kazanım; funnel uydurma yok",
    },
    {
        "id": "buyer_abc",
        "input": "Bu alıcılardan hangilerine öncelik vermeliyim?",
        "context": FAIR_FACTS,
        "matches": BUYER_CARDS,
        "expected": "A/B/C; skor ≠ win; eksik iletişim = doğrula",
    },
    {
        "id": "supplier_choice",
        "input": "İki tedarikçi var, hangisini seçeyim?",
        "context": "Fiyat, MOQ, teslimat doğrulanmadı.",
        "expected": "kör seçim yok; MOQ/teslimat sor",
    },
    {
        "id": "capacity",
        "input": "En büyük pazara gideyim mi?",
        "context": "ŞİRKET PROFİLİ: kapasite düşük; ürün zeytinyağı.",
        "session": {"capacity": "kapasite düşük", "product": "zeytinyağı"},
        "expected": "kapasite riski; yüksek hacme kör gitme",
    },
    {
        "id": "volume_vs_margin",
        "input": "Bir müşteri çok yüksek hacim istiyor ama marj düşük. Kabul edeyim mi?",
        "context": "Kapasite sınırlı olabilir. Marj eşiği kayıtlı değil.",
        "expected": "hacmi kovalama; marj/kapasite",
    },
    {
        "id": "competitor_pressure",
        "input": "Rakibim fiyatı benden %20 düşük.",
        "context": "Rakip teklifi kullanıcı iddiası. Web yok.",
        "expected": "iddia; market-stats fallback değil; fiyat eşitleme",
    },
    {
        "id": "new_country",
        "input": "Fransa'da hiç müşterim yok ama pazar büyük. Gireyim mi?",
        "context": "Fransa büyüklüğü kaynaklı değil. Lead yok.",
        "expected": "pazar büyüklüğü iddia; lead yokken maliyet",
    },
    {
        "id": "keep_customer",
        "input": "Mevcut Almanya müşterisini bırakıp yeni ülkeye mi geçeyim?",
        "context": FAIR_FACTS,
        "expected": "mevcut fırsatı bırakma",
    },
    {
        "id": "collection_risk",
        "input": "Bu müşteri büyük sipariş istiyor ama ödemesi gecikebilir. Kabul mü?",
        "context": "Kredi/tahsilat kaydı yok.",
        "expected": "tahsilat riski; kör evet yok",
    },
    {
        "id": "moq",
        "input": "MOQ'u düşüreyim mi bu müşteri için?",
        "context": "Kayıtlı MOQ ve kapasite yok.",
        "expected": "kör MOQ indirme; maliyet/kapasite sor",
    },
    {
        "id": "incoterms",
        "input": "FOB mu CIF mi vereyim?",
        "context": "Lojistik maliyeti doğrulanmadı.",
        "expected": "kör tercih yok; maliyet ve risk ayır",
    },
    {
        "id": "today_plan",
        "input": "Bugün ne yapmalıyım?",
        "context": FAIR_FACTS,
        "expected": "somut sonraki adım; uydurma günlük istatistik yok",
    },
    {
        "id": "uncertain_data",
        "input": "Elimde net veri yok, yine de Almanya mı Fransa mı karar ver.",
        "context": "Doğrulanmış pazar verisi yok.",
        "expected": "yapay kesinlik yok; ne eksik söyle",
    },
]
