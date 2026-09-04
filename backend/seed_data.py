"""Örnek dış ticaret verisi üretir ve POST /trade-items/bulk ile yükler."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
JSON_PATH = BACKEND_DIR / "seed_data.json"
CSV_PATH = BACKEND_DIR / "seed_data.csv"
DEFAULT_API_URL = os.getenv("NEXT_PUBLIC_API_URL") or os.getenv(
    "API_URL", "http://localhost:8000"
)
BATCH_SIZE = 8
REQUEST_TIMEOUT_S = 300

SEED_ITEMS: list[dict[str, object]] = [
    {
        "product_name": "Extra virgin olive oil",
        "hs_code": "1509.10",
        "description": "Bottled extra virgin olive oil for EU retail shelves.",
        "origin_country": "TR",
        "destination_country": "DE",
        "direction": "export",
        "quantity": 420000,
        "unit": "kg",
        "value_usd": 2_850_000,
        "trade_date": "2025-11-12",
        "source": "seed",
    },
    {
        "product_name": "Shelled hazelnuts",
        "hs_code": "0802.22",
        "description": "Grade 1 Levant hazelnuts for confectionery manufacturers.",
        "origin_country": "TR",
        "destination_country": "IT",
        "direction": "export",
        "quantity": 185000,
        "unit": "kg",
        "value_usd": 1_640_000,
        "trade_date": "2025-10-03",
        "source": "seed",
    },
    {
        "product_name": "Knitted cotton apparel",
        "hs_code": "6109.10",
        "description": "Cotton T-shirts and basic knits for US mass retail.",
        "origin_country": "TR",
        "destination_country": "US",
        "direction": "export",
        "quantity": 960000,
        "unit": "pcs",
        "value_usd": 4_120_000,
        "trade_date": "2026-01-18",
        "source": "seed",
    },
    {
        "product_name": "Automotive transmission parts",
        "hs_code": "8708.40",
        "description": "Gearbox components for passenger vehicle assembly.",
        "origin_country": "DE",
        "destination_country": "TR",
        "direction": "import",
        "quantity": 12800,
        "unit": "pcs",
        "value_usd": 6_740_000,
        "trade_date": "2026-02-09",
        "source": "seed",
    },
    {
        "product_name": "Hot rolled steel coils",
        "hs_code": "7208.10",
        "description": "Hot-rolled non-alloy steel coils for construction.",
        "origin_country": "TR",
        "destination_country": "RO",
        "direction": "export",
        "quantity": 24000,
        "unit": "ton",
        "value_usd": 15_600_000,
        "trade_date": "2025-12-21",
        "source": "seed",
    },
    {
        "product_name": "Generic medicines",
        "hs_code": "3004.90",
        "description": "Packaged generic pharmaceuticals for hospital tenders.",
        "origin_country": "IN",
        "destination_country": "TR",
        "direction": "import",
        "quantity": 54000,
        "unit": "kg",
        "value_usd": 3_220_000,
        "trade_date": "2026-03-04",
        "source": "seed",
    },
    {
        "product_name": "Fresh tomatoes",
        "hs_code": "0702.00",
        "description": "Greenhouse tomatoes for Northern European wholesale.",
        "origin_country": "TR",
        "destination_country": "NL",
        "direction": "export",
        "quantity": 3100000,
        "unit": "kg",
        "value_usd": 2_480_000,
        "trade_date": "2026-01-07",
        "source": "seed",
    },
    {
        "product_name": "Durum wheat",
        "hs_code": "1001.19",
        "description": "Milling durum wheat for pasta production.",
        "origin_country": "RU",
        "destination_country": "TR",
        "direction": "import",
        "quantity": 85000,
        "unit": "ton",
        "value_usd": 22_100_000,
        "trade_date": "2025-09-16",
        "source": "seed",
    },
    {
        "product_name": "CNC machining centers",
        "hs_code": "8457.10",
        "description": "Vertical CNC machining centers for metalworking SMEs.",
        "origin_country": "JP",
        "destination_country": "TR",
        "direction": "import",
        "quantity": 36,
        "unit": "pcs",
        "value_usd": 9_850_000,
        "trade_date": "2026-02-27",
        "source": "seed",
    },
    {
        "product_name": "Raw cotton",
        "hs_code": "5201.00",
        "description": "Staple cotton for spinning mills in the Aegean region.",
        "origin_country": "US",
        "destination_country": "TR",
        "direction": "import",
        "quantity": 18000,
        "unit": "ton",
        "value_usd": 31_500_000,
        "trade_date": "2025-11-29",
        "source": "seed",
    },
    {
        "product_name": "Refined copper wire",
        "hs_code": "7408.11",
        "description": "Refined copper wire exceeding 6 mm for cable plants.",
        "origin_country": "TR",
        "destination_country": "GB",
        "direction": "export",
        "quantity": 4200,
        "unit": "ton",
        "value_usd": 38_400_000,
        "trade_date": "2026-01-22",
        "source": "seed",
    },
    {
        "product_name": "Photovoltaic modules",
        "hs_code": "8541.43",
        "description": "Crystalline silicon solar modules for utility-scale parks.",
        "origin_country": "CN",
        "destination_country": "TR",
        "direction": "import",
        "quantity": 210000,
        "unit": "pcs",
        "value_usd": 27_300_000,
        "trade_date": "2026-03-11",
        "source": "seed",
    },
    {
        "product_name": "Aged kashar cheese",
        "hs_code": "0406.90",
        "description": "Vacuum-packed aged kashar for HORECA distributors.",
        "origin_country": "TR",
        "destination_country": "IQ",
        "direction": "export",
        "quantity": 76000,
        "unit": "kg",
        "value_usd": 980_000,
        "trade_date": "2025-12-08",
        "source": "seed",
    },
    {
        "product_name": "Dried figs",
        "hs_code": "0804.20",
        "description": "Aydin dried figs, natural and pasteurized lots.",
        "origin_country": "TR",
        "destination_country": "FR",
        "direction": "export",
        "quantity": 54000,
        "unit": "kg",
        "value_usd": 720_000,
        "trade_date": "2025-10-19",
        "source": "seed",
    },
    {
        "product_name": "Natural marble blocks",
        "hs_code": "2515.12",
        "description": "Afyon white marble blocks for Far East stone yards.",
        "origin_country": "TR",
        "destination_country": "CN",
        "direction": "export",
        "quantity": 9800,
        "unit": "ton",
        "value_usd": 4_560_000,
        "trade_date": "2026-02-14",
        "source": "seed",
    },
    {
        "product_name": "Household refrigerators",
        "hs_code": "8418.10",
        "description": "Combined refrigerator-freezers for EU white-goods chains.",
        "origin_country": "TR",
        "destination_country": "ES",
        "direction": "export",
        "quantity": 64000,
        "unit": "pcs",
        "value_usd": 18_900_000,
        "trade_date": "2026-01-30",
        "source": "seed",
    },
    {
        "product_name": "Polyethylene resins",
        "hs_code": "3901.10",
        "description": "HDPE resins for packaging film extrusion.",
        "origin_country": "SA",
        "destination_country": "TR",
        "direction": "import",
        "quantity": 16000,
        "unit": "ton",
        "value_usd": 14_200_000,
        "trade_date": "2025-11-05",
        "source": "seed",
    },
    {
        "product_name": "Smartphone displays",
        "hs_code": "8524.91",
        "description": "OLED display modules for consumer electronics assembly.",
        "origin_country": "KR",
        "destination_country": "TR",
        "direction": "import",
        "quantity": 890000,
        "unit": "pcs",
        "value_usd": 41_700_000,
        "trade_date": "2026-03-18",
        "source": "seed",
    },
    {
        "product_name": "Hand-knotted wool carpets",
        "hs_code": "5701.10",
        "description": "Hereke-style wool carpets for US interior designers.",
        "origin_country": "TR",
        "destination_country": "US",
        "direction": "export",
        "quantity": 3200,
        "unit": "pcs",
        "value_usd": 2_150_000,
        "trade_date": "2025-12-02",
        "source": "seed",
    },
    {
        "product_name": "Fresh citrus fruit",
        "hs_code": "0805.10",
        "description": "Washington navel oranges for Eastern European wholesale.",
        "origin_country": "TR",
        "destination_country": "UA",
        "direction": "export",
        "quantity": 2_400_000,
        "unit": "kg",
        "value_usd": 1_860_000,
        "trade_date": "2026-01-11",
        "source": "seed",
    },
    {
        "product_name": "Woven clothing labels (dokuma etiket)",
        "hs_code": "5807.10",
        "description": "Damask woven brand labels for German ready-to-wear importers and apparel brands.",
        "origin_country": "TR",
        "destination_country": "DE",
        "direction": "export",
        "quantity": 2_500_000,
        "unit": "m",
        "value_usd": 1_180_000,
        "trade_date": "2026-02-14",
        "source": "seed",
    },
    {
        "product_name": "Dokuma etiket / woven garment labels",
        "hs_code": "5807.10",
        "description": "Woven labels and care tags for Italian apparel brands and garment manufacturers.",
        "origin_country": "TR",
        "destination_country": "IT",
        "direction": "export",
        "quantity": 1_800_000,
        "unit": "m",
        "value_usd": 940_000,
        "trade_date": "2026-01-22",
        "source": "seed",
    },
    {
        "product_name": "Apparel brand woven labels",
        "hs_code": "5807.90",
        "description": "Woven clothing labels for French prêt-à-porter houses and fabric importers.",
        "origin_country": "TR",
        "destination_country": "FR",
        "direction": "export",
        "quantity": 960_000,
        "unit": "m",
        "value_usd": 510_000,
        "trade_date": "2025-12-08",
        "source": "seed",
    },
    {
        "product_name": "Woven labels for ready-wear",
        "hs_code": "5807.10",
        "description": "Dokuma etiket shipments to Dutch apparel wholesalers and brand buyers.",
        "origin_country": "TR",
        "destination_country": "NL",
        "direction": "export",
        "quantity": 720_000,
        "unit": "m",
        "value_usd": 385_000,
        "trade_date": "2026-03-01",
        "source": "seed",
    },
    {
        "product_name": "Satin and damask woven labels",
        "hs_code": "5807.10",
        "description": "Woven brand labels for Spanish ready-wear manufacturers and clothing importers.",
        "origin_country": "TR",
        "destination_country": "ES",
        "direction": "export",
        "quantity": 640_000,
        "unit": "m",
        "value_usd": 298_000,
        "trade_date": "2026-02-03",
        "source": "seed",
    },
]


def write_seed_files(items: list[dict[str, object]]) -> None:
    JSON_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(items[0].keys())
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)
    print(f"Yazıldı: {JSON_PATH.name} ({len(items)} kalem)")
    print(f"Yazıldı: {CSV_PATH.name}")


def load_items(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError("JSON kökü bir dizi veya {\"items\": [...]} olmalı.")
    return data


def normalize_item(raw: dict[str, object]) -> dict[str, object]:
    item: dict[str, object] = {}
    for key in (
        "product_name",
        "hs_code",
        "description",
        "origin_country",
        "destination_country",
        "direction",
        "unit",
        "trade_date",
        "source",
    ):
        value = raw.get(key)
        if value not in (None, ""):
            item[key] = str(value).strip()
    for key in ("quantity", "value_usd"):
        value = raw.get(key)
        if value in (None, ""):
            continue
        item[key] = float(value)
    if "origin_country" in item:
        item["origin_country"] = str(item["origin_country"]).upper()[:2]
    if "destination_country" in item:
        item["destination_country"] = str(item["destination_country"]).upper()[:2]
    return item


def chunked(items: list[dict[str, object]], size: int) -> list[list[dict[str, object]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def post_bulk(api_url: str, items: list[dict[str, object]]) -> dict[str, object]:
    url = f"{api_url.rstrip('/')}/trade-items/bulk"
    payload = json.dumps({"items": items}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"API'ye bağlanılamadı ({url}). uvicorn çalışıyor mu? {exc.reason}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Örnek dış ticaret verisini hazırlar ve /trade-items/bulk'a yükler."
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="FastAPI kök URL (varsayılan: http://localhost:8000)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=JSON_PATH,
        help="Okunacak JSON veya CSV (yoksa örnek veri yazılır)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dosyaları yaz, API'ye gönderme",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Her POST /trade-items/bulk isteğindeki kalem sayısı",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_seed_files(SEED_ITEMS)
    source = args.file if args.file.exists() else JSON_PATH

    items = [normalize_item(row) for row in load_items(source)]
    items = [row for row in items if row.get("product_name")]
    if not items:
        print("Yüklenecek kalem yok.", file=sys.stderr)
        return 1

    print(f"{len(items)} kalem hazır ({source.name}).")
    if args.dry_run:
        print("dry-run: POST atlandı.")
        return 0

    inserted = 0
    failed = 0
    for batch_index, batch in enumerate(chunked(items, max(args.batch_size, 1)), start=1):
        print(f"POST /trade-items/bulk  batch {batch_index} ({len(batch)} kalem)…")
        result = post_bulk(args.api_url, batch)
        inserted += int(result.get("inserted") or 0)
        failed += int(result.get("failed") or 0)
        errors = result.get("errors") or []
        if errors:
            for error in errors:
                print(f"  hata [{error.get('index')}]: {error.get('error')}")

    print(f"Bitti. inserted={inserted} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
