"""Dosya / tablo analizi altyapısı. ConsultDesk henüz dosya yüklemez."""

from __future__ import annotations

from .tools_base import ToolResult, failed_tool


def analyze_documents(*, file_names: list[str] | None, question: str) -> ToolResult:
    names = [name.strip() for name in (file_names or []) if name and name.strip()]
    if not names:
        return failed_tool(
            "document_analysis",
            "Yüklü Excel, CSV, PDF veya görsel yok. Tablo uydurulmayacak.",
        )
    return ToolResult(
        name="document_analysis",
        ok=False,
        summary=(
            "Dosya alındı ancak ayrıştırıcı henüz bağlı değil. "
            "İleride trend, anomali, kanal ve ülke performansı çıkarılacak."
        ),
        data={"files": names, "question": question},
        error="parser_not_wired",
        provenance="TOOL_FAILED",
    )


def diagnostic_outline(question: str) -> str:
    return (
        "Satış düşüşünü tek cümleyle kapatmam. Önce kaynağı ayıralım: "
        "trafik mi düştü, dönüşüm mü, müşteri kaybı mı, ortalama sipariş değeri mi, "
        "yoksa belirli bir ülke/kanal mı? "
        "Son iki aydaki düşüş mevcut müşterilerden mi geliyor, yoksa yeni müşteri "
        "girişinin durmasından mı? Elimde Excel veya CRM dökümü varsa doğrudan bakarım."
    )
