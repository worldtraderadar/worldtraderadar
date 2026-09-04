"""Kontrollü tool sonuçları. LLM sınırsız kod çalıştırmaz."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSource:
    title: str
    url: str = ""
    date: str = ""
    confidence: str = "low"


@dataclass
class ToolResult:
    name: str
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[ToolSource] = field(default_factory=list)
    error: str | None = None
    provenance: str = "INFERENCE"

    def for_prompt(self) -> str:
        status = "OK" if self.ok else "TOOL_FAILED"
        lines = [f"[{self.name}] {status} kaynak={self.provenance}: {self.summary}"]
        if self.error and not self.ok:
            lines.append(f"Hata: {self.error}")
            lines.append("Bu araçtan sayı veya firma uydurma. Başarı iddia etme.")
        for source in self.sources[:6]:
            bit = source.title
            if source.url:
                bit += f" ({source.url})"
            if source.date:
                bit += f" [{source.date}]"
            lines.append(f"- kaynak: {bit} güven={source.confidence}")
        return "\n".join(lines)


def failed_tool(name: str, reason: str) -> ToolResult:
    return ToolResult(
        name=name,
        ok=False,
        summary="Araç sonucu yok; uydurma yasak.",
        error=reason,
        provenance="TOOL_FAILED",
        data={"failed": True},
    )
