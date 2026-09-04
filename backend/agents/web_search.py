"""Kontrollü web araştırması. Anahtar yoksa uydurma yok."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .tools_base import ToolResult, ToolSource, failed_tool


def web_search_configured() -> bool:
    return bool(
        (os.getenv("TAVILY_API_KEY") or "").strip()
        or (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
    )


async def web_search(
    http: httpx.AsyncClient,
    query: str,
    *,
    max_results: int = 5,
) -> ToolResult:
    q = (query or "").strip()
    if not q:
        return failed_tool("web_search", "Arama sorgusu boş.")
    tavily = (os.getenv("TAVILY_API_KEY") or "").strip()
    brave = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
    if tavily:
        return await _tavily(http, q, tavily, max_results)
    if brave:
        return await _brave(http, q, brave, max_results)
    return failed_tool(
        "web_search",
        "Web araştırması yapılandırılmadı (TAVILY_API_KEY veya BRAVE_SEARCH_API_KEY yok).",
    )


async def _tavily(
    http: httpx.AsyncClient,
    query: str,
    api_key: str,
    max_results: int,
) -> ToolResult:
    try:
        response = await http.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except Exception as exc:
        return failed_tool("web_search", f"Tavily isteği tamamlanamadı: {exc}")
    sources: list[ToolSource] = []
    findings: list[str] = []
    for item in payload.get("results") or []:
        title = str(item.get("title") or "").strip() or "Kaynak"
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        sources.append(
            ToolSource(title=title, url=url, date="", confidence="medium")
        )
        if snippet:
            findings.append(f"{title}: {snippet[:280]}")
    if not sources:
        return ToolResult(
            name="web_search",
            ok=True,
            summary="Web araması sonuç döndürmedi. Firma veya fiyat uydurulmayacak.",
            data={"query": query, "findings": []},
            provenance="SOURCE_WEB",
        )
    return ToolResult(
        name="web_search",
        ok=True,
        summary=f"{len(sources)} kaynak tarandı. Yalnızca bu bulgulara dayan.",
        data={"query": query, "findings": findings[:max_results]},
        sources=sources[:max_results],
        provenance="SOURCE_WEB",
    )


async def _brave(
    http: httpx.AsyncClient,
    query: str,
    api_key: str,
    max_results: int,
) -> ToolResult:
    try:
        response = await http.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=20.0,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except Exception as exc:
        return failed_tool("web_search", f"Brave isteği tamamlanamadı: {exc}")
    sources: list[ToolSource] = []
    findings: list[str] = []
    web = (payload.get("web") or {}).get("results") or []
    for item in web:
        title = str(item.get("title") or "").strip() or "Kaynak"
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("description") or "").strip()
        sources.append(ToolSource(title=title, url=url, date="", confidence="medium"))
        if snippet:
            findings.append(f"{title}: {snippet[:280]}")
    if not sources:
        return ToolResult(
            name="web_search",
            ok=True,
            summary="Web araması sonuç döndürmedi. Firma veya fiyat uydurulmayacak.",
            data={"query": query, "findings": []},
            provenance="SOURCE_WEB",
        )
    return ToolResult(
        name="web_search",
        ok=True,
        summary=f"{len(sources)} kaynak tarandı. Yalnızca bu bulgulara dayan.",
        data={"query": query, "findings": findings[:max_results]},
        sources=sources[:max_results],
        provenance="SOURCE_WEB",
    )
