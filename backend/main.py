"""World Trade Radar API — FastAPI + Ollama BGE-M3 + Supabase RPC."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from supabase import AsyncClient, create_async_client

from agents import fail_run, run_orchestrator
from agents.session import (
    SessionAccessDenied,
    create_session,
    delete_session,
    get_session_for_account,
    hydrate,
    list_sessions,
    session_notes,
    session_snapshot,
    session_title,
)
from agents.types import AgentDeps, AgentStep
from auth import AuthMiddleware, require_principal
from billing import (
    QuotaExceeded,
    QuotaMiddleware,
    add_tokens,
    confirm_checkout,
    consume_quota,
    get_snapshot,
    list_plans,
    plan_id_for_request,
    redact_record,
    redact_records,
    set_plan,
    start_checkout,
    take_tokens,
)
from agents.commercial import is_reasoning_task
from llm import (
    bind_canary_model,
    bind_session,
    canary_bucket,
    canary_enabled,
    canary_percent,
    canary_unload_peer,
    chat_model,
    current_session_id,
    embed_model,
    model_for_task,
    ollama_base_url,
    ollama_chat_payload,
    reasoning_model,
    reasoning_override,
    reasoning_think,
    shadow_model,
    thinking_from_ollama,
    unload_peer_after,
    user_content_from_ollama,
    user_visible_text,
)
from prompts import (
    CONSULT_SYSTEM_PROMPT,
    TURKISH_RETRY_LOCK,
    is_buyer_search,
    is_decision_question,
    is_diagnostic_request,
    is_draft_request,
    is_general_intake,
    is_language_barrier,
    is_memory_recall,
    is_method_question,
    is_sector_chat,
    is_small_talk,
    is_strategy_request,
    is_supplier_search,
    looks_english,
    ollama_chat_options,
    polish_chat_reply,
    wants_technical_detail,
    wrap_user_prompt,
)
from tts_local import TTS_LANGUAGE, get_tts, synthesize_wav, warmup as warmup_tts

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_LOCAL_PATH = ROOT_DIR / ".env.local"

EMBEDDING_DIM = 1024
AGENT_NAME = "worldtraderadar-api"


def _unwrap_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_env_local(path: Path = ENV_LOCAL_PATH) -> dict[str, str]:
    """Kök dizindeki .env.local dosyasını okur ve os.environ'a yazar."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        parsed = _unwrap_env_value(value)
        values[key] = parsed
        os.environ[key] = parsed
    return values


_env_local = load_env_local()

OLLAMA_EMBED_MODEL = embed_model()
OLLAMA_CHAT_MODEL = chat_model()
OLLAMA_BASE_URL = ollama_base_url()

NEXT_PUBLIC_SUPABASE_URL = (
    _env_local.get("NEXT_PUBLIC_SUPABASE_URL")
    or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
).strip()
SUPABASE_SERVICE_ROLE_KEY = (
    _env_local.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
).strip()


def _is_placeholder_url(url: str) -> bool:
    lowered = url.lower()
    return (not url) or ("proje-id" in lowered) or ("your-project" in lowered)


def _is_placeholder_key(key: str) -> bool:
    return (not key) or key.startswith("senin_") or key.startswith("your_")


def _supabase_credentials() -> tuple[str, str]:
    if _is_placeholder_url(NEXT_PUBLIC_SUPABASE_URL):
        raise HTTPException(
            status_code=500,
            detail=(
                "NEXT_PUBLIC_SUPABASE_URL kök dizindeki .env.local içinde "
                "gerçek Supabase proje URL'si olarak ayarlanmalı."
            ),
        )
    if _is_placeholder_key(SUPABASE_SERVICE_ROLE_KEY):
        raise HTTPException(
            status_code=500,
            detail=(
                "SUPABASE_SERVICE_ROLE_KEY kök dizindeki .env.local içinde "
                "gerçek service_role anahtarı olarak ayarlanmalı."
            ),
        )
    return NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY


def build_embedding_text(
    *,
    product_name: str,
    description: str | None = None,
    hs_code: str | None = None,
    origin_country: str | None = None,
    destination_country: str | None = None,
) -> str:
    parts = [product_name.strip()]
    if hs_code:
        parts.append(f"HS {hs_code.strip()}")
    if description:
        parts.append(description.strip())
    if origin_country or destination_country:
        parts.append(
            f"{(origin_country or '?').upper()} -> {(destination_country or '?').upper()}"
        )
    return " | ".join(part for part in parts if part)


def trade_item_to_row(
    item: TradeItemCreate,
    embedding: list[float],
    embedding_text: str,
) -> dict[str, Any]:
    payload = item.model_dump(mode="json")
    for field in ("origin_country", "destination_country"):
        if payload.get(field):
            payload[field] = payload[field].upper()
    payload.update(
        {
            "embedding": embedding,
            "embedding_text": embedding_text,
            "embedding_model": OLLAMA_EMBED_MODEL,
        }
    )
    return payload


async def embed_with_ollama(http: httpx.AsyncClient, text: str) -> list[float]:
    """Ollama /api/embed (yedek: /api/embeddings) ile bge-m3 dense vektör üretir."""
    payload_text = text.strip()
    if not payload_text:
        raise HTTPException(status_code=400, detail="Gömülecek metin boş olamaz.")

    try:
        response = await http.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": payload_text},
        )
        if response.status_code == 404:
            response = await http.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": payload_text},
            )
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama'ya bağlanılamadı ({OLLAMA_BASE_URL}). "
                "ollama serve çalışıyor mu? Model: ollama pull bge-m3"
            ),
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama embedding hatası ({OLLAMA_EMBED_MODEL}): {exc.response.text}",
        ) from exc

    data = response.json()
    embeddings = data.get("embeddings")
    vector: list[float] | None
    if isinstance(embeddings, list) and embeddings:
        vector = embeddings[0]
    else:
        vector = data.get("embedding")

    if not isinstance(vector, list) or not vector:
        raise HTTPException(status_code=502, detail="Ollama boş embedding döndü.")
    if len(vector) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=502,
            detail=(
                f"BGE-M3 1024 boyut bekleniyor, gelen {len(vector)}. "
                f"Model: {OLLAMA_EMBED_MODEL}"
            ),
        )
    return [float(value) for value in vector]


async def match_trade_items(
    supabase: AsyncClient,
    query_embedding: list[float],
    *,
    match_count: int,
    match_threshold: float,
    organization_id: UUID | None,
    product_query: str | None = None,
) -> list[dict[str, Any]]:
    """Supabase `match_trade_items` RPC çağrısı."""
    payload: dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "match_threshold": match_threshold,
        "filter_organization_id": str(organization_id) if organization_id else None,
    }
    if product_query and product_query.strip():
        payload["filter_query"] = product_query.strip()[:120]
    try:
        result = await supabase.rpc("match_trade_items", payload).execute()
    except Exception:
        payload.pop("filter_query", None)
        result = await supabase.rpc("match_trade_items", payload).execute()
    return result.data or []


def _history_turns(history: list[Any] | None) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = str(item.get("role") or "")
            content = str(item.get("content") or "")
        else:
            role = str(getattr(item, "role", ""))
            content = str(getattr(item, "content", ""))
        if role not in ("user", "assistant"):
            continue
        body = content.strip()
        if not body:
            continue
        turns.append({"role": role, "content": body})
    return turns


async def _ollama_complete(
    http: httpx.AsyncClient,
    *,
    system: str,
    user: str,
    options: dict[str, float],
    history: list[Any] | None = None,
    task: str = "chat",
) -> tuple[str, dict[str, Any]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(_history_turns(history))
    messages.append({"role": "user", "content": user})
    transcript = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in messages
        if turn["role"] != "system"
    )
    model_name = model_for_task(task)
    base = ollama_base_url()
    chat_payload = ollama_chat_payload(
        model_name,
        messages,
        think=reasoning_think(),
        options=dict(options or {}),
    )
    t0 = time.perf_counter()
    try:
        response = await http.post(f"{base}/api/chat", json=chat_payload)
        if response.status_code == 404:
            response = await http.post(
                f"{base}/api/generate",
                json={
                    "model": model_name,
                    "stream": False,
                    "prompt": transcript,
                    "system": system,
                    "think": reasoning_think(),
                    "options": chat_payload.get("options") or options,
                    "keep_alive": chat_payload.get("keep_alive"),
                },
            )
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama'ya bağlanılamadı ({base}). "
                f"ollama serve çalışıyor mu? Model: ollama pull {model_name}"
            ),
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama sohbet hatası ({model_name}): {exc.response.text}",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Ollama sohbet yanıtı zaman aşımına uğradı ({model_name}).",
        ) from exc

    data = response.json()
    content = user_visible_text(
        user_content_from_ollama(data),
        thinking_from_ollama(data),
    )
    thinking = thinking_from_ollama(data)
    elapsed = time.perf_counter() - t0
    eval_count = data.get("eval_count") or 0
    eval_ns = data.get("eval_duration") or 0
    prompt_ns = data.get("prompt_eval_duration") or 0
    load_ns = data.get("load_duration") or 0
    tps = (eval_count / (eval_ns / 1e9)) if eval_ns else None
    from agents.quality import record_inference

    record_inference(
        {
            "model": model_name,
            "think": chat_payload.get("think"),
            "task": task,
            "num_predict": (chat_payload.get("options") or {}).get("num_predict"),
            "latency_s": round(elapsed, 3),
            "eval_count": eval_count,
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": prompt_ns,
            "prompt_eval_s": round(prompt_ns / 1e9, 3) if prompt_ns else None,
            "gen_s": round(eval_ns / 1e9, 3) if eval_ns else None,
            "load_s": round(load_ns / 1e9, 3) if load_ns else None,
            "tokens_per_sec": round(tps, 2) if tps else None,
            "done_reason": data.get("done_reason") or "",
            "content_length": len(content),
            "thinking_length": len(thinking),
            "load_duration": load_ns,
            "session_bucket": canary_bucket(current_session_id())
            if current_session_id()
            else None,
        }
    )
    if not content:
        raise HTTPException(status_code=502, detail=f"Ollama {model_name} boş yanıt döndü.")
    return content, data


def _count_tokens(data: dict[str, Any], system: str, user: str, text: str) -> int:
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return max(prompt_tokens + completion_tokens, 1)
    return max((len(system) + len(user) + len(text) + 3) // 4, 1)


async def generate_with_llama3(
    http: httpx.AsyncClient,
    prompt: str,
    system_prompt: str | None = None,
    history: list[Any] | None = None,
    polish: bool = True,
) -> str:
    """Ollama /api/chat (yedek: /api/generate) ile sohbet metni üretir."""
    system = system_prompt or CONSULT_SYSTEM_PROMPT
    user = wrap_user_prompt(prompt)
    options = ollama_chat_options()
    prior = _history_turns(history)
    task = "commercial" if is_reasoning_task(prompt) else "chat"
    content, data = await _ollama_complete(
        http, system=system, user=user, options=options, history=prior, task=task
    )
    used = _count_tokens(data, system, user, content)
    if looks_english(content):
        retry_system = f"{system}\n\n{TURKISH_RETRY_LOCK}"
        content, retry_data = await _ollama_complete(
            http,
            system=retry_system,
            user=user,
            options=options,
            history=prior,
            task=task,
        )
        used += _count_tokens(retry_data, retry_system, user, content)
    if not polish:
        add_tokens(used)
        return content.strip()
    has_history = bool(prior)
    advisor = (
        is_strategy_request(prompt)
        or is_method_question(prompt)
        or is_language_barrier(prompt)
        or is_draft_request(prompt)
        or is_memory_recall(prompt)
        or is_diagnostic_request(prompt)
    )
    text = polish_chat_reply(
        content,
        allow_codes=wants_technical_detail(prompt),
        small_talk=is_small_talk(prompt, has_history=has_history) and not has_history,
        sector_chat=is_sector_chat(prompt, has_history=has_history) and not advisor,
        intake=is_general_intake(prompt) and not has_history and not advisor,
        action_first=is_buyer_search(prompt) or is_supplier_search(prompt),
        advisor_report=advisor,
    )
    add_tokens(used)
    return text


async def log_agent(
    supabase: AsyncClient,
    *,
    action: str,
    status: Literal["pending", "running", "success", "error"],
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    error_message: str | None = None,
    duration_ms: int | None = None,
    organization_id: UUID | None = None,
    trade_item_id: UUID | None = None,
    account_id: str | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "agent_name": AGENT_NAME,
        "action": action,
        "status": status,
        "input": dict(input_data or {}),
        "output": output_data or {},
        "error_message": error_message,
        "duration_ms": duration_ms,
        "organization_id": str(organization_id) if organization_id else None,
        "trade_item_id": str(trade_item_id) if trade_item_id else None,
    }
    if account_id:
        payload["account_id"] = account_id
        payload["input"] = {**payload["input"], "account_id": account_id}
    try:
        result = await supabase.table("agent_logs").insert(payload).execute()
        return (result.data or [None])[0]
    except Exception:
        if "account_id" in payload:
            payload.pop("account_id", None)
            try:
                result = await supabase.table("agent_logs").insert(payload).execute()
                return (result.data or [None])[0]
            except Exception:
                return None
        return None


def get_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.http


async def get_supabase(request: Request) -> AsyncClient:
    client = request.app.state.supabase
    if client is not None:
        return client

    async with request.app.state.supabase_lock:
        if request.app.state.supabase is not None:
            return request.app.state.supabase
        url, key = _supabase_credentials()
        try:
            request.app.state.supabase = await create_async_client(url, key)
        except Exception as exc:
            request.app.state.supabase_error = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Supabase istemcisi oluşturulamadı: {exc}",
            ) from exc
        request.app.state.supabase_error = None
        return request.app.state.supabase


# ---------------------------------------------------------------------------
# Pydantic modelleri
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    embedding: list[float]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    match_count: int = Field(default=10, ge=1, le=100)
    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    organization_id: UUID | None = None


class LockedFlags(BaseModel):
    contact: bool = False
    volume: bool = False


class TradeItemMatch(BaseModel):
    id: UUID
    organization_id: UUID | None = None
    hs_code: str | None = None
    product_name: str
    description: str | None = None
    origin_country: str | None = None
    destination_country: str | None = None
    similarity: float
    quantity: float | None = None
    unit: str | None = None
    value_usd: float | None = None
    organization_name: str | None = None
    contact_email: str | None = None
    website: str | None = None
    has_volume: bool = False
    has_contact: bool = False
    plan_id: str | None = None
    locked: LockedFlags = Field(default_factory=LockedFlags)


class SearchResponse(BaseModel):
    query: str
    model: str
    results: list[TradeItemMatch]


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    lang: str = Field(default="tr")


class ChatTurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ConsultRequest(BaseModel):
    question: str = Field(..., min_length=1)
    match_count: int = Field(default=8, ge=1, le=50)
    match_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    organization_id: UUID | None = None
    session_id: str | None = None
    history: list[ChatTurnIn] = Field(default_factory=list, max_length=40)
    file_names: list[str] = Field(default_factory=list)


class AgentStepOut(BaseModel):
    key: str
    agent: str
    label: str
    status: str
    detail: str = ""


class ConsultResponse(BaseModel):
    question: str
    advice: str
    embed_model: str
    chat_model: str
    context: list[TradeItemMatch]
    run_id: str | None = None
    intent: str | None = None
    selected_agent: str | None = None
    steps: list[AgentStepOut] = Field(default_factory=list)
    speak_url: str = "/speak"
    session_id: str | None = None
    history: list[ChatTurnIn] = Field(default_factory=list)
    session_title: str | None = None
    product: str | None = None
    capacity: str | None = None
    market: str | None = None


def format_consult_context(
    matches: list[TradeItemMatch],
    question: str = "",
) -> str:
    """Modele giden iç not: sohbet malzemesi, rapor ham maddesi değil."""
    if not matches:
        return "Kayıt yok."
    technical = wants_technical_detail(question)
    shown = matches[:6] if technical else matches[:4]
    lines: list[str] = []
    for item in shown:
        origin = (item.origin_country or "?").strip() or "?"
        destination = (item.destination_country or "?").strip() or "?"
        name = (item.product_name or "").strip() or "isimsiz kalem"
        if technical:
            hs_code = (item.hs_code or "").strip() or "—"
            lines.append(f"- {name}, {origin} → {destination}, kod {hs_code}")
        else:
            lines.append(f"- {name}, {origin} → {destination}")
        description = (item.description or "").strip()
        if description and technical:
            lines.append(f"  {description[:180]}")
    return "\n".join(lines)


class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=1)
    slug: str | None = None
    country_code: str | None = Field(default=None, min_length=2, max_length=2)
    city: str | None = None
    organization_type: Literal[
        "importer", "exporter", "trader", "broker", "logistics", "unknown"
    ] = "unknown"
    tax_id: str | None = None
    website: str | None = None
    contact_email: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradeItemCreate(BaseModel):
    product_name: str = Field(..., min_length=1)
    organization_id: UUID | None = None
    hs_code: str | None = None
    description: str | None = None
    origin_country: str | None = Field(default=None, min_length=2, max_length=2)
    destination_country: str | None = Field(default=None, min_length=2, max_length=2)
    direction: Literal["import", "export"] | None = None
    quantity: float | None = None
    unit: str | None = None
    value_usd: float | None = None
    trade_date: date | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradeItemBulkRequest(BaseModel):
    items: list[TradeItemCreate] = Field(..., min_length=1, max_length=100)


class TradeItemBulkItemError(BaseModel):
    index: int
    product_name: str
    error: str


class TradeItemBulkResponse(BaseModel):
    inserted: int
    failed: int
    embed_model: str
    items: list[dict[str, Any]]
    errors: list[TradeItemBulkItemError]


class AgentLogCreate(BaseModel):
    action: str = Field(..., min_length=1)
    status: Literal["pending", "running", "success", "error"] = "pending"
    agent_name: str = AGENT_NAME
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    duration_ms: int | None = None
    organization_id: UUID | None = None
    trade_item_id: UUID | None = None


class BillingCheckoutRequest(BaseModel):
    plan_id: str = "pro"


class BillingConfirmRequest(BaseModel):
    checkout_id: str = Field(..., min_length=1)


class BillingPlanChangeRequest(BaseModel):
    plan_id: str


async def get_supabase_optional(request: Request) -> AsyncClient | None:
    try:
        return await get_supabase(request)
    except HTTPException:
        return None


async def flush_stream_tokens(request: Request, supabase: AsyncClient | None) -> None:
    used = take_tokens()
    if not used:
        return
    principal = require_principal(request)
    slug = principal.account_slug
    try:
        await consume_quota(supabase, slug, searches=0, tokens=used)
    except QuotaExceeded:
        pass


TRADE_ITEM_PAYWALL_SELECT = (
    "id, organization_id, hs_code, product_name, description, "
    "origin_country, destination_country, direction, quantity, unit, "
    "value_usd, trade_date, source, embedding_model, created_at, "
    "organizations(id, name, contact_email, website, tax_id, country_code, "
    "city, organization_type)"
)


def flatten_organization(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    org = payload.pop("organizations", None) or payload.get("organization")
    if isinstance(org, list):
        org = org[0] if org else None
    if isinstance(org, dict):
        payload["organization"] = org
        payload["organization_name"] = org.get("name")
        payload["contact_email"] = org.get("contact_email")
        payload["website"] = org.get("website")
    return payload


async def fetch_trade_details(
    supabase: AsyncClient, item_ids: list[str]
) -> dict[str, dict[str, Any]]:
    if not item_ids:
        return {}
    try:
        result = (
            await supabase.table("trade_items")
            .select(TRADE_ITEM_PAYWALL_SELECT)
            .in_("id", item_ids)
            .execute()
        )
    except Exception:
        result = (
            await supabase.table("trade_items")
            .select(
                "id, organization_id, quantity, unit, value_usd, "
                "product_name, hs_code, origin_country, destination_country, "
                "description, direction"
            )
            .in_("id", item_ids)
            .execute()
        )
    details: dict[str, dict[str, Any]] = {}
    for row in result.data or []:
        if row.get("id"):
            details[str(row["id"])] = flatten_organization(row)
    return details


async def apply_item_paywall(
    request: Request,
    supabase: AsyncClient | None,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    plan_id = await plan_id_for_request(request, supabase)
    return redact_records([flatten_organization(row) for row in rows], plan_id)


async def paywall_matches(
    request: Request,
    supabase: AsyncClient,
    matches: list[TradeItemMatch],
) -> list[TradeItemMatch]:
    details = await fetch_trade_details(
        supabase, [str(item.id) for item in matches]
    )
    plan_id = await plan_id_for_request(request, supabase)
    locked_matches: list[TradeItemMatch] = []
    for item in matches:
        merged = {**item.model_dump(mode="json"), **details.get(str(item.id), {})}
        merged["similarity"] = item.similarity
        locked_matches.append(
            TradeItemMatch.model_validate(redact_record(merged, plan_id))
        )
    return locked_matches


# ---------------------------------------------------------------------------
# Uygulama
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0))
    app.state.supabase = None
    app.state.supabase_error = None
    app.state.supabase_lock = asyncio.Lock()
    threading.Thread(target=warmup_tts, name="tts-warmup", daemon=True).start()
    yield
    await app.state.http.aclose()


app = FastAPI(
    title="World Trade Radar API",
    description=(
        "Kök `.env.local` → Supabase service_role. "
        "Ollama `bge-m3` embedding + `match_trade_items` RPC + llama3 danışmanlık."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(QuotaMiddleware, get_supabase=get_supabase)
app.add_middleware(AuthMiddleware, get_supabase=get_supabase)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    ollama_ok = False
    ollama_has_bge_m3 = False
    ollama_has_llama3 = False
    ollama_error: str | None = None
    try:
        response = await get_http(request).get(f"{OLLAMA_BASE_URL}/api/tags")
        ollama_ok = response.status_code == 200
        if ollama_ok:
            models = response.json().get("models") or []
            names = [str(item.get("name", "")) for item in models]
            ollama_has_bge_m3 = any(
                name == OLLAMA_EMBED_MODEL or name.startswith(f"{OLLAMA_EMBED_MODEL}:")
                for name in names
            )
            ollama_has_llama3 = any(
                name == OLLAMA_CHAT_MODEL or name.startswith(f"{OLLAMA_CHAT_MODEL}:")
                for name in names
            )
        else:
            ollama_error = response.text
    except httpx.HTTPError as exc:
        ollama_error = str(exc)

    supabase_configured = not (
        _is_placeholder_url(NEXT_PUBLIC_SUPABASE_URL)
        or _is_placeholder_key(SUPABASE_SERVICE_ROLE_KEY)
    )

    return {
        "status": "ok"
        if ollama_ok and ollama_has_bge_m3 and ollama_has_llama3 and supabase_configured
        else "degraded",
        "env_local": {
            "path": str(ENV_LOCAL_PATH),
            "exists": ENV_LOCAL_PATH.is_file(),
            "NEXT_PUBLIC_SUPABASE_URL": not _is_placeholder_url(NEXT_PUBLIC_SUPABASE_URL),
            "SUPABASE_SERVICE_ROLE_KEY": not _is_placeholder_key(SUPABASE_SERVICE_ROLE_KEY),
        },
        "supabase": {
            "configured": supabase_configured,
            "connected": request.app.state.supabase is not None,
            "error": request.app.state.supabase_error,
        },
        "ollama": {
            "ok": ollama_ok,
            "base_url": OLLAMA_BASE_URL,
            "embed_model": OLLAMA_EMBED_MODEL,
            "chat_model": OLLAMA_CHAT_MODEL,
            "reasoning_model": reasoning_model(),
            "shadow_model": shadow_model() or "",
            "think": reasoning_think(),
            "canary_enabled": canary_enabled(),
            "canary_percent": canary_percent() if canary_enabled() else 0,
            "bge_m3_available": ollama_has_bge_m3,
            "llama3_available": ollama_has_llama3,
            "error": ollama_error,
        },
        "billing": {
            "plans": ["free", "pro"],
            "metered_paths": ["/search", "/consult", "/consult/stream"],
        },
        "tts": get_tts().status(),
    }


@app.get("/speak/status")
async def speak_status() -> dict[str, Any]:
    return get_tts().status()


@app.post("/speak")
async def speak(body: SpeakRequest) -> Response:
    lang = (body.lang or TTS_LANGUAGE).strip().lower()
    if not lang.startswith("tr"):
        raise HTTPException(
            status_code=400,
            detail="Yerel ses motoru yalnızca Türkçe (tr) konuşur.",
        )
    t0 = time.perf_counter()
    try:
        result = await asyncio.to_thread(synthesize_wav, body.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Yerel TTS üretilemedi: {exc}",
        ) from exc
    elapsed = time.perf_counter() - t0
    from agents.quality import record_tts

    record_tts(
        {
            "latency_s": round(elapsed, 3),
            "engine": result.engine,
            "bytes": len(result.wav_bytes or b""),
            "cached": result.cached,
        }
    )
    return Response(
        content=result.wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="speak.wav"',
            "X-TTS-Engine": result.engine,
            "X-TTS-Language": result.language,
            "X-TTS-Cached": "1" if result.cached else "0",
            "X-TTS-Latency-Ms": str(int(elapsed * 1000)),
            "Cache-Control": "private, max-age=1200",
        },
    )


@app.get("/billing/plans")
async def billing_plans(request: Request) -> dict[str, Any]:
    supabase = await get_supabase_optional(request)
    plans = await list_plans(supabase)
    return {"plans": plans}


@app.get("/auth/me")
async def auth_me(request: Request) -> dict[str, Any]:
    """Trusted account context from JWT → membership (not client-selected)."""
    principal = require_principal(request)
    return {
        "user_id": principal.user_id,
        "account_id": principal.account_id,
        "account_slug": principal.account_slug,
        "role": principal.role,
        "plan_id": principal.plan_id,
    }


@app.get("/billing/me")
async def billing_me(request: Request) -> dict[str, Any]:
    principal = require_principal(request)
    supabase = await get_supabase_optional(request)
    snapshot = await get_snapshot(supabase, principal.account_slug)
    return snapshot.to_dict()


@app.post("/billing/checkout")
async def billing_checkout(
    body: BillingCheckoutRequest, request: Request
) -> dict[str, Any]:
    principal = require_principal(request)
    supabase = await get_supabase_optional(request)
    try:
        session = await start_checkout(supabase, principal.account_slug, body.plan_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return session


@app.post("/billing/checkout/confirm")
async def billing_checkout_confirm(
    body: BillingConfirmRequest, request: Request
) -> dict[str, Any]:
    principal = require_principal(request)
    supabase = await get_supabase_optional(request)
    try:
        session, snapshot = await confirm_checkout(
            supabase, principal.account_slug, body.checkout_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"checkout": session, "account": snapshot.to_dict()}


@app.post("/billing/plan")
async def billing_change_plan(
    body: BillingPlanChangeRequest, request: Request
) -> dict[str, Any]:
    if body.plan_id not in {"free", "pro"}:
        raise HTTPException(status_code=400, detail="Plan free veya pro olmalı.")
    principal = require_principal(request)
    supabase = await get_supabase_optional(request)
    snapshot = await set_plan(supabase, principal.account_slug, body.plan_id)
    # Keep in-request principal plan in sync for subsequent redact in same process
    # (membership memory / next request will reload from DB).
    return snapshot.to_dict()


@app.post("/embed", response_model=EmbedResponse)
async def embed(body: EmbedRequest, request: Request) -> EmbedResponse:
    vector = await embed_with_ollama(get_http(request), body.text)
    return EmbedResponse(
        model=OLLAMA_EMBED_MODEL,
        dimensions=len(vector),
        embedding=vector,
    )


@app.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    principal = require_principal(request)
    supabase = await get_supabase(request)
    started = time.perf_counter()
    try:
        query_embedding = await embed_with_ollama(get_http(request), body.query)
        rows = await match_trade_items(
            supabase,
            query_embedding,
            match_count=body.match_count,
            match_threshold=body.match_threshold,
            organization_id=body.organization_id,
        )
    except HTTPException as exc:
        await log_agent(
            supabase,
            action="search",
            status="error",
            input_data={
                **body.model_dump(mode="json"),
                "account_id": principal.account_id,
            },
            error_message=str(exc.detail),
            duration_ms=int((time.perf_counter() - started) * 1000),
            organization_id=body.organization_id,
            account_id=principal.account_id,
        )
        raise
    except Exception as exc:
        await log_agent(
            supabase,
            action="search",
            status="error",
            input_data={
                **body.model_dump(mode="json"),
                "account_id": principal.account_id,
            },
            error_message=str(exc),
            duration_ms=int((time.perf_counter() - started) * 1000),
            organization_id=body.organization_id,
            account_id=principal.account_id,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Supabase RPC match_trade_items hatası: {exc}",
        ) from exc

    results = [TradeItemMatch.model_validate(row) for row in rows]
    results = await paywall_matches(request, supabase, results)
    await log_agent(
        supabase,
        action="search",
        status="success",
        input_data={
            **body.model_dump(mode="json"),
            "account_id": principal.account_id,
        },
        output_data={"result_count": len(results)},
        duration_ms=int((time.perf_counter() - started) * 1000),
        organization_id=body.organization_id,
        account_id=principal.account_id,
    )
    return SearchResponse(query=body.query, model=OLLAMA_EMBED_MODEL, results=results)


@app.post("/consult", response_model=ConsultResponse)
async def consult(body: ConsultRequest, request: Request) -> ConsultResponse:
    principal = require_principal(request)
    if body.session_id:
        try:
            hydrate(
                body.session_id,
                None,
                account_id=principal.account_id,
                created_by_user_id=principal.user_id,
                enforce_ownership=True,
            )
        except SessionAccessDenied:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.") from None
    supabase = await get_supabase(request)
    http = get_http(request)
    started = time.perf_counter()
    try:
        response = await _run_consult_pipeline(
            body,
            http,
            supabase,
            account_slug=principal.account_slug,
            account_id=principal.account_id,
            user_id=principal.user_id,
        )
        response.context = await paywall_matches(request, supabase, response.context)
    except SessionAccessDenied:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı.") from None
    except HTTPException as exc:
        await log_agent(
            supabase,
            action="consult",
            status="error",
            input_data={
                **body.model_dump(mode="json"),
                "account_id": principal.account_id,
            },
            error_message=str(exc.detail),
            duration_ms=int((time.perf_counter() - started) * 1000),
            organization_id=body.organization_id,
            account_id=principal.account_id,
        )
        raise
    except Exception as exc:
        await log_agent(
            supabase,
            action="consult",
            status="error",
            input_data={
                **body.model_dump(mode="json"),
                "account_id": principal.account_id,
            },
            error_message=str(exc),
            duration_ms=int((time.perf_counter() - started) * 1000),
            organization_id=body.organization_id,
            account_id=principal.account_id,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Danışmanlık üretilemedi: {exc}",
        ) from exc

    await _log_consult_success(
        supabase, body, response, started, account_id=principal.account_id
    )
    return response


@app.post("/consult/stream")
async def consult_stream(body: ConsultRequest, request: Request) -> StreamingResponse:
    principal = require_principal(request)
    if body.session_id:
        try:
            hydrate(
                body.session_id,
                None,
                account_id=principal.account_id,
                created_by_user_id=principal.user_id,
                enforce_ownership=True,
            )
        except SessionAccessDenied:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı.") from None
    supabase = await get_supabase(request)
    http = get_http(request)
    started = time.perf_counter()

    async def event_source():
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def on_step(step: AgentStep) -> None:
            await queue.put(("step", step.to_dict()))

        async def work() -> None:
            try:
                result = await _run_consult_pipeline(
                    body,
                    http,
                    supabase,
                    on_step=on_step,
                    account_slug=principal.account_slug,
                    account_id=principal.account_id,
                    user_id=principal.user_id,
                )
                result.context = await paywall_matches(
                    request, supabase, result.context
                )
                await _log_consult_success(
                    supabase, body, result, started, account_id=principal.account_id
                )
                await queue.put(("complete", result.model_dump(mode="json")))
            except SessionAccessDenied:
                await queue.put(
                    ("error", {"detail": "Oturum bulunamadı.", "status": 404})
                )
            except HTTPException as exc:
                await log_agent(
                    supabase,
                    action="consult",
                    status="error",
                    input_data={
                        **body.model_dump(mode="json"),
                        "account_id": principal.account_id,
                    },
                    error_message=str(exc.detail),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    organization_id=body.organization_id,
                    account_id=principal.account_id,
                )
                await queue.put(
                    ("error", {"detail": exc.detail, "status": exc.status_code})
                )
            except Exception as exc:
                await fail_run(supabase, None, str(exc))
                await log_agent(
                    supabase,
                    action="consult",
                    status="error",
                    input_data={
                        **body.model_dump(mode="json"),
                        "account_id": principal.account_id,
                    },
                    error_message=str(exc),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    organization_id=body.organization_id,
                    account_id=principal.account_id,
                )
                await queue.put(("error", {"detail": str(exc), "status": 502}))
            finally:
                await flush_stream_tokens(request, supabase)

        task = asyncio.create_task(work())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "step":
                    message = {"type": "step", "step": payload}
                elif kind == "complete":
                    message = {"type": "complete", "result": payload}
                else:
                    message = {"type": "error", **payload}
                yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
                if kind in {"complete", "error"}:
                    break
        finally:
            await task

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/agent-runs")
async def list_agent_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    principal = require_principal(request)
    supabase = await get_supabase(request)
    try:
        query = (
            supabase.table("agent_runs")
            .select("*")
            .eq("account_id", principal.account_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        result = await query.execute()
        return result.data or []
    except Exception:
        # Column may be missing until migration — never return global rows.
        return []


def _consult_deps(
    body: ConsultRequest,
    http: httpx.AsyncClient,
    supabase: AsyncClient,
    session,
    *,
    account_slug: str,
    account_id: str = "",
) -> AgentDeps:
    question = body.question.strip()
    prior = session.history_dicts() if session is not None else []

    def format_context(matches: list[Any]) -> str:
        return format_consult_context(matches, question)

    async def generate(
        http_client: httpx.AsyncClient,
        prompt: str,
        system_prompt: str | None = None,
        history: list[Any] | None = None,
        polish: bool = True,
    ) -> str:
        turns = history if history is not None else prior
        if turns:
            turns = turns[-8:]
        notes = session_notes(session)
        system = system_prompt or CONSULT_SYSTEM_PROMPT
        if notes:
            system = f"{system}\n\n{notes}"
        return await generate_with_llama3(
            http_client, prompt, system, history=turns, polish=polish
        )

    return AgentDeps(
        http=http,
        supabase=supabase,
        embed=embed_with_ollama,
        match=match_trade_items,
        generate=generate,
        format_context=format_context,
        match_model=TradeItemMatch,
        organization_id=body.organization_id,
        match_count=body.match_count,
        match_threshold=body.match_threshold,
        session=session,
        account_slug=account_slug,
        account_id=account_id or "",
        file_names=list(body.file_names or []),
    )


def _record_canary_consult(
    session_id: str, assigned: str | None, outcome, question: str
) -> None:
    from agents.quality import record_consult_quality

    record_consult_quality(
        session_id=session_id,
        assigned=assigned,
        advice=getattr(outcome, "advice", "") or "",
        question=question,
    )


async def _run_consult_pipeline(
    body: ConsultRequest,
    http: httpx.AsyncClient,
    supabase: AsyncClient,
    on_step=None,
    *,
    account_slug: str,
    account_id: str,
    user_id: str,
) -> ConsultResponse:
    from agents.memory_store import persist_from_turn

    session = hydrate(
        body.session_id,
        body.history,
        account_id=account_id,
        created_by_user_id=user_id,
        enforce_ownership=True,
    )
    slug = account_slug.strip()
    assigned = bind_canary_model(session.session_id)
    with bind_session(session.session_id), reasoning_override(assigned):
        run_id, outcome = await run_orchestrator(
            question=body.question.strip(),
            deps=_consult_deps(
                body,
                http,
                supabase,
                session,
                account_slug=slug,
                account_id=account_id,
            ),
            on_step=on_step,
        )
        session.last_intent = outcome.intent
        session.append("user", body.question.strip())
        session.append("assistant", outcome.advice)
        await persist_from_turn(
            supabase,
            account_slug=slug,
            user_text=body.question.strip(),
            assistant_text=outcome.advice,
            session=session,
        )
        if canary_unload_peer():
            await unload_peer_after(http, assigned or reasoning_model())
    _record_canary_consult(session.session_id, assigned, outcome, body.question.strip())
    context_items = [TradeItemMatch.model_validate(row) for row in outcome.context]
    return ConsultResponse(
        question=body.question,
        advice=outcome.advice,
        embed_model=embed_model(),
        chat_model=chat_model(),
        context=context_items,
        run_id=run_id,
        intent=outcome.intent,
        selected_agent=outcome.selected_agent,
        steps=[AgentStepOut.model_validate(step.to_dict()) for step in outcome.steps],
        session_id=session.session_id,
        history=[ChatTurnIn.model_validate(turn.to_dict()) for turn in session.messages],
        session_title=session_title(session),
        product=session.product,
        capacity=session.capacity,
        market=session.market,
    )


def _consult_success_output(body: ConsultRequest, response: ConsultResponse) -> dict[str, Any]:
    from agents.quality import last_consult_log_telemetry, merge_consult_log_output

    tel = last_consult_log_telemetry()
    sid = response.session_id or getattr(body, "session_id", None) or tel.get("session_id") or ""
    if sid:
        tel["session_id"] = sid
    return merge_consult_log_output(
        {
            "question": response.question,
            "advice": response.advice,
            "context_count": len(response.context),
            "chat_model": response.chat_model,
            "embed_model": response.embed_model,
            "intent": response.intent,
            "selected_agent": response.selected_agent,
            "run_id": response.run_id,
            "steps": [step.model_dump() for step in response.steps],
            "context": [
                {
                    "id": str(item.id),
                    "product_name": item.product_name,
                    "hs_code": item.hs_code,
                    "origin_country": item.origin_country,
                    "destination_country": item.destination_country,
                    "similarity": item.similarity,
                }
                for item in response.context
            ],
        },
        tel,
    )


async def _log_consult_success(
    supabase: AsyncClient,
    body: ConsultRequest,
    response: ConsultResponse,
    started: float,
    *,
    account_id: str | None = None,
) -> None:
    input_data = body.model_dump(mode="json")
    if account_id:
        input_data["account_id"] = account_id
        if response.session_id:
            input_data["session_id"] = response.session_id
    await log_agent(
        supabase,
        action="consult",
        status="success",
        input_data=input_data,
        output_data=_consult_success_output(body, response),
        duration_ms=int((time.perf_counter() - started) * 1000),
        organization_id=body.organization_id,
        account_id=account_id,
    )


@app.post("/organizations")
async def create_organization(body: OrganizationCreate, request: Request) -> dict[str, Any]:
    supabase = await get_supabase(request)
    payload = body.model_dump(mode="json")
    if payload.get("country_code"):
        payload["country_code"] = payload["country_code"].upper()
    try:
        result = await supabase.table("organizations").insert(payload).execute()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result.data:
        raise HTTPException(status_code=400, detail="Kuruluş oluşturulamadı.")
    return result.data[0]


@app.get("/organizations")
async def list_organizations(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    supabase = await get_supabase(request)
    result = (
        await supabase.table("organizations")
        .select(
            "id, name, slug, country_code, city, organization_type, "
            "website, contact_email, tax_id, created_at"
        )
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    plan_id = await plan_id_for_request(request, supabase)
    return redact_records(result.data or [], plan_id)


@app.post("/trade-items")
async def create_trade_item(body: TradeItemCreate, request: Request) -> dict[str, Any]:
    principal = require_principal(request)
    supabase = await get_supabase(request)
    started = time.perf_counter()
    embedding_text = build_embedding_text(
        product_name=body.product_name,
        description=body.description,
        hs_code=body.hs_code,
        origin_country=body.origin_country,
        destination_country=body.destination_country,
    )
    try:
        embedding = await embed_with_ollama(get_http(request), embedding_text)
        payload = trade_item_to_row(body, embedding, embedding_text)
        result = await supabase.table("trade_items").insert(payload).execute()
    except HTTPException:
        raise
    except Exception as exc:
        await log_agent(
            supabase,
            action="ingest_trade_item",
            status="error",
            input_data={
                **body.model_dump(mode="json"),
                "account_id": principal.account_id,
            },
            error_message=str(exc),
            duration_ms=int((time.perf_counter() - started) * 1000),
            organization_id=body.organization_id,
            account_id=principal.account_id,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not result.data:
        raise HTTPException(status_code=400, detail="Ticaret kalemi oluşturulamadı.")

    created = result.data[0]
    await log_agent(
        supabase,
        action="ingest_trade_item",
        status="success",
        input_data={
            "product_name": body.product_name,
            "hs_code": body.hs_code,
            "account_id": principal.account_id,
        },
        output_data={"id": created.get("id")},
        duration_ms=int((time.perf_counter() - started) * 1000),
        organization_id=body.organization_id,
        trade_item_id=UUID(created["id"]) if created.get("id") else None,
        account_id=principal.account_id,
    )
    return created


@app.post("/trade-items/bulk", response_model=TradeItemBulkResponse)
async def bulk_create_trade_items(
    body: TradeItemBulkRequest,
    request: Request,
) -> TradeItemBulkResponse:
    principal = require_principal(request)
    supabase = await get_supabase(request)
    http = get_http(request)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[TradeItemBulkItemError] = []

    for index, item in enumerate(body.items):
        embedding_text = build_embedding_text(
            product_name=item.product_name,
            description=item.description,
            hs_code=item.hs_code,
            origin_country=item.origin_country,
            destination_country=item.destination_country,
        )
        try:
            embedding = await embed_with_ollama(http, embedding_text)
            rows.append(trade_item_to_row(item, embedding, embedding_text))
        except HTTPException as exc:
            if exc.status_code == 503:
                await log_agent(
                    supabase,
                    action="ingest_trade_items_bulk",
                    status="error",
                    input_data={
                        "count": len(body.items),
                        "failed_at_index": index,
                        "account_id": principal.account_id,
                    },
                    error_message=str(exc.detail),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    account_id=principal.account_id,
                )
                raise
            errors.append(
                TradeItemBulkItemError(
                    index=index,
                    product_name=item.product_name,
                    error=str(exc.detail),
                )
            )
        except Exception as exc:
            errors.append(
                TradeItemBulkItemError(
                    index=index,
                    product_name=item.product_name,
                    error=str(exc),
                )
            )

    inserted: list[dict[str, Any]] = []
    if rows:
        try:
            result = await supabase.table("trade_items").insert(rows).execute()
            inserted = result.data or []
        except Exception as exc:
            await log_agent(
                supabase,
                action="ingest_trade_items_bulk",
                status="error",
                input_data={
                    "count": len(body.items),
                    "prepared": len(rows),
                    "account_id": principal.account_id,
                },
                error_message=str(exc),
                duration_ms=int((time.perf_counter() - started) * 1000),
                account_id=principal.account_id,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Toplu kayıt başarısız: {exc}",
            ) from exc

    await log_agent(
        supabase,
        action="ingest_trade_items_bulk",
        status="success" if inserted and not errors else ("error" if not inserted else "success"),
        input_data={"count": len(body.items), "account_id": principal.account_id},
        output_data={"inserted": len(inserted), "failed": len(errors)},
        duration_ms=int((time.perf_counter() - started) * 1000),
        account_id=principal.account_id,
    )
    return TradeItemBulkResponse(
        inserted=len(inserted),
        failed=len(errors),
        embed_model=OLLAMA_EMBED_MODEL,
        items=inserted,
        errors=errors,
    )


@app.get("/trade-items")
async def list_trade_items(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    organization_id: UUID | None = None,
) -> list[dict[str, Any]]:
    supabase = await get_supabase(request)
    query = (
        supabase.table("trade_items")
        .select(TRADE_ITEM_PAYWALL_SELECT)
        .order("created_at", desc=True)
        .limit(limit)
    )
    if organization_id:
        query = query.eq("organization_id", str(organization_id))
    try:
        result = await query.execute()
        rows = result.data or []
    except Exception:
        fallback = (
            supabase.table("trade_items")
            .select(
                "id, organization_id, hs_code, product_name, description, "
                "origin_country, destination_country, direction, quantity, unit, "
                "value_usd, trade_date, source, embedding_model, created_at"
            )
            .order("created_at", desc=True)
            .limit(limit)
        )
        if organization_id:
            fallback = fallback.eq("organization_id", str(organization_id))
        result = await fallback.execute()
        rows = result.data or []
    return await apply_item_paywall(request, supabase, rows)


@app.get("/suppliers")
async def list_suppliers(
    request: Request,
    limit: int = Query(default=40, ge=1, le=200),
) -> dict[str, Any]:
    supabase = await get_supabase(request)
    items = await list_trade_items(request, limit=limit)
    organizations: list[dict[str, Any]] = []
    try:
        org_result = (
            await supabase.table("organizations")
            .select(
                "id, name, slug, country_code, city, organization_type, "
                "website, contact_email, tax_id, created_at"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        plan_id = await plan_id_for_request(request, supabase)
        organizations = redact_records(org_result.data or [], plan_id)
    except Exception:
        organizations = []
    return {
        "plan_id": items[0]["plan_id"] if items else (
            await plan_id_for_request(request, supabase)
        ),
        "items": items,
        "organizations": organizations,
    }


@app.post("/agent-logs")
async def create_agent_log(body: AgentLogCreate, request: Request) -> dict[str, Any]:
    principal = require_principal(request)
    supabase = await get_supabase(request)
    payload = body.model_dump(mode="json")
    for field in ("organization_id", "trade_item_id"):
        if payload.get(field):
            payload[field] = str(payload[field])
    payload["account_id"] = principal.account_id
    payload["input"] = {
        **(payload.get("input") or {}),
        "account_id": principal.account_id,
    }
    try:
        result = await supabase.table("agent_logs").insert(payload).execute()
    except Exception as exc:
        payload.pop("account_id", None)
        try:
            result = await supabase.table("agent_logs").insert(payload).execute()
        except Exception as exc2:
            raise HTTPException(status_code=400, detail=str(exc2)) from exc2
    if not result.data:
        raise HTTPException(status_code=400, detail="Ajan kaydı oluşturulamadı.")
    return result.data[0]


@app.get("/agent-logs")
async def list_agent_logs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    action: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    principal = require_principal(request)
    try:
        supabase = await asyncio.wait_for(
            get_supabase(request), timeout=HISTORY_FETCH_TIMEOUT_S
        )
        query = (
            supabase.table("agent_logs")
            .select("*")
            .eq("account_id", principal.account_id)
            .order("created_at", desc=True)
        )
        if action:
            query = query.eq("action", action)
        result = await asyncio.wait_for(
            query.limit(limit).execute(), timeout=HISTORY_FETCH_TIMEOUT_S
        )
        return result.data or []
    except Exception:
        # Prefer input.account_id filter if column missing.
        try:
            supabase = await get_supabase(request)
            query = (
                supabase.table("agent_logs")
                .select("*")
                .filter("input->>account_id", "eq", principal.account_id)
                .order("created_at", desc=True)
                .limit(limit)
            )
            if action:
                query = query.eq("action", action)
            result = await query.execute()
            return result.data or []
        except Exception:
            return []


@app.post("/consult/sessions")
async def new_consult_session(request: Request) -> dict[str, Any]:
    """Yeni izole sohbet oturumu — authenticated account ownership."""
    principal = require_principal(request)
    state = create_session(
        account_id=principal.account_id,
        created_by_user_id=principal.user_id,
    )
    return session_snapshot(state)


@app.get("/consult/sessions")
async def list_consult_sessions(request: Request) -> list[dict[str, Any]]:
    principal = require_principal(request)
    return list_sessions(account_id=principal.account_id)


@app.get("/consult/sessions/{session_id}")
async def get_consult_session(session_id: str, request: Request) -> dict[str, Any]:
    principal = require_principal(request)
    state = get_session_for_account(session_id, principal.account_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
    return session_snapshot(state)


@app.delete("/consult/sessions/{session_id}")
async def remove_consult_session(
    session_id: str,
    request: Request,
) -> dict[str, Any]:
    """Sohbet oturumunu bellekten ve varsa ajan kayıtlarından siler."""
    principal = require_principal(request)
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Oturum kimliği gerekli.")
    deleted = delete_session(sid, account_id=principal.account_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
    asyncio.create_task(
        _purge_consult_logs(request, sid, account_id=principal.account_id)
    )
    return {"ok": True, "session_id": sid, "deleted": deleted}


HISTORY_FETCH_TIMEOUT_S = 8.0


async def _purge_consult_logs(
    request: Request, session_id: str, *, account_id: str | None = None
) -> None:
    try:
        supabase = await asyncio.wait_for(
            get_supabase(request), timeout=HISTORY_FETCH_TIMEOUT_S
        )
        q = (
            supabase.table("agent_logs")
            .delete()
            .eq("action", "consult")
            .filter("input->>session_id", "eq", session_id)
        )
        if account_id:
            q = q.eq("account_id", account_id)
        await asyncio.wait_for(q.execute(), timeout=HISTORY_FETCH_TIMEOUT_S)
    except Exception:
        return


async def _fetch_consult_history(
    request: Request,
    limit: int,
) -> list[dict[str, Any]]:
    principal = require_principal(request)
    try:
        supabase = await asyncio.wait_for(
            get_supabase(request), timeout=HISTORY_FETCH_TIMEOUT_S
        )
        result = await asyncio.wait_for(
            supabase.table("agent_logs")
            .select("*")
            .eq("action", "consult")
            .eq("account_id", principal.account_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute(),
            timeout=HISTORY_FETCH_TIMEOUT_S,
        )
        return result.data or []
    except Exception:
        try:
            supabase = await get_supabase(request)
            result = await (
                supabase.table("agent_logs")
                .select("*")
                .eq("action", "consult")
                .filter("input->>account_id", "eq", principal.account_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception:
            return []


@app.get("/consult/history")
async def consult_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    return await _fetch_consult_history(request, limit)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
