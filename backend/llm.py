"""Ollama sohbet ve gömme modeli.

Persona ve reasoning modelden bağımsızdır.
Sohbet modeli yalnızca OLLAMA_CHAT_MODEL ile seçilir (llama3 varsayılan).
Routing env üzerinden model bağımsızdır. Production default llama3 kalır.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Literal

ModelTask = Literal["chat", "classify", "decision", "diagnostic", "retrieve", "commercial"]

QWEN_27B = "qwen3.5:27b"
QWEN_NUM_PREDICT = 280

_reasoning_override: ContextVar[str | None] = ContextVar(
    "wtr_reasoning_override", default=None
)
_session_id: ContextVar[str] = ContextVar("wtr_session_id", default="")


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def chat_model() -> str:
    return (os.getenv("OLLAMA_CHAT_MODEL") or "llama3").strip() or "llama3"


def embed_model() -> str:
    return (os.getenv("OLLAMA_EMBED_MODEL") or "bge-m3").strip() or "bge-m3"


def reasoning_model() -> str:
    override = _reasoning_override.get()
    if override:
        return override
    return (os.getenv("OLLAMA_REASONING_MODEL") or "").strip() or chat_model()


def fast_model() -> str:
    return (os.getenv("OLLAMA_FAST_MODEL") or "").strip() or chat_model()


def reasoning_think() -> bool:
    """V5.3 default: think=false. Thinking asla kullanıcı cevabı değildir."""
    return _env_flag("OLLAMA_REASONING_THINK", False)


def shadow_model() -> str:
    """Boş = kapalı. Canary açıkken shadow zorunlu kapalı (VRAM)."""
    if canary_enabled():
        return ""
    return (os.getenv("OLLAMA_SHADOW_MODEL") or "").strip()


def ollama_num_predict() -> int | None:
    raw = (os.getenv("OLLAMA_NUM_PREDICT") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def ollama_keep_alive() -> str:
    return (os.getenv("OLLAMA_KEEP_ALIVE") or "30m").strip() or "30m"


def canary_enabled() -> bool:
    """Default OFF. Açıkken session-hash ile Qwen payı seçilir."""
    return _env_flag("MODEL_CANARY_ENABLED", False)


def canary_percent() -> int:
    raw = (os.getenv("MODEL_CANARY_PERCENT") or "").strip()
    if not raw:
        return 10 if canary_enabled() else 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, min(100, value))


def canary_target() -> str:
    return (os.getenv("MODEL_CANARY_TARGET") or QWEN_27B).strip() or QWEN_27B


def canary_bucket(session_key: str) -> int:
    key = (session_key or "").strip() or "anon"
    digest = hashlib.sha256(f"wtr-canary:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def canary_assigns_qwen(session_key: str) -> bool:
    if not canary_enabled():
        return False
    pct = canary_percent()
    if pct <= 0:
        return False
    if pct >= 100:
        return True
    return canary_bucket(session_key) < pct


def bind_canary_model(session_key: str) -> str | None:
    """None = env reasoning (llama3). Qwen yalnızca canary atamasında."""
    if canary_assigns_qwen(session_key):
        return canary_target()
    return None


@contextmanager
def reasoning_override(model: str | None) -> Iterator[None]:
    if not model:
        yield
        return
    token = _reasoning_override.set(model)
    try:
        yield
    finally:
        _reasoning_override.reset(token)


@contextmanager
def bind_session(session_id: str | None) -> Iterator[None]:
    token = _session_id.set((session_id or "").strip())
    try:
        yield
    finally:
        _session_id.reset(token)


def current_session_id() -> str:
    return _session_id.get() or ""


def default_num_predict(model: str) -> int | None:
    env = ollama_num_predict()
    if env:
        return env
    if "qwen3.5" in (model or "").casefold():
        return QWEN_NUM_PREDICT
    return None


def model_for_task(kind: ModelTask | str) -> str:
    """Karar/teşhis → reasoning; sohbet/sınıf → fast; retrieval → embed. Varsayılan chat."""
    if kind in ("decision", "diagnostic", "commercial"):
        return reasoning_model()
    if kind in ("chat", "classify"):
        return fast_model()
    if kind == "retrieve":
        return embed_model()
    return chat_model()


def ollama_base_url() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")


def user_content_from_ollama(data: dict[str, Any]) -> str:
    """Yalnızca message.content. thinking asla content değildir."""
    msg = data.get("message") if isinstance(data.get("message"), dict) else {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if not content:
        content = data.get("response")
    return content.strip() if isinstance(content, str) else ""


def thinking_from_ollama(data: dict[str, Any]) -> str:
    msg = data.get("message") if isinstance(data.get("message"), dict) else {}
    thinking = msg.get("thinking") if isinstance(msg, dict) else None
    if not thinking:
        thinking = data.get("thinking")
    return thinking if isinstance(thinking, str) else ""


def user_visible_text(content: str, thinking: str = "") -> str:
    """thinking → user/TTS/history yasak. Yalnızca content."""
    del thinking
    return (content or "").strip()


def ollama_chat_payload(
    model: str,
    messages: list[dict[str, str]],
    *,
    think: bool | None = None,
    options: dict[str, Any] | None = None,
    keep_alive: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": options or {},
    }
    flag = reasoning_think() if think is None else think
    body["think"] = bool(flag)
    alive = ollama_keep_alive() if keep_alive is None else keep_alive
    if alive:
        body["keep_alive"] = alive
    opts = dict(body["options"] or {})
    npred = default_num_predict(model)
    if npred and "num_predict" not in opts:
        opts["num_predict"] = npred
    body["options"] = opts
    return body


def canary_unload_peer() -> bool:
    """Canary açıkken kullanılmayan reasoning modelini unload et.

    Default kapalı (V5.5 resident davranış). MODEL_CANARY_UNLOAD_PEER=true
    ile SAFE lifecycle açılır. Canary kapalıyken her zaman false.
    """
    if not canary_enabled():
        return False
    return _env_flag("MODEL_CANARY_UNLOAD_PEER", False)


def peer_reasoning_model(active: str) -> str | None:
    """Aktif reasoning modelinin eşi. Embed/TTS dokunulmaz."""
    used = (active or "").strip()
    if not used:
        return None
    llama = chat_model()
    qwen = canary_target()
    low = used.casefold()
    llama_l = llama.casefold()
    qwen_l = qwen.casefold()
    if qwen_l in low or "qwen3.5" in low:
        return llama if llama_l != low else None
    if llama_l in low or low.startswith("llama"):
        return qwen if qwen_l != low else None
    return None


async def unload_ollama_model(http: Any, model: str) -> dict[str, Any]:
    """Ollama keep_alive=0. Yeni inference stack yok."""
    name = (model or "").strip()
    if not name:
        return {"ok": False, "model": "", "error": "empty"}
    try:
        response = await http.post(
            f"{ollama_base_url()}/api/generate",
            json={"model": name, "prompt": " ", "keep_alive": 0, "stream": False},
            timeout=60.0,
        )
        return {
            "ok": response.status_code < 400,
            "model": name,
            "status": response.status_code,
            "error": "",
        }
    except Exception as exc:
        return {"ok": False, "model": name, "status": 0, "error": repr(exc)}


async def unload_peer_after(http: Any, active: str) -> dict[str, Any]:
    """Aktif modeli tut, diğer reasoning modelini VRAM'den çıkar."""
    peer = peer_reasoning_model(active)
    if not peer:
        return {"ok": True, "unloaded": None, "skipped": True}
    result = await unload_ollama_model(http, peer)
    result["unloaded"] = peer
    result["active"] = active
    return result
