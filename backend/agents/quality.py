"""CMO kalite izi. Kullanıcıya gösterilmez; log + test içindir."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

log = logging.getLogger("wtr.quality")

ComposePath = Literal[
    "LLM_CMO_SUCCESS",
    "LLM_CMO_RETRY_SUCCESS",
    "COMMERCIAL_FALLBACK",
    "PLAYBOOK_FALLBACK",
    "REJECTED_NO_SAFE_RESPONSE",
]


@dataclass
class ComposeTrace:
    text: str
    path: ComposePath
    llm_direct: str = ""
    llm_retry: str = ""
    commercial_fallback: str = ""
    playbook: str = ""
    direct_status: str = ""
    retry_status: str = ""
    english_leak: bool = False
    unsupported_fact: bool = False
    brief_echo: bool = False
    memory_dishonesty: bool = False
    false_certainty: bool = False
    question: str = ""
    model: str = ""
    think: bool | None = None
    num_predict: int | None = None
    latency_s: float | None = None
    eval_count: int | None = None
    tokens_per_sec: float | None = None
    done_reason: str = ""
    content_length: int = 0
    thinking_length: int = 0
    first_done_reason: str = ""

    @property
    def public_path(self) -> str:
        return {
            "LLM_CMO_SUCCESS": "LLM_DIRECT",
            "LLM_CMO_RETRY_SUCCESS": "LLM_RETRY",
            "COMMERCIAL_FALLBACK": "COMMERCIAL_FALLBACK",
            "PLAYBOOK_FALLBACK": "PLAYBOOK_FALLBACK",
            "REJECTED_NO_SAFE_RESPONSE": "SAFE_FALLBACK",
        }.get(self.path, self.path)


@dataclass
class QualitySnapshot:
    n: int = 0
    paths: Counter = field(default_factory=Counter)
    english_leaks: int = 0
    unsupported_facts: int = 0
    decision_without_reason: int = 0
    decision_without_action: int = 0
    unnecessary_questions: int = 0
    brief_echoes: int = 0
    memory_dishonest: int = 0
    false_certainties: int = 0

    @property
    def llm_direct_success_rate(self) -> float:
        return self._rate("LLM_CMO_SUCCESS")

    @property
    def llm_retry_success_rate(self) -> float:
        return self._rate("LLM_CMO_RETRY_SUCCESS")

    @property
    def commercial_fallback_rate(self) -> float:
        return self._rate("COMMERCIAL_FALLBACK")

    @property
    def playbook_fallback_rate(self) -> float:
        return self._rate("PLAYBOOK_FALLBACK")

    @property
    def rejection_rate(self) -> float:
        return self._rate("REJECTED_NO_SAFE_RESPONSE")

    @property
    def english_leakage_rate(self) -> float:
        return self.english_leaks / self.n if self.n else 0.0

    @property
    def unsupported_fact_rate(self) -> float:
        return self.unsupported_facts / self.n if self.n else 0.0

    @property
    def decision_without_reason_rate(self) -> float:
        return self.decision_without_reason / self.n if self.n else 0.0

    @property
    def decision_without_action_rate(self) -> float:
        return self.decision_without_action / self.n if self.n else 0.0

    @property
    def unnecessary_question_rate(self) -> float:
        return self.unnecessary_questions / self.n if self.n else 0.0

    @property
    def real_llm_success_rate(self) -> float:
        if not self.n:
            return 0.0
        return (
            self.paths.get("LLM_CMO_SUCCESS", 0)
            + self.paths.get("LLM_CMO_RETRY_SUCCESS", 0)
        ) / self.n

    @property
    def brief_echo_rate(self) -> float:
        return self.brief_echoes / self.n if self.n else 0.0

    def _rate(self, path: str) -> float:
        return self.paths.get(path, 0) / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "llm_direct_success_rate": self.llm_direct_success_rate,
            "llm_retry_success_rate": self.llm_retry_success_rate,
            "commercial_fallback_rate": self.commercial_fallback_rate,
            "playbook_fallback_rate": self.playbook_fallback_rate,
            "rejection_rate": self.rejection_rate,
            "english_leakage_rate": self.english_leakage_rate,
            "unsupported_fact_rate": self.unsupported_fact_rate,
            "decision_without_reason_rate": self.decision_without_reason_rate,
            "decision_without_action_rate": self.decision_without_action_rate,
            "unnecessary_question_rate": self.unnecessary_question_rate,
            "real_llm_success_rate": self.real_llm_success_rate,
            "brief_echo_rate": self.brief_echo_rate,
        }


_FALLBACK_PATHS = frozenset(
    {
        "COMMERCIAL_FALLBACK",
        "PLAYBOOK_FALLBACK",
        "REJECTED_NO_SAFE_RESPONSE",
    }
)
_PII_LOG_KEYS = frozenset(
    {
        "thinking",
        "raw_thinking",
        "message",
        "prompt",
        "response",
        "advice",
        "question",
        "text",
        "wav",
        "user_text",
        "assistant_text",
        "history",
        "context",
    }
)

_BOOK: list[ComposeTrace] = []
_LAST_INFERENCE: dict = {}
_ROUTE: list[dict] = []
_CANARY: list[dict] = []
_LAST_TTS: dict = {}
_LAST_MODEL: str = ""
_LAST_CONSULT_LOG: dict = {}


def reset_quality() -> None:
    global _LAST_MODEL
    _BOOK.clear()
    _LAST_INFERENCE.clear()
    _ROUTE.clear()
    _CANARY.clear()
    _LAST_TTS.clear()
    _LAST_CONSULT_LOG.clear()
    _LAST_MODEL = ""
    from .shadow import reset_shadow

    reset_shadow()


def record_inference(meta: dict) -> None:
    """Internal telemetry. thinking metni saklanmaz."""
    global _LAST_MODEL
    _LAST_INFERENCE.clear()
    safe = {
        k: v
        for k, v in (meta or {}).items()
        if k not in ("thinking", "raw_thinking", "message", "prompt", "response")
    }
    model = str(safe.get("model") or "")
    prev = _LAST_MODEL
    if model and prev and model.casefold() != prev.casefold():
        safe["model_switch"] = f"{prev}->{model}"
    if model:
        _LAST_MODEL = model
    _LAST_INFERENCE.update(safe)
    _ROUTE.append(dict(safe))
    if len(_ROUTE) > 400:
        del _ROUTE[: len(_ROUTE) - 400]


def last_inference() -> dict:
    return dict(_LAST_INFERENCE)


def record_tts(meta: dict) -> None:
    _LAST_TTS.clear()
    safe = {
        k: v
        for k, v in (meta or {}).items()
        if k not in ("thinking", "text", "wav", "prompt")
    }
    _LAST_TTS.update(safe)


def last_tts() -> dict:
    return dict(_LAST_TTS)


def record_canary_turn(meta: dict) -> None:
    """Canary/production turn telemetry. Raw thinking yok."""
    safe = {
        k: v
        for k, v in (meta or {}).items()
        if k not in ("thinking", "raw_thinking", "prompt", "response", "advice")
    }
    _CANARY.append(safe)
    if len(_CANARY) > 400:
        del _CANARY[: len(_CANARY) - 400]
    log.info(
        "canary model=%s bucket=%s path=%s first=%s retry=%s done=%s lat=%s switch=%s",
        safe.get("model") or "-",
        safe.get("session_bucket"),
        safe.get("final_path") or "-",
        safe.get("validator_first_shot") or "-",
        safe.get("retry") or "-",
        safe.get("done_reason") or "-",
        safe.get("latency_s"),
        safe.get("model_switch") or "-",
    )


def canary_turns() -> list[dict]:
    return list(_CANARY)


def last_consult_log_telemetry() -> dict:
    """Last persistable consult fields for agent_logs.output. No PII text."""
    return dict(_LAST_CONSULT_LOG)


def persistable_consult_telemetry(
    *,
    reasoning_model: str,
    canary_qwen: bool,
    validator_result: str = "",
    retry: str = "",
    compose_path: str = "",
    latency_s: float | None = None,
    cmo_total: int | None = None,
    decision: int | None = None,
    false_certainty: bool = False,
    unsupported_fact: bool = False,
    brief_echo: bool = False,
    session_id: str | None = None,
) -> dict:
    """Durable consult metrics for agent_logs.output.

    decision is the existing CMO DECISION dimension (0–2), not a new classifier.
    latency_ms is Ollama/model wall time. Pipeline duration stays duration_ms.
    chat_model is never copied into reasoning_model.
    """
    latency_ms = None
    if isinstance(latency_s, (int, float)):
        latency_ms = int(round(float(latency_s) * 1000))
    payload = {
        "reasoning_model": reasoning_model or "",
        "canary_qwen": bool(canary_qwen),
        "model_path": "canary" if canary_qwen else "default",
        "validator_result": validator_result or "",
        "retry": retry or "",
        "fallback": compose_path in _FALLBACK_PATHS,
        "latency_ms": latency_ms,
        "cmo_total": cmo_total,
        "decision": decision,
        "false_certainty": bool(false_certainty),
        "unsupported_fact": bool(unsupported_fact),
        "brief_echo": bool(brief_echo),
        "session_id": session_id or "",
    }
    return {k: v for k, v in payload.items() if k not in _PII_LOG_KEYS and k != "chat_model"}


def record_consult_quality(
    *,
    session_id: str,
    assigned: str | None,
    advice: str,
    question: str,
) -> dict:
    """Score existing CMO/validator traces and stash persistable consult telemetry.

    Does not change evaluator/validator. Does not store prompt or advice.
    """
    from .evaluator import score_cmo_v51
    from llm import canary_bucket, reasoning_model

    inf = last_inference()
    book = traces()
    trace = book[-1] if book else None
    scored = score_cmo_v51(advice or "", question=question)
    model = inf.get("model") or assigned or reasoning_model()
    path = getattr(trace, "path", "") if trace else ""
    validator_result = getattr(trace, "direct_status", "") if trace else ""
    retry = getattr(trace, "retry_status", "") if trace else ""
    record_canary_turn(
        {
            "model": model,
            "task": inf.get("task") or "commercial",
            "session_id": session_id or "",
            "session_bucket": canary_bucket(session_id) if session_id else None,
            "canary_qwen": bool(assigned),
            "latency_s": inf.get("latency_s"),
            "prompt_eval_s": inf.get("prompt_eval_s"),
            "generation_s": inf.get("gen_s"),
            "load_s": inf.get("load_s"),
            "tokens": inf.get("eval_count"),
            "tokens_per_sec": inf.get("tokens_per_sec"),
            "done_reason": inf.get("done_reason") or "",
            "validator_first_shot": validator_result,
            "retry": retry,
            "final_path": path,
            "cmo_score": scored.total,
            "decision": scored.decision,
            "BRIEF_ECHO": scored.brief_echo,
            "ENGLISH": scored.english_leakage,
            "FALSE_CERTAINTY": scored.false_certainty,
            "UNSUPPORTED_FACT": scored.unsupported_fact,
            "MEMORY_DISHONESTY": scored.memory_dishonesty,
            "PROMPT_LEAK": scored.prompt_leak,
            "tts_latency_s": last_tts().get("latency_s"),
            "model_switch": inf.get("model_switch") or "",
            "thinking_length": inf.get("thinking_length") or 0,
            "content_empty": not bool((inf.get("content_length") or 0)),
        }
    )
    payload = persistable_consult_telemetry(
        reasoning_model=str(model or ""),
        canary_qwen=bool(assigned),
        validator_result=validator_result,
        retry=retry,
        compose_path=path,
        latency_s=inf.get("latency_s"),
        cmo_total=scored.total,
        decision=scored.decision,
        false_certainty=scored.false_certainty,
        unsupported_fact=scored.unsupported_fact,
        brief_echo=scored.brief_echo,
        session_id=session_id,
    )
    _LAST_CONSULT_LOG.clear()
    _LAST_CONSULT_LOG.update(payload)
    return payload


def merge_consult_log_output(base: dict, telemetry: dict | None = None) -> dict:
    """Attach persistable metrics without overwriting public chat_model."""
    out = dict(base or {})
    tel = dict(telemetry if telemetry is not None else _LAST_CONSULT_LOG)
    tel.pop("chat_model", None)
    for key in _PII_LOG_KEYS:
        tel.pop(key, None)
    reasoning = tel.get("reasoning_model")
    out.update(tel)
    if reasoning is not None:
        out["reasoning_model"] = reasoning
    return out


def canary_snapshot() -> dict:
    rows = list(_CANARY)
    n = len(rows)
    qwen = [r for r in rows if "qwen" in str(r.get("model") or "").casefold()]
    llama = [r for r in rows if "llama" in str(r.get("model") or "").casefold()]
    return {
        "total": n,
        "qwen": len(qwen),
        "llama3": len(llama),
        "qwen_share": round(len(qwen) / n, 3) if n else 0.0,
    }


def route_snapshot() -> dict:
    rows = list(_ROUTE)
    n = len(rows)
    qwen = sum(1 for r in rows if "qwen" in str(r.get("model") or "").casefold())
    llama = sum(1 for r in rows if "llama" in str(r.get("model") or "").casefold())
    length = sum(1 for r in rows if r.get("done_reason") == "length")
    errors = sum(1 for r in rows if r.get("error"))
    lats = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float))]
    return {
        "total_requests": n,
        "qwen_requests": qwen,
        "llama3_requests": llama,
        "done_reason_length": length,
        "done_reason_length_pct": round(length / n, 3) if n else 0.0,
        "errors": errors,
        "avg_latency_s": round(sum(lats) / len(lats), 3) if lats else None,
    }


def done_reason_stats(items: list[ComposeTrace] | None = None) -> dict:
    rows = items if items is not None else _BOOK
    n = len(rows) or 0
    length = sum(1 for r in rows if (r.done_reason or "") == "length")
    return {
        "n": n,
        "length": length,
        "pct": round(length / n, 3) if n else 0.0,
    }


def record_trace(trace: ComposeTrace) -> None:
    _BOOK.append(trace)
    log.info(
        "compose_path=%s direct=%s retry=%s leak=%s fact=%s model=%s think=%s "
        "done=%s eval=%s content_len=%s thinking_len=%s",
        trace.path,
        trace.direct_status or "-",
        trace.retry_status or "-",
        trace.english_leak,
        trace.unsupported_fact,
        trace.model or "-",
        trace.think,
        trace.done_reason or "-",
        trace.eval_count,
        trace.content_length,
        trace.thinking_length,
    )


def traces() -> list[ComposeTrace]:
    return list(_BOOK)


def snapshot(items: list[ComposeTrace] | None = None) -> QualitySnapshot:
    rows = items if items is not None else _BOOK
    snap = QualitySnapshot(n=len(rows))
    for row in rows:
        snap.paths[row.path] += 1
        if row.english_leak:
            snap.english_leaks += 1
        if row.unsupported_fact:
            snap.unsupported_facts += 1
        if getattr(row, "brief_echo", False):
            snap.brief_echoes += 1
        if getattr(row, "memory_dishonesty", False):
            snap.memory_dishonest += 1
        if getattr(row, "false_certainty", False):
            snap.false_certainties += 1
        body = (row.text or "").casefold()
        if row.question and ("mı" in (row.question or "").casefold() or "mi" in (row.question or "").casefold()):
            if "çünkü" not in body and "neden" not in body and "için" not in body:
                snap.decision_without_reason += 1
            if not any(tok in body for tok in ("önce", "sonraki", "adım", "ölç", "karşılaştır")):
                snap.decision_without_action += 1
        if body.count("?") > 3:
            snap.unnecessary_questions += 1
    return snap
