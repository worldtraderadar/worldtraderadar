"""V5.6 reject sınıflandırması. Validator kurallarını değiştirmez."""

from __future__ import annotations

import re

from .validator import ValidationResult, validate_response

# A gerçek kalite, B false positive, C truncation, D format, E unsupported, F başka
CAUSE_LABELS = {
    "A": "gerçek kalite problemi",
    "B": "validator false positive",
    "C": "response truncation",
    "D": "format problemi",
    "E": "unsupported claim",
    "F": "başka",
}

_END_PUNCT = re.compile(r"[.!?;:…]$")


def looks_truncated(text: str, done_reason: str = "") -> bool:
    """done_reason=length ve son karakter cümle sonu değilse kesilmiş say."""
    if str(done_reason or "") != "length":
        return False
    body = (text or "").rstrip()
    if not body:
        return True
    return not bool(_END_PUNCT.search(body[-1] if body else ""))


def classify_reject_cause(
    text: str = "",
    *,
    reasons: list[str] | None = None,
    done_reason: str = "",
    status: str = "",
) -> dict:
    """REJECT için A–F. B yalnızca açık false-positive kanıtında; varsayılan değil."""
    reasons = list(reasons or [])
    if not reasons and (status or "").upper() == "REJECT":
        verdict = validate_response(text)
        reasons = list(verdict.reasons)
        status = verdict.status
    joined = " ".join(reasons).casefold()
    truncated = looks_truncated(text, done_reason)
    code = "F"
    if "kaynaksız rakam" in joined or "kaynaksız firma" in joined:
        code = "E"
    elif (
        "mutlak kesinlik" in joined
        or "hatırlıyorum" in joined
        or "kazanma ihtimali" in joined
        or "başarı iddiası" in joined
        or "güncel olay" in joined
    ):
        code = "A"
    elif (
        "brief" in joined
        or "ingilizce" in joined
        or "mail taslağı" in joined
        or "markdown" in joined
    ):
        code = "D"
    elif truncated:
        code = "C"
    elif "kaynaksız" in joined:
        code = "E"
    elif reasons:
        code = "A"
    elif truncated:
        code = "C"
    return {
        "code": code,
        "label": CAUSE_LABELS[code],
        "reasons": reasons,
        "truncated": truncated,
        "done_reason": done_reason or "",
        "status": status or "",
        "usable_if_length": bool(
            str(done_reason or "") == "length" and not truncated and (status or "") == "PASS"
        ),
    }


def classify_from_verdict(
    text: str,
    verdict: ValidationResult | None = None,
    *,
    done_reason: str = "",
) -> dict:
    result = verdict or validate_response(text)
    return classify_reject_cause(
        text,
        reasons=list(result.reasons),
        done_reason=done_reason,
        status=result.status,
    )
