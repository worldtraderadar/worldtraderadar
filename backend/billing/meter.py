from __future__ import annotations

from contextvars import ContextVar

_token_bucket: ContextVar[list[int] | None] = ContextVar(
    "quota_token_bucket", default=None
)


def init_token_meter() -> list[int]:
    bucket = [0]
    _token_bucket.set(bucket)
    return bucket


def add_tokens(count: int) -> None:
    if count <= 0:
        return
    bucket = _token_bucket.get()
    if bucket is None:
        return
    bucket[0] += int(count)


def peek_tokens() -> int:
    bucket = _token_bucket.get()
    if not bucket:
        return 0
    return bucket[0]


def take_tokens() -> int:
    bucket = _token_bucket.get()
    if not bucket:
        return 0
    used = bucket[0]
    bucket[0] = 0
    return used
