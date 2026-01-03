"""Shared retry helper for backend requests."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

import httpx

T = TypeVar("T")


class BackendUnavailableError(RuntimeError):
    """Raised when backend retries are exhausted."""


async def retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    retries: int = 0,
    backoff: float = 0.25,
) -> T:
    """Retry an async callable, raising BackendUnavailableError when exhausted."""
    for attempt in range(retries + 1):
        try:
            return await fn(*args)
        except httpx.RequestError as exc:
            if attempt >= retries:
                raise BackendUnavailableError("LMStudio backend unavailable") from exc
            await asyncio.sleep(backoff * (2**attempt))
    raise BackendUnavailableError("LMStudio backend unavailable")
