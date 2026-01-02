"""Factory helpers for the shim."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable

import httpx

from logging_config import request_id_ctx


def coerce_client_factory(
    factory: Any,
    *,
    default_factory: Callable[[], httpx.AsyncClient],
    logger,
) -> Callable[[], Awaitable[httpx.AsyncClient]]:
    """Validate or wrap a client factory.

    The argument may be ``None`` or a callable returning an ``httpx.AsyncClient``,
    including async callables that return an awaitable.
    """
    if not callable(factory):
        async def default_wrapper() -> httpx.AsyncClient:
            return default_factory()

        return default_wrapper

    async def wrapped_factory() -> httpx.AsyncClient:
        result = factory()
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, httpx.AsyncClient):
            logger.error(
                "client_factory returned unexpected type",
                extra={"request_id": request_id_ctx.get("-")},
            )
            raise TypeError("client_factory must return httpx.AsyncClient")
        return result

    return wrapped_factory
