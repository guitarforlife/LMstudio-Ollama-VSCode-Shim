"""Shared typing helpers."""

from __future__ import annotations

from typing import Any, AsyncContextManager, Protocol

import httpx


class BackendLike(Protocol):
    """Protocol for backend HTTP clients used by the shim."""

    async def get(self, url: str, *, timeout: float | None = None) -> httpx.Response:
        """Perform a GET request."""

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
    ) -> httpx.Response:
        """Perform a generic request."""

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
    ) -> AsyncContextManager[httpx.Response]:
        """Open a streaming response context."""
