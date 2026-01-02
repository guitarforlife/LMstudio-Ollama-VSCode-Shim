"""HTTP client factory and backend request helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from config import Settings
from logging_config import logger, request_id_ctx
from utils import json
from utils.retry import BackendUnavailableError, retry
from utils.types import BackendLike



@dataclass
class BackendError(Exception):
    """Backend error with status and payload details."""

    status_code: int
    error: str
    detail: str


@dataclass(frozen=True)
class RequestOptions:
    """Request options for backend HTTP calls."""

    payload: Optional[Dict[str, Any]] = None
    retries: int = 0
    backoff: float = 0.25


def default_client_factory(settings: Settings) -> httpx.AsyncClient:
    """Create the shared HTTP client for the app."""
    timeout = httpx.Timeout(settings.http_timeout) if settings.http_timeout is not None else None
    limits = httpx.Limits(
        max_keepalive_connections=settings.max_keepalive_connections or None,
        max_connections=settings.max_connections or None,
    )
    transport = httpx.AsyncHTTPTransport(retries=3, verify=settings.verify_ssl)
    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        verify=settings.verify_ssl,
        http2=True,
        transport=transport,
    )


async def request_json(
    client: BackendLike,
    method: str,
    url: str,
    *,
    options: Optional[RequestOptions] = None,
) -> Dict[str, Any]:
    """Perform a JSON request to the backend with unified error handling."""
    request_id = request_id_ctx.get("-")
    resolved = options or RequestOptions()
    try:
        response = await retry(
            lambda: client.request(method, url, json=resolved.payload),
            retries=resolved.retries,
            backoff=resolved.backoff,
        )
    except BackendUnavailableError:
        logger.error(
            "Backend request failed",
            extra={"url": url, "request_id_ctx": request_id},
            exc_info=True,
        )
        raise

    if response.is_error:
        logger.error(
            "Backend request error",
            extra={
                "url": url,
                "status": response.status_code,
                "body": response.text,
                "request_id_ctx": request_id,
            },
        )
        raise BackendError(
            status_code=response.status_code,
            error="backend_error",
            detail=response.text or "LMStudio error",
        )

    try:
        return response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "Backend returned invalid JSON",
            extra={"url": url, "response_body": response.text, "request_id_ctx": request_id},
            exc_info=True,
        )
        raise BackendError(
            status_code=502,
            error="invalid_json",
            detail="LMStudio returned invalid JSON",
        ) from exc
