"""HTTP request helpers for the shim."""

from __future__ import annotations

from typing import Any, Dict, Optional, Literal
import logging

import httpx
from fastapi import HTTPException

from state import settings
from utils import json
from utils.model_selection import raise_if_unloaded
from utils.retry import BackendUnavailableError, retry

logger = logging.getLogger(__name__)


def proxy_json(response: httpx.Response) -> Dict[str, Any]:
    """Return JSON payload or raise HTTPException on backend errors."""
    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text or "LMStudio error",
        )
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="LMStudio returned invalid JSON") from exc


async def proxy_request(
    client: httpx.AsyncClient,
    method: Literal["GET", "POST"],
    url: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    retries: int = 0,
) -> Dict[str, Any]:
    """Perform a backend request with shared retry/error handling."""
    try:
        if method == "GET":
            response = await retry(
                client.get,
                url,
                retries=retries,
                backoff=settings.request_retry_backoff,
            )
        else:
            response = await retry(
                lambda: client.post(url, json=json_body),
                retries=retries,
                backoff=settings.request_retry_backoff,
            )
    except BackendUnavailableError as exc:
        detail = "LMStudio backend unavailable"
        logger.error(detail, extra={"url": url}, exc_info=True)
        raise HTTPException(status_code=502, detail=detail) from exc

    if response.is_error and method == "POST" and json_body is not None:
        await raise_if_unloaded(client, None, json_body, response)
    return proxy_json(response)
