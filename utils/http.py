"""HTTP request helpers for the shim."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from utils.ttl import inject_ttl


def inject_ttl_if_missing(
    body: Dict[str, Any],
    keep_alive: Optional[Any],
    settings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Prepare request body with TTL injection."""
    return inject_ttl(body, keep_alive, settings=settings)


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
