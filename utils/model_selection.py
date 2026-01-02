"""Model selection helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

import backend
import backend_api
from backend import ModelCache
from utils.ttl import inject_ttl


async def prepare_payload(
    client: httpx.AsyncClient,
    model_cache: Optional[ModelCache],
    model: str,
    payload: Dict[str, Any],
    keep_alive: Optional[Any],
) -> Dict[str, Any]:
    """Resolve model and inject TTL into a new payload."""
    resolved = await backend_api.api.ensure_selected(client, model_cache, model)
    updated = dict(payload)
    updated["model"] = resolved
    return inject_ttl(updated, keep_alive)


async def raise_if_unloaded(
    client: httpx.AsyncClient,
    model_cache: Optional[ModelCache],
    payload: Dict[str, Any],
    response: httpx.Response,
) -> None:
    """Raise a 409 if the backend indicates the model is not loaded."""
    if response.status_code not in (400, 404):
        return
    model = str(payload.get("model") or "")
    if not model:
        return
    exists, state = await backend.model_exists_and_state(client, model_cache, model)
    if exists and state and state.lower() in {"not-loaded", "unloaded"}:
        raise HTTPException(
            status_code=409,
            detail=(
                f"LM Studio reports model '{model}' is not loaded. "
                "Enable 'Just in Time Model Loading' in LM Studio Server Settings "
                "or manually load the model in LM Studio (or via `lms load`)."
            ),
        )
