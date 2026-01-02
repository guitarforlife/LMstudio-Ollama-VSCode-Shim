"""HTTP request helpers for the shim."""

from __future__ import annotations

from typing import Any, Dict, Optional

from state import settings as app_settings
from ttl_processor import TTLProcessor

_CACHE = {"processor": None, "settings": None}


def _get_ttl_processor(settings: Optional[Any] = None) -> TTLProcessor:
    """Return the cached TTL processor, initializing it if needed."""
    resolved_settings = settings or app_settings
    if _CACHE["processor"] is None or _CACHE["settings"] is not resolved_settings:
        _CACHE["processor"] = TTLProcessor(resolved_settings)
        _CACHE["settings"] = resolved_settings
    return _CACHE["processor"]


def prepare_body(
    body: Dict[str, Any],
    keep_alive: Optional[Any],
    settings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Prepare request body with TTL injection."""
    if keep_alive is None and "ttl" in body:
        return dict(body)

    ttl_processor = _get_ttl_processor(settings)
    return ttl_processor.inject_ttl(body, keep_alive=keep_alive)
