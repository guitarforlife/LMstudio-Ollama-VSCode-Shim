"""TTL helpers for LM Studio payloads."""

from __future__ import annotations

from typing import Any, Dict, Optional

from state import settings as app_settings
from ttl_processor import TTLProcessor


def inject_ttl(
    payload: Dict[str, Any],
    keep_alive: Optional[Any],
    settings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a new payload with TTL injected when applicable."""
    if keep_alive is None and "ttl" in payload:
        return dict(payload)
    processor = TTLProcessor(settings or app_settings)
    return processor.inject_ttl(payload, keep_alive=keep_alive)
