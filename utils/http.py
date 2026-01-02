"""HTTP request helpers for the shim."""

from __future__ import annotations

from typing import Any, Dict, Optional

from state import settings as app_settings
from ttl_processor import TTLProcessor


class TTLService:
    """Service for injecting TTL into request payloads."""

    def __init__(self, settings_provider) -> None:
        self._settings_provider = settings_provider
        self._processor: Optional[TTLProcessor] = None
        self._settings: Optional[Any] = None

    def _get_processor(self, settings: Optional[Any] = None) -> TTLProcessor:
        resolved_settings = settings or self._settings_provider()
        if self._processor is None or self._settings is not resolved_settings:
            self._processor = TTLProcessor(resolved_settings)
            self._settings = resolved_settings
        return self._processor

    def inject_ttl(
        self,
        payload: Dict[str, Any],
        keep_alive: Optional[Any],
        settings: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return payload with TTL injected when applicable."""
        if keep_alive is None and "ttl" in payload:
            # Respect an explicit ttl provided by the client payload.
            return dict(payload)
        processor = self._get_processor(settings)
        return processor.inject_ttl(payload, keep_alive=keep_alive)

    def reset(self) -> None:
        """Clear cached processor state (useful for tests)."""
        self._processor = None
        self._settings = None


_ttl_service = TTLService(lambda: app_settings)


def inject_ttl_if_missing(
    body: Dict[str, Any],
    keep_alive: Optional[Any],
    settings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Prepare request body with TTL injection."""
    return _ttl_service.inject_ttl(body, keep_alive, settings=settings)
