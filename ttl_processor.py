"""TTL parsing and injection helpers for LM Studio requests."""

import re
from typing import Any, Optional

from pydantic_settings import BaseSettings

_DURATION_RE = re.compile(
    r"^\s*(?P<num>-?\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m|h|d)?\s*$",
    re.IGNORECASE,
)


class TTLProcessor:
    """Translate Ollama keep_alive values into LM Studio ttl seconds.

    ``inject_ttl`` returns a new payload and injects a ``ttl`` field when applicable.
    """

    def __init__(self, settings: BaseSettings) -> None:
        self._settings = settings

    def parse_keep_alive_to_ttl_seconds(self, keep_alive: Any) -> Optional[int]:
        """
        Map Ollama's keep_alive to LM Studio's ttl (idle TTL, seconds).
        Returns None when no TTL should be injected.

        Args:
            keep_alive: Ollama keep_alive value (number or duration string).

        Returns:
            TTL in seconds, or None if no TTL should be injected.
        """
        # pylint: disable=too-many-return-statements,too-many-branches
        if keep_alive is None:
            return None
        unload_ttl = int(getattr(self._settings, "unload_ttl_seconds") or 0)

        if isinstance(keep_alive, (int, float)):
            if keep_alive <= 0:
                return unload_ttl if unload_ttl > 0 else None
            return int(keep_alive)

        if isinstance(keep_alive, str):
            s = keep_alive.strip().lower()
            if s in {"", "null", "none"}:
                return None
            if s in {"-1", "infinite", "forever"}:
                return None
            if s in {"0", "0s", "0m", "0h"}:
                return unload_ttl if unload_ttl > 0 else None

            m = _DURATION_RE.match(s)
            if not m:
                return None

            num = float(m.group("num"))
            unit = (m.group("unit") or "s").lower()
            if num <= 0:
                return unload_ttl if unload_ttl > 0 else None

            if unit == "ms":
                return max(1, int(num / 1000.0))
            if unit == "s":
                return int(num)
            if unit == "m":
                return int(num * 60)
            if unit == "h":
                return int(num * 3600)
            if unit == "d":
                return int(num * 86400)
            return int(num)

        return None

    def inject_ttl(self, payload: dict, keep_alive: Any = None) -> dict:
        """Return a copy of payload with ttl injected when appropriate.

        Args:
            payload: Request payload to augment.
            keep_alive: Ollama keep_alive value to map.

        Returns:
            New payload dictionary with ttl injected when applicable.
        """
        ttl = self.parse_keep_alive_to_ttl_seconds(keep_alive)
        default_ttl = int(getattr(self._settings, "default_ttl_seconds") or 0)
        if ttl is None and default_ttl > 0:
            ttl = default_ttl

        if ttl is not None:
            updated = dict(payload)
            updated["ttl"] = int(ttl)
            return updated
        return dict(payload)
