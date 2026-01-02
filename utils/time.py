"""Time helpers for the shim."""

from __future__ import annotations

import datetime as _dt


def now() -> str:
    """Return current UTC time in ISO format."""
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
