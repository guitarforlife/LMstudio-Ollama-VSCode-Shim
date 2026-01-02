"""JSON helpers using orjson when available."""

from __future__ import annotations

from typing import Any, Optional

try:
    import orjson as ORJSON
except ModuleNotFoundError:
    ORJSON = None

if ORJSON is not None:
    JSONDecodeError = ORJSON.JSONDecodeError

    def dumps(obj: Any, *, option: Optional[int] = None) -> str:
        """Serialize to JSON string with orjson."""
        opts = ORJSON.OPT_SERIALIZE_NUMPY if option is None else option
        return ORJSON.dumps(obj, option=opts).decode()

    def dumps_bytes(obj: Any, *, option: Optional[int] = None) -> bytes:
        """Serialize to JSON bytes with orjson."""
        opts = ORJSON.OPT_SERIALIZE_NUMPY if option is None else option
        return ORJSON.dumps(obj, option=opts)

    def loads(data: Any) -> Any:
        """Parse JSON from str/bytes with orjson."""
        return ORJSON.loads(data)
else:
    import json as JSON

    JSONDecodeError = JSON.JSONDecodeError

    def dumps(obj: Any, *, option: Optional[int] = None) -> str:
        """Serialize to JSON string with stdlib json."""
        _ = option
        return JSON.dumps(obj)

    def dumps_bytes(obj: Any, *, option: Optional[int] = None) -> bytes:
        """Serialize to JSON bytes with stdlib json."""
        _ = option
        return JSON.dumps(obj).encode()

    def loads(data: Any) -> Any:
        """Parse JSON from str/bytes with stdlib json."""
        return JSON.loads(data)


__all__ = ["JSONDecodeError", "dumps", "dumps_bytes", "loads"]
