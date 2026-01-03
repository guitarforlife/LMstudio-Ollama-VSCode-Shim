"""JSON helpers using orjson when available."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Callable, Optional, cast

_orjson_module: Optional[ModuleType] = None
_orjson_dumps: Optional[Callable[..., Any]] = None
_orjson_loads: Optional[Callable[..., Any]] = None
JSONDecodeError: type[Exception] = ValueError
_opt_serialize_numpy = 0

try:
    _orjson_module = importlib.import_module("orjson")
except ModuleNotFoundError:
    _orjson_module = None
else:
    try:
        _orjson_dumps = getattr(_orjson_module, "dumps")
        _orjson_loads = getattr(_orjson_module, "loads")
        JSONDecodeError = getattr(_orjson_module, "JSONDecodeError", ValueError)
        _opt_serialize_numpy = getattr(_orjson_module, "OPT_SERIALIZE_NUMPY", 0)
    except AttributeError:
        _orjson_module = None
    else:
        if not callable(_orjson_dumps) or not callable(_orjson_loads):
            _orjson_module = None

# --- orjson branch ---------------------------------------------------------
if _orjson_module is not None:
    if not callable(_orjson_dumps) or not callable(_orjson_loads):
        raise ValueError("orjson.dumps/loads unavailable")
    _orjson_dumps_fn = cast(Callable[..., bytes], _orjson_dumps)
    _orjson_loads_fn = cast(Callable[..., Any], _orjson_loads)

    def dumps(obj: Any, *, option: Optional[int] = None) -> str:
        """Serialize to JSON string with orjson."""
        opts = _opt_serialize_numpy if option is None else option
        return _orjson_dumps_fn(obj, option=opts).decode()

    def dumps_bytes(obj: Any, *, option: Optional[int] = None) -> bytes:
        """Serialize to JSON bytes with orjson."""
        opts = _opt_serialize_numpy if option is None else option
        return _orjson_dumps_fn(obj, option=opts)

    def loads(data: Any) -> Any:
        """Parse JSON from str/bytes with orjson."""
        return _orjson_loads_fn(data)

# --- stdlib json fallback --------------------------------------------------
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
