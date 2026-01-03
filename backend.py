"""Backend helpers for LM Studio requests and model selection."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional, Tuple

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from client import BackendError, RequestOptions, request_json
from constants import ERROR_MODEL_NOT_LOADED
from logging_config import request_id_ctx
from state import LMSTUDIO_OPENAI_BASE, LMSTUDIO_REST_BASE, settings
from utils import json
from utils.retry import BackendUnavailableError
from utils.types import BackendLike
from utils.time import now

logger = logging.getLogger(__name__)



class ModelCache:
    """Simple in-memory cache for model listings."""

    def __init__(self, ttl_seconds: float) -> None:
        """Initialize the cache with a TTL in seconds."""
        self._ttl_seconds = ttl_seconds
        self._cached_at = 0.0
        self._cached_parsed: Optional[Tuple["ModelEntry", ...]] = None

    def get(self) -> Optional[Tuple["ModelEntry", ...]]:
        """Return cached models if fresh, otherwise None."""
        if not self._cached_parsed or self._ttl_seconds <= 0:
            return None
        if time.monotonic() - self._cached_at > self._ttl_seconds:
            return None
        return self._cached_parsed

    def set(self, payload: Iterable["ModelEntry"]) -> None:
        """Store model payload in the cache."""
        if self._ttl_seconds <= 0:
            return
        self._cached_parsed = tuple(payload)
        self._cached_at = time.monotonic()


class BackendClient:
    """Backend URL builder and accessor."""

    def __init__(self, client: httpx.AsyncClient, model_cache: Optional[ModelCache]) -> None:
        """Initialize with a shared client and optional model cache."""
        self.client = client
        self.model_cache = model_cache

    def openai_url(self, path: str) -> str:
        """Build the OpenAI-compatible URL for a given path."""
        return f"{LMSTUDIO_OPENAI_BASE}{path}"

    def rest_url(self, path: str) -> str:
        """Build the REST URL for a given path."""
        return f"{LMSTUDIO_REST_BASE}{path}"

    async def models(self) -> Tuple["ModelEntry", ...]:
        """Return models from the backend with caching."""
        return await lm_models(self.client, model_cache=self.model_cache)


def ollama_base_fields(model: str) -> Dict[str, Any]:
    """Return base fields for an Ollama response."""
    return {"model": model, "created_at": now()}


class ModelEntry(BaseModel):
    """Typed representation of an LM Studio model entry."""

    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: Optional[str] = None
    model: Optional[str] = None
    state: Optional[str] = None
    status: Optional[str] = None
    loaded: Optional[bool] = None
    is_loaded: Optional[bool] = None
    active: Optional[bool] = None
    owned_by: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    size: Optional[int] = None
    digest: Optional[str] = None


class ModelsPayload(BaseModel):
    """Typed response payload for model listings."""

    model_config = ConfigDict(extra="allow")
    data: Optional[List[ModelEntry]] = None
    models: Optional[List[ModelEntry]] = None


def _extract_model_id(entry: ModelEntry) -> str:
    """Extract a model identifier from a model entry."""
    for value in (entry.id, entry.name, entry.model):
        if value:
            return str(value)
    return ""


def _ollama_model_name(name: str) -> str:
    """Normalize Ollama model names by stripping tags (foo:latest -> foo)."""
    if not name:
        return name
    return str(name.split(":")[0])


def _entry_is_loaded(entry: ModelEntry) -> bool:
    """Return True if a model entry appears loaded/active."""
    state = entry.state or entry.status or ""
    return state.lower() in {"loaded", "active", "ready", "running"}


class ModelSelector:  # pylint: disable=too-few-public-methods
    """Keep track of the active model selection with concurrency control."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_model: Optional[str] = None
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_size = int(getattr(settings, "model_selector_cache_size", 0) or 0)

    def _cache_get(self, requested: str) -> Optional[str]:
        """Return cached model id and refresh LRU position."""
        cached = self._cache.get(requested)
        if cached is None:
            return None
        self._cache.move_to_end(requested)
        return cached

    def _cache_set(self, requested: str, model_id: str) -> None:
        """Store cached model id and enforce LRU size."""
        if self._cache_size <= 0:
            return
        self._cache[requested] = model_id
        self._cache.move_to_end(requested)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    async def ensure_selected(
        self, client: BackendLike, model_cache: Optional[ModelCache], requested: str
    ) -> str:
        """Resolve and store the active model id for LM Studio requests."""
        if not requested:
            return requested

        async with self._lock:
            cached = self._cache_get(requested)
            if cached is not None:
                model_id = cached
                entry = None
            else:
                model_id = None
                entry = None

        if model_id is None:
            resolved_id, entry = await _resolve_model_id(client, model_cache, requested)
        else:
            resolved_id = model_id

        async with self._lock:
            cached = self._cache_get(requested)
            if cached is None:
                self._cache_set(requested, resolved_id)
                model_id = resolved_id
            else:
                model_id = cached
                entry = None

            if self._active_model != model_id:
                logger.info(
                    "Model selection %s -> %s",
                    self._active_model,
                    model_id,
                    extra={"request_id_ctx": request_id_ctx.get("-")},
                )
            self._active_model = model_id

            if entry is not None and not _entry_is_loaded(entry):
                pass

            return model_id


model_selector = ModelSelector()


def ollama_done(model: str) -> str:
    """Return a streaming completion payload with done=true."""
    payload = {**ollama_base_fields(model), "done": True, "done_reason": "stop"}
    return json.dumps(payload) + "\n"


def ollama_error(model: str, message: str) -> str:
    """Return a streaming error payload with done=true."""
    payload = {**ollama_base_fields(model), "error": message, "done": True, "done_reason": "error"}
    return json.dumps(payload) + "\n"


def openai_stream_error(message: str) -> bytes:
    """Return an SSE error payload for OpenAI streams."""
    payload = {"error": {"message": message, "type": "server_error"}}
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _iter_response_bytes(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    """Yield response bytes with optional chunking."""
    chunk_size = int(getattr(settings, "stream_chunk_size", 0) or 0)
    iterator = response.aiter_bytes(chunk_size) if chunk_size > 0 else response.aiter_bytes()
    try:
        async for chunk in iterator:
            if chunk:
                yield chunk
    except httpx.StreamConsumed:
        if response.content:
            yield response.content


async def _handle_stream_httpx_error(
    exc: httpx.HTTPError,
    url: str,
    on_error: Callable[[str], Iterable[bytes]],
    request_id: str,
) -> AsyncGenerator[bytes, None]:
    """Emit error chunks for HTTPX streaming errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        logger.error(
            "LMStudio error",
            extra={
                "url": str(exc.request.url) if exc.request else url,
                "status": exc.response.status_code,
                "body": exc.response.text,
                "request_id_ctx": request_id,
            },
        )
        message = exc.response.text or "LMStudio error"
    elif isinstance(exc, httpx.RequestError):
        message = (
            "LMStudio request timed out"
            if isinstance(exc, httpx.ReadTimeout)
            else "LMStudio backend unavailable"
        )
        logger.error(
            "LMStudio request failed",
            extra={"url": url, "request_id_ctx": request_id},
            exc_info=True,
        )
    else:
        logger.error(
            "LMStudio HTTP error",
            extra={"url": url, "request_id_ctx": request_id},
            exc_info=True,
        )
        message = "LMStudio HTTP error"

    for chunk in on_error(message):
        yield chunk


async def _stream_post_raw(
    client: BackendLike,
    url: str,
    payload: Dict[str, Any],
    *,
    log_label: str,
    on_error: Callable[[str], Iterable[bytes]],
) -> AsyncGenerator[bytes, None]:
    """Stream a POST response body as raw bytes with unified error handling."""
    request_id = request_id_ctx.get("-")

    try:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for chunk in _iter_response_bytes(response):
                yield chunk
    except asyncio.CancelledError:
        logger.info(
            "%s stream cancelled",
            log_label,
            extra={"url": url, "request_id_ctx": request_id},
        )
        return
    except httpx.HTTPError as exc:
        async for chunk in _handle_stream_httpx_error(
            exc,
            url,
            on_error,
            request_id,
        ):
            yield chunk


async def preflight_lmstudio(client: BackendLike) -> None:
    """Check LM Studio is reachable before streaming."""
    urls = (f"{LMSTUDIO_REST_BASE}/models", f"{LMSTUDIO_OPENAI_BASE}/models")
    tasks = [client.get(url, timeout=5.0) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    request_error: Optional[BaseException] = None
    status_error: Optional[httpx.HTTPStatusError] = None

    for url, result in zip(urls, results):
        if isinstance(result, httpx.Response):
            if result.status_code == 404:
                continue
            try:
                result.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                status_error = exc
        elif isinstance(result, httpx.RequestError):
            request_error = result
        elif isinstance(result, Exception):
            request_error = result

    if request_error is not None:
        logger.error(
            "LMStudio connection error",
            extra={"request_id_ctx": request_id_ctx.get("-")},
            exc_info=True,
        )
        raise BackendUnavailableError("LMStudio backend unavailable") from request_error

    if status_error is not None:
        logger.error(
            "LMStudio preflight error",
            extra={
                "url": str(status_error.request.url) if status_error.request else "",
                "status": status_error.response.status_code,
                "body": status_error.response.text,
                "request_id_ctx": request_id_ctx.get("-"),
            },
        )
        raise BackendError(
            status_code=status_error.response.status_code,
            error="backend_error",
            detail=status_error.response.text or "LMStudio error",
        ) from status_error

    raise BackendUnavailableError("LMStudio backend unavailable")


def _parse_model_payload(payload: Any) -> List[ModelEntry]:
    """Parse model payload into typed entries."""
    try:
        if isinstance(payload, dict):
            parsed = ModelsPayload.model_validate(payload)
            if parsed.data is not None:
                return parsed.data
            if parsed.models is not None:
                return parsed.models
            return []
        if isinstance(payload, list):
            return [ModelEntry.model_validate(item) for item in payload]
    except ValidationError:
        return []
    return []


async def lm_models(
    client: BackendLike, model_cache: Optional[ModelCache] = None
) -> Tuple[ModelEntry, ...]:
    """Retrieve model list from LM Studio."""
    if model_cache:
        cached = model_cache.get()
        if cached is not None:
            return cached

    try:
        payload = await request_json(
            client,
            "GET",
            f"{LMSTUDIO_REST_BASE}/models",
            options=RequestOptions(
                retries=settings.request_retries,
                backoff=settings.request_retry_backoff,
            ),
        )
        models = _parse_model_payload(payload)
        if models:
            parsed = tuple(models)
            if model_cache:
                model_cache.set(parsed)
                return model_cache.get() or parsed
            return parsed
    except BackendError as exc:
        if exc.status_code != 404 and logger.isEnabledFor(logging.DEBUG):
            logger.debug("REST models failed: %s", exc.detail)

    payload = await request_json(
        client,
        "GET",
        f"{LMSTUDIO_OPENAI_BASE}/models",
        options=RequestOptions(
            retries=settings.request_retries,
            backoff=settings.request_retry_backoff,
        ),
    )
    models = _parse_model_payload(payload)
    parsed = tuple(models)
    if model_cache:
        model_cache.set(parsed)
        return model_cache.get() or parsed
    return parsed


async def _model_exists_and_state(
    client: BackendLike, model_cache: Optional[ModelCache], model: str
) -> Tuple[bool, Optional[str]]:
    """Return (exists, state) if the model exists in the REST model list."""
    base = _ollama_model_name(model)
    try:
        models = await lm_models(client, model_cache=model_cache)
    except BackendError:
        return False, None

    for entry in models:
        mid = _extract_model_id(entry)
        if not mid:
            continue
        if mid == model or mid == base or _ollama_model_name(mid) == base:
            state = entry.state or entry.status or ""
            return True, str(state)
    return False, None


async def _resolve_model_id(
    client: BackendLike, model_cache: Optional[ModelCache], requested: str
) -> Tuple[str, Optional[ModelEntry]]:
    """Resolve a requested model name to a known LM Studio model id if possible."""
    if not requested:
        return requested, None
    base = _ollama_model_name(requested)

    try:
        models = await lm_models(client, model_cache=model_cache)
    except BackendError:
        return requested, None

    fallback: Optional[Tuple[str, ModelEntry]] = None
    for entry in models:
        mid = _extract_model_id(entry)
        if not mid:
            continue
        if mid in {requested, base}:
            return str(mid), entry
        if fallback is None and _ollama_model_name(mid) == base:
            fallback = (str(mid), entry)

    if fallback is not None:
        return fallback

    logger.warning("Requested model %s not found; using as-is", requested)
    return str(requested), None


async def model_exists_and_state(
    client: httpx.AsyncClient, model_cache: Optional[ModelCache], model: str
) -> Tuple[bool, Optional[str]]:
    """Public wrapper for model existence checks."""
    return await _model_exists_and_state(client, model_cache, model)


async def resolve_model_id(
    client: httpx.AsyncClient, model_cache: Optional[ModelCache], requested: str
) -> Tuple[str, Optional[ModelEntry]]:
    """Public wrapper for resolving model identifiers."""
    return await _resolve_model_id(client, model_cache, requested)


def stream_post_raw(
    client: BackendLike,
    url: str,
    payload: Dict[str, Any],
    *,
    log_label: str,
    on_error: Callable[[str], Iterable[bytes]],
) -> AsyncGenerator[bytes, None]:
    """Public wrapper for streaming a raw POST response."""
    return _stream_post_raw(
        client,
        url,
        payload,
        log_label=log_label,
        on_error=on_error,
    )


async def post_openai_json(
    client: BackendLike,
    model_cache: Optional[ModelCache],
    path: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Send a POST request to the OpenAI compatibility endpoint."""
    try:
        return await request_json(
            client,
            "POST",
            f"{LMSTUDIO_OPENAI_BASE}{path}",
            options=RequestOptions(
                payload=payload,
                retries=settings.request_retries,
                backoff=settings.request_retry_backoff,
            ),
        )
    except BackendError as exc:
        if exc.status_code in (400, 404):
            model = str(payload.get("model") or "")
            if model:
                exists, state = await _model_exists_and_state(client, model_cache, model)
                if exists and state and state.lower() in {"not-loaded", "unloaded"}:
                    raise BackendError(
                        status_code=409,
                        error=ERROR_MODEL_NOT_LOADED,
                        detail=(
                            f"LM Studio reports model '{model}' is not loaded. "
                            "Enable 'Just in Time Model Loading' in LM Studio Server Settings "
                            "or manually load the model in LM Studio (or via `lms load`)."
                        ),
                    ) from exc
        raise


__all__ = [
    "BackendError",
    "BackendUnavailableError",
    "BackendClient",
    "ModelCache",
    "model_selector",
    "ollama_base_fields",
    "ollama_done",
    "ollama_error",
    "openai_stream_error",
    "post_openai_json",
    "preflight_lmstudio",
    "_extract_model_id",
    "_entry_is_loaded",
    "_ollama_model_name",
    "_model_exists_and_state",
    "_resolve_model_id",
    "_stream_post_raw",
    "lm_models",
    "model_exists_and_state",
    "resolve_model_id",
    "stream_post_raw",
]
