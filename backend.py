"""Backend helpers for LM Studio requests and model selection."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional, Tuple

import httpx

from client import BackendError, BackendUnavailableError, RequestOptions, request_json
from logging_config import request_id_ctx
from state import LMSTUDIO_OPENAI_BASE, LMSTUDIO_REST_BASE, settings
from utils.time import now

logger = logging.getLogger("lmstudio_shim")


class ModelCache:
    """Simple in-memory cache for model listings."""

    def __init__(self, ttl_seconds: float) -> None:
        """Initialize the cache with a TTL in seconds."""
        self._ttl_seconds = ttl_seconds
        self._cached_at = 0.0
        self._payload: Optional[List[Dict[str, Any]]] = None

    def get(self) -> Optional[List[Dict[str, Any]]]:
        """Return cached models if fresh, otherwise None."""
        if not self._payload or self._ttl_seconds <= 0:
            return None
        if time.monotonic() - self._cached_at > self._ttl_seconds:
            return None
        return list(self._payload)

    def set(self, payload: List[Dict[str, Any]]) -> None:
        """Store model payload in the cache."""
        if self._ttl_seconds <= 0:
            return
        self._payload = list(payload)
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

    async def models(self) -> List[Dict[str, Any]]:
        """Return models from the backend with caching."""
        return await lm_models(self.client, model_cache=self.model_cache)


def ollama_base_fields(model: str) -> Dict[str, Any]:
    """Return base fields for an Ollama response."""
    return {"model": model, "created_at": now()}


def _extract_model_id(entry: Dict[str, Any]) -> str:
    """Extract a model identifier from a model entry."""
    for key in ("id", "name", "model"):
        value = entry.get(key)
        if value:
            return str(value)
    return ""


def _ollama_model_name(name: str) -> str:
    """Normalize Ollama model names by stripping tags (foo:latest -> foo)."""
    if not name:
        return name
    return str(name.split(":")[0])


def _entry_is_loaded(entry: Dict[str, Any]) -> bool:
    """Return True if a model entry appears loaded/active."""
    for key in ("loaded", "is_loaded", "active"):
        if entry.get(key) is True:
            return True
    state = str(entry.get("state") or entry.get("status") or "").lower()
    return state in {"loaded", "active", "ready", "running"}


class ModelSelector:  # pylint: disable=too-few-public-methods
    """Keep track of the active model selection with concurrency control."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_model: Optional[str] = None
        self._cache: Dict[str, str] = {}

    async def ensure_selected(
        self, client: httpx.AsyncClient, model_cache: Optional[ModelCache], requested: str
    ) -> str:
        """Resolve and store the active model id for LM Studio requests."""
        if not requested:
            return requested

        async with self._lock:
            cached = self._cache.get(requested)
            if cached is not None:
                model_id = cached
                entry = None
            else:
                model_id, entry = await _resolve_model_id(client, model_cache, requested)
                self._cache[requested] = model_id
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


async def _stream_post_raw(
    client: httpx.AsyncClient,
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
            try:
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
            except httpx.StreamConsumed:
                if response.content:
                    yield response.content
    except asyncio.CancelledError:
        logger.info(
            "%s stream cancelled",
            log_label,
            extra={"url": url, "request_id_ctx": request_id},
        )
        return
    except httpx.HTTPStatusError as exc:
        logger.error(
            "LMStudio error",
            extra={
                "url": str(exc.request.url) if exc.request else url,
                "status": exc.response.status_code,
                "body": exc.response.text,
                "request_id_ctx": request_id,
            },
        )
        for chunk in on_error(exc.response.text or "LMStudio error"):
            yield chunk
    except httpx.RequestError as exc:
        if isinstance(exc, httpx.ReadTimeout):
            message = "LMStudio request timed out"
        else:
            message = "LMStudio backend unavailable"
        logger.error(
            "LMStudio request failed",
            extra={"url": url, "request_id_ctx": request_id},
            exc_info=True,
        )
        for chunk in on_error(message):
            yield chunk
    except httpx.HTTPError:
        logger.error(
            "LMStudio HTTP error",
            extra={"url": url, "request_id_ctx": request_id},
            exc_info=True,
        )
        for chunk in on_error("LMStudio HTTP error"):
            yield chunk


async def preflight_lmstudio(client: httpx.AsyncClient) -> None:
    """Check LM Studio is reachable before streaming."""
    for url in (f"{LMSTUDIO_REST_BASE}/models", f"{LMSTUDIO_OPENAI_BASE}/models"):
        try:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            return
        except httpx.RequestError as exc:
            logger.error(
                "LMStudio connection error",
                extra={"url": url, "request_id_ctx": request_id_ctx.get("-")},
                exc_info=True,
            )
            raise BackendUnavailableError("LMStudio backend unavailable") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "LMStudio preflight error",
                extra={
                    "url": str(exc.request.url) if exc.request else url,
                    "status": exc.response.status_code,
                    "body": exc.response.text,
                    "request_id_ctx": request_id_ctx.get("-"),
                },
            )
            raise BackendError(
                status_code=exc.response.status_code,
                error="backend_error",
                detail=exc.response.text or "LMStudio error",
            ) from exc
    raise BackendUnavailableError("LMStudio backend unavailable")


async def lm_models(
    client: httpx.AsyncClient, model_cache: Optional[ModelCache] = None
) -> List[Dict[str, Any]]:
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
            if model_cache:
                model_cache.set(models)
            return models
    except BackendError as exc:
        if exc.status_code != 404:
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
    if model_cache:
        model_cache.set(models)
    return models


def _parse_model_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
        models = payload.get("models")
        if isinstance(models, list):
            return models
    if isinstance(payload, list):
        return payload
    return []


async def _model_exists_and_state(
    client: httpx.AsyncClient, model_cache: Optional[ModelCache], model: str
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
            return True, str(entry.get("state") or entry.get("status") or "")
    return False, None


async def _resolve_model_id(
    client: httpx.AsyncClient, model_cache: Optional[ModelCache], requested: str
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Resolve a requested model name to a known LM Studio model id if possible."""
    if not requested:
        return requested, None
    base = _ollama_model_name(requested)

    try:
        models = await lm_models(client, model_cache=model_cache)
    except BackendError:
        return requested, None

    for entry in models:
        mid = _extract_model_id(entry)
        if not mid:
            continue
        if mid in {requested, base}:
            return str(mid), entry

    for entry in models:
        mid = _extract_model_id(entry)
        if not mid:
            continue
        if _ollama_model_name(mid) == base:
            return str(mid), entry

    logger.warning("Requested model %s not found; using as-is", requested)
    return str(requested), None


async def model_exists_and_state(
    client: httpx.AsyncClient, model_cache: Optional[ModelCache], model: str
) -> Tuple[bool, Optional[str]]:
    """Public wrapper for model existence checks."""
    return await _model_exists_and_state(client, model_cache, model)


async def resolve_model_id(
    client: httpx.AsyncClient, model_cache: Optional[ModelCache], requested: str
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Public wrapper for resolving model identifiers."""
    return await _resolve_model_id(client, model_cache, requested)


def stream_post_raw(
    client: httpx.AsyncClient,
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
    client: httpx.AsyncClient,
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
                        error="model_not_loaded",
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
