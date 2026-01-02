"""shim - Ollama to LM Studio bridge.

Primary implementation lives here; `shim/main.py` re-exports for module usage.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, NoReturn, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import backend
from backend import BackendError, BackendUnavailableError, ModelCache
from client import default_client_factory
from deps import get_client
from middleware import (
    api_key_middleware,
    prometheus_middleware,
    request_id_middleware,
    request_size_limit_middleware,
)
from routes import health_router, ollama_router, openai_router, version_router
from logging_config import request_id_ctx
from state import (
    LMSTUDIO_OPENAI_BASE,
    LMSTUDIO_REST_BASE,
    OLLAMA_VERSION,
    SHIM_VERSION,
    logger,
    settings,
)

VERSION = SHIM_VERSION


async def lm_models(client: httpx.AsyncClient):
    """Backward-compatible model list helper for tests."""
    return await backend.lm_models(client, model_cache=None)


async def _resolve_model_id(
    client: httpx.AsyncClient,
    requested: str,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Backward-compatible model resolution helper for tests."""
    return await backend.resolve_model_id(client, None, requested)


async def _model_exists_and_state(
    client: httpx.AsyncClient,
    model: str,
) -> Tuple[bool, Optional[str]]:
    """Backward-compatible model lookup helper for tests."""
    return await backend.model_exists_and_state(client, None, model)


async def _retry_request(
    fn: Callable[..., Awaitable[httpx.Response]],
    *args: Any,
    retries: int = 0,
) -> httpx.Response:
    for attempt in range(retries + 1):
        try:
            return await fn(*args)
        except httpx.RequestError as exc:
            if attempt >= retries:
                raise BackendUnavailableError("LMStudio backend unavailable") from exc
            await asyncio.sleep(0.25)
    raise BackendUnavailableError("LMStudio backend unavailable")


def _handle_backend_error(exc: Exception) -> NoReturn:
    raise HTTPException(status_code=502, detail="LMStudio backend unavailable") from exc


async def _proxy_get(
    client: httpx.AsyncClient,
    url: str,
    retries: int = 0,
) -> Dict[str, Any]:
    try:
        response = await _retry_request(client.get, url, retries=retries)
    except BackendUnavailableError as exc:
        _handle_backend_error(exc)

    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text or "LMStudio error",
        )
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="LMStudio returned invalid JSON") from exc


async def _proxy_post_json(
    client: httpx.AsyncClient, url: str, payload: Dict[str, Any], retries: int = 0
) -> Dict[str, Any]:
    try:
        response = await _retry_request(lambda: client.post(url, json=payload), retries=retries)
    except BackendUnavailableError as exc:
        _handle_backend_error(exc)

    if response.is_error:
        model = ""
        if response.status_code in (400, 404):
            model = str(payload.get("model") or "")
        if model:
            exists, state = await _model_exists_and_state(client, model)
            if exists and state and state.lower() in {"not-loaded", "unloaded"}:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"LM Studio reports model '{model}' is not loaded. "
                        "Enable 'Just in Time Model Loading' in LM Studio Server Settings "
                        "or manually load the model in LM Studio (or via `lms load`)."
                    ),
                )
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text or "LMStudio error",
        )

    try:
        return response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="LMStudio returned invalid JSON") from exc


def _coerce_client_factory(factory: Any) -> Callable[[], httpx.AsyncClient]:
    """Validate or wrap a client factory.

    The argument may be a callable returning an ``httpx.AsyncClient`` or ``None``.
    """
    if not callable(factory):
        def default_factory() -> httpx.AsyncClient:
            return default_client_factory(settings)

        return default_factory

    def wrapped_factory() -> httpx.AsyncClient:
        client = factory()
        if not isinstance(client, httpx.AsyncClient):
            logger.error(
                "client_factory returned unexpected type",
                extra={"request_id": request_id_ctx.get("-")},
            )
            raise TypeError("client_factory must return httpx.AsyncClient")
        return client

    return wrapped_factory


@asynccontextmanager
async def lifespan(
    fastapi_app: FastAPI,
    client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown lifecycle for the FastAPI app."""
    client_factory_attr = getattr(fastapi_app.state, "client_factory", None)
    client_factory = _coerce_client_factory(client_factory or client_factory_attr)
    try:
        client = client_factory()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to create HTTP client", exc_info=True)
        raise RuntimeError("Failed to start shim") from exc

    fastapi_app.state.client = client
    fastapi_app.state.model_cache = ModelCache(settings.model_cache_ttl_seconds)
    logger.info("Shim started - Ollama version %s", OLLAMA_VERSION)
    logger.info("LM Studio OpenAI base: %s", LMSTUDIO_OPENAI_BASE)
    logger.info("LM Studio REST base:   %s", LMSTUDIO_REST_BASE)
    try:
        yield
    finally:
        try:
            await client.aclose()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Error closing HTTP client")


app = FastAPI(
    title="Ollama to LM Studio Shim",
    version=SHIM_VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.middleware("http")(request_size_limit_middleware)
app.middleware("http")(request_id_middleware)
app.middleware("http")(api_key_middleware)
app.middleware("http")(prometheus_middleware)

app.include_router(version_router)
app.include_router(health_router)
app.include_router(openai_router)
app.include_router(ollama_router)


@app.exception_handler(BackendError)
async def backend_error_handler(_request, exc: BackendError) -> JSONResponse:
    """Return a consistent backend error response."""
    return JSONResponse(
        {"error": exc.error, "detail": exc.detail},
        status_code=exc.status_code,
    )


@app.exception_handler(BackendUnavailableError)
async def backend_unavailable_handler(_request, exc: BackendUnavailableError) -> JSONResponse:
    """Return a consistent backend unavailable response."""
    return JSONResponse(
        {"error": "backend_unavailable", "detail": str(exc)},
        status_code=503,
    )


def run() -> None:
    """Start the Uvicorn server for the shim."""
    try:
        uvicorn.run(app, host=settings.host, port=settings.port)
    except KeyboardInterrupt:
        pass


def run_server() -> None:
    """Backward-compatible entry point."""
    run()


__all__ = ["app", "settings", "get_client"]

# Re-exported for tests and external tooling.
ModelSelector = backend.ModelSelector
model_selector = backend.model_selector
stream_post_raw = backend.stream_post_raw
__all__ += ["ModelSelector", "model_selector", "stream_post_raw"]


if __name__ == "__main__":
    run()
