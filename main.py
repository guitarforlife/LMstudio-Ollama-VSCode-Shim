"""FastAPI entry point for the LMStudio-Ollama shim.

This is the canonical app implementation. `shim/main.py` is a thin wrapper that
re-exports the same symbols for module-based execution and backward
compatibility.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import backend
from constants import ERROR_BACKEND_UNAVAILABLE
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
from utils.factory import coerce_client_factory
from utils.http import proxy_request
import state
from logging_config import logger
from state import LMSTUDIO_OPENAI_BASE, LMSTUDIO_REST_BASE, OLLAMA_VERSION, SHIM_VERSION

settings = state.settings


async def lm_models(client: httpx.AsyncClient) -> list[dict[str, Any]]:
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


async def _proxy_get(
    client: httpx.AsyncClient,
    url: str,
    retries: int = 0,
) -> Dict[str, Any]:
    return await proxy_request(client, "GET", url, retries=retries)


async def _proxy_post_json(
    client: httpx.AsyncClient, url: str, payload: Dict[str, Any], retries: int = 0
) -> Dict[str, Any]:
    return await proxy_request(client, "POST", url, json_body=payload, retries=retries)


@asynccontextmanager
async def lifespan(
    fastapi_app: FastAPI,
    client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown lifecycle for the FastAPI app."""
    client_factory_attr = getattr(fastapi_app.state, "client_factory", None)
    resolved_factory = coerce_client_factory(
        client_factory or client_factory_attr,
        default_factory=lambda: default_client_factory(state.settings),
        logger=logger,
    )
    try:
        client = await resolved_factory()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to create HTTP client", exc_info=True)
        raise RuntimeError("Failed to start shim") from exc

    fastapi_app.state.client = client
    fastapi_app.state.model_cache = ModelCache(state.settings.model_cache_ttl_seconds)
    logger.info(
        "Shim started",
        extra={"version": OLLAMA_VERSION},
    )
    logger.info(
        "LM Studio OpenAI base",
        extra={"base_url": LMSTUDIO_OPENAI_BASE},
    )
    logger.info(
        "LM Studio REST base",
        extra={"base_url": LMSTUDIO_REST_BASE},
    )
    try:
        yield
    finally:
        try:
            timeout = state.settings.http_timeout
            if timeout is None:
                timeout = 5.0
            await asyncio.wait_for(client.aclose(), timeout=timeout)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Error closing HTTP client")


app = FastAPI(
    title="Ollama to LM Studio Shim",
    version=SHIM_VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=state.settings.allowed_origins,
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
        {"error": ERROR_BACKEND_UNAVAILABLE, "detail": str(exc)},
        status_code=503,
    )


def run() -> None:
    """Start the Uvicorn server for the shim."""
    try:
        uvicorn.run(app, host=state.settings.host, port=state.settings.port)
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
