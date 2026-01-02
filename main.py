"""FastAPI entry point for the LMStudio-Ollama shim.

This is the canonical app implementation. `shim/main.py` is a thin wrapper that
re-exports the same symbols for module-based execution and backward
compatibility.
"""

from __future__ import annotations

import asyncio
import importlib
import multiprocessing
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Optional, cast

import httpx
import uvicorn
from fastapi import FastAPI, Request
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
    suppress_shutdown_cancel,
)
from routes import health_router, ollama_router, openai_router, version_router
from utils.factory import coerce_client_factory
from utils.http import proxy_request
import state
from logging_config import logger
from state import LMSTUDIO_OPENAI_BASE, LMSTUDIO_REST_BASE, OLLAMA_VERSION, SHIM_VERSION

try:
    UVLOOP = importlib.import_module("uvloop")
except ModuleNotFoundError:
    UVLOOP = None

_UVICORN_SUBPROCESS_ORIGINAL: Optional[Callable[..., None]] = None


def _suppress_spawned_process_interrupts() -> None:
    """Suppress noisy shutdown tracebacks in spawned worker processes."""
    if multiprocessing.current_process().name == "MainProcess":
        return

    def _hook(exc_type: type[BaseException], exc: BaseException, tb) -> None:
        if exc_type in (KeyboardInterrupt, asyncio.CancelledError):
            return
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


_suppress_spawned_process_interrupts()


def _wrapped_subprocess_started(*args: Any, **kwargs: Any) -> None:
    """Invoke uvicorn subprocess entrypoint with shutdown suppression."""
    try:
        original = _UVICORN_SUBPROCESS_ORIGINAL
        if original is None:
            return
        original(*args, **kwargs)
    except (KeyboardInterrupt, asyncio.CancelledError):
        return


def _patch_uvicorn_subprocess() -> None:
    """Wrap uvicorn subprocess entrypoint to suppress shutdown tracebacks."""
    global _UVICORN_SUBPROCESS_ORIGINAL  # pylint: disable=global-statement
    try:
        uvicorn_subprocess = importlib.import_module("uvicorn._subprocess")
    except ModuleNotFoundError:
        return

    original = getattr(uvicorn_subprocess, "subprocess_started", None)
    if original is None or getattr(original, "_shim_wrapped", False):
        return

    _UVICORN_SUBPROCESS_ORIGINAL = original
    setattr(_wrapped_subprocess_started, "_shim_wrapped", True)
    setattr(uvicorn_subprocess, "subprocess_started", _wrapped_subprocess_started)


_patch_uvicorn_subprocess()


def install_uvloop() -> bool:
    """Install uvloop if available for faster event loops."""
    if UVLOOP is not None and not os.getenv("DISABLE_UVLOOP"):
        try:
            UVLOOP.install()
            logger.info("uvloop enabled")
            return True
        except (RuntimeError, ValueError):
            return False
    try:
        asyncio.get_event_loop_policy().set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        pass
    return False

async def lm_models(client: httpx.AsyncClient) -> tuple[backend.ModelEntry, ...]:
    """Backward-compatible model list helper for tests."""
    return await backend.lm_models(client, model_cache=None)


async def _resolve_model_id(
    client: httpx.AsyncClient,
    requested: str,
) -> tuple[str, Optional[backend.ModelEntry]]:
    """Resolve a model identifier via the shim's public wrapper."""
    return await backend.resolve_model_id(client, None, requested)


async def _model_exists_and_state(
    client: httpx.AsyncClient,
    model: str,
) -> tuple[bool, Optional[str]]:
    """Check existence and state of a model."""
    return await backend.model_exists_and_state(client, None, model)


async def _proxy_get(
    client: httpx.AsyncClient,
    url: str,
    retries: int = 0,
) -> dict[str, Any]:
    """GET a JSON payload from the backend with retry handling."""
    return await proxy_request(client, "GET", url, retries=retries)


async def _proxy_post_json(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], retries: int = 0
) -> dict[str, Any]:
    """POST a JSON payload to the backend with retry handling."""
    return await proxy_request(client, "POST", url, json_body=payload, retries=retries)


async def _init_client(
    fastapi_app: FastAPI,
    client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
) -> httpx.AsyncClient:
    """Create the shared HTTP client using the configured factory."""
    client_factory_attr = getattr(fastapi_app.state, "client_factory", None)
    resolved_factory = coerce_client_factory(
        client_factory or client_factory_attr,
        default_factory=lambda: default_client_factory(state.settings),
        logger=logger,
    )
    return await resolved_factory()


async def _close_client(client: httpx.AsyncClient) -> None:
    """Gracefully close the client respecting the configured timeout."""
    try:
        timeout = state.settings.http_timeout or 5.0
        await asyncio.wait_for(client.aclose(), timeout=timeout)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Error closing HTTP client")


@asynccontextmanager
async def lifespan(
    fastapi_app: FastAPI,
    client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown lifecycle for the FastAPI app."""
    try:
        client = await _init_client(fastapi_app, client_factory=client_factory)
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
        "LM Studio OpenAI base: %s",
        LMSTUDIO_OPENAI_BASE,
        extra={"base_url": LMSTUDIO_OPENAI_BASE},
    )
    logger.info(
        "LM Studio REST base: %s",
        LMSTUDIO_REST_BASE,
        extra={"base_url": LMSTUDIO_REST_BASE},
    )
    try:
        yield
    finally:
        state.SHUTDOWN_IN_PROGRESS = True
        shutdown_event = getattr(fastapi_app.state, "shutdown_event", None)
        if shutdown_event is not None:
            shutdown_event.set()
        await _close_client(client)


async def backend_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return a consistent backend error response.

    The FastAPI ``add_exception_handler`` expects a handler that accepts a generic
    ``Exception``. We therefore accept ``exc`` as ``Exception`` and, if it is a
    ``BackendError``, format the response accordingly. For any other exception type,
    a generic error response is returned.
    """
    if isinstance(exc, BackendError):
        return JSONResponse(
            {"error": exc.error, "detail": exc.detail},
            status_code=exc.status_code,
        )
    # Fallback for unexpected exception types
    return JSONResponse({"error": "unknown_error", "detail": str(exc)}, status_code=500)


async def backend_unavailable_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return a consistent backend unavailable response.

    ``exc`` is accepted as a generic ``Exception`` to satisfy FastAPI's type
    expectations. If it is a ``BackendUnavailableError``, we format the response;
    otherwise, a generic error response is returned.
    """
    if isinstance(exc, BackendUnavailableError):
        return JSONResponse(
            {"error": ERROR_BACKEND_UNAVAILABLE, "detail": str(exc)},
            status_code=503,
        )
    return JSONResponse({"error": "unknown_error", "detail": str(exc)}, status_code=500)



def create_app() -> FastAPI:
    """Factory that builds the FastAPI instance with all routers & middleware."""
    fastapi_app = FastAPI(
        title="Ollama to LM Studio Shim",
        version=SHIM_VERSION,
        lifespan=lifespan,
    )
    fastapi_app.state.shutdown_event = asyncio.Event()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=state.settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    fastapi_app.middleware("http")(suppress_shutdown_cancel)
    fastapi_app.middleware("http")(request_size_limit_middleware)
    fastapi_app.middleware("http")(request_id_middleware)
    fastapi_app.middleware("http")(api_key_middleware)
    fastapi_app.middleware("http")(prometheus_middleware)

    fastapi_app.include_router(version_router)
    fastapi_app.include_router(health_router)
    fastapi_app.include_router(openai_router)
    fastapi_app.include_router(ollama_router)

    fastapi_app.add_exception_handler(BackendError, backend_error_handler)
    fastapi_app.add_exception_handler(BackendUnavailableError, backend_unavailable_handler)

    return fastapi_app


app = create_app()


def run() -> None:
    """Start the Uvicorn server for the shim."""
    try:
        use_uvloop = install_uvloop()
        workers = state.settings.workers
        if workers is None:
            workers = 1
        app_target: Any = app
        if workers and workers > 1:
            app_target = "main:app"
        uvicorn.run(
            app_target,
            host=state.settings.host,
            port=state.settings.port,
            timeout_graceful_shutdown=cast(int | None, 0.5),
            workers=workers,
            loop="uvloop" if use_uvloop else "asyncio",
        )
    except KeyboardInterrupt:
        pass


def run_server() -> None:
    """Backward-compatible entry point."""
    run()


__all__ = ["app", "get_client"]

# Re-exported for tests and external tooling.
ModelSelector = backend.ModelSelector
model_selector = backend.model_selector
stream_post_raw = backend.stream_post_raw
__all__ += ["ModelSelector", "model_selector", "stream_post_raw"]


if __name__ == "__main__":
    run()
