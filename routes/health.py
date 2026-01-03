"""Health and metrics endpoints."""

from __future__ import annotations

import asyncio
from typing import Callable, Optional, TYPE_CHECKING

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from backend import BackendError
from backend_api import api as backend_api
from constants import ERROR_BACKEND_UNAVAILABLE
from deps import get_client
from state import OLLAMA_VERSION
from utils.retry import BackendUnavailableError
from utils.time import now

router = APIRouter()

if TYPE_CHECKING:
    pass

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest as _generate_latest

    PROMETHEUS_AVAILABLE = True
    GENERATE_LATEST: Optional[Callable[..., bytes]] = _generate_latest
except ImportError:  # pragma: no cover - optional dependency
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    GENERATE_LATEST = None


async def _backend_is_up(client: httpx.AsyncClient, timeout: float) -> bool:
    """Return True if the backend responds to a preflight request."""
    try:
        await asyncio.wait_for(backend_api.preflight(client), timeout=timeout)
        return True
    except (BackendError, BackendUnavailableError, asyncio.TimeoutError):
        return False


def _backend_unavailable() -> JSONResponse:
    return JSONResponse(
        {"error": ERROR_BACKEND_UNAVAILABLE, "detail": "LMStudio backend unavailable"},
        503,
    )


@router.get("/health")
async def health() -> JSONResponse:
    """Return liveness status and timestamp."""
    return JSONResponse({"status": "ok", "service": "shim", "timestamp": now()})


@router.get("/ping")
async def ping() -> JSONResponse:
    """Return a basic liveness response without backend checks."""
    return JSONResponse({"status": "ok"})


@router.get("/ready")
async def ready(client: httpx.AsyncClient = Depends(get_client)) -> JSONResponse:
    """Return readiness status after verifying LM Studio is reachable."""
    ok = await _backend_is_up(client, timeout=3.0)
    if not ok:
        return _backend_unavailable()
    return JSONResponse(
        {
            "status": "ok",
            "backend": "lmstudio",
            "version": OLLAMA_VERSION,
            "timestamp": now(),
        }
    )


@router.get("/healthz")
async def healthz(client: httpx.AsyncClient = Depends(get_client)) -> JSONResponse:
    """Return a combined shim/backend health payload."""
    backend_ok = await _backend_is_up(client, timeout=1.0)
    payload = {
        "shim": "ok",
        "backend": "ok" if backend_ok else "unavailable",
        "timestamp": now(),
    }
    status = 200 if backend_ok else 503
    return JSONResponse(payload, status_code=status)


@router.get("/metrics")
async def metrics(client: httpx.AsyncClient = Depends(get_client)) -> Response:
    """Return Prometheus metrics when available."""
    if PROMETHEUS_AVAILABLE and GENERATE_LATEST is not None:
        return Response(GENERATE_LATEST(), media_type=CONTENT_TYPE_LATEST)

    up = 1 if await _backend_is_up(client, timeout=1.0) else 0
    body = f"shim_up {up}\nshim_version{{version=\"{OLLAMA_VERSION}\"}} 1\n"
    return PlainTextResponse(body)
