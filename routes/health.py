"""Health and metrics endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from backend import BackendError, preflight_lmstudio
from client import BackendUnavailableError
from deps import get_client
from state import OLLAMA_VERSION
from utils.time import now

router = APIRouter()

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"


async def _preflight(client) -> None:
    await preflight_lmstudio(client)


def _backend_unavailable() -> JSONResponse:
    return JSONResponse(
        {"error": "backend_unavailable", "detail": "LMStudio backend unavailable"},
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
async def ready(client=Depends(get_client)) -> JSONResponse:
    """Return readiness status after verifying LM Studio is reachable."""
    try:
        await asyncio.wait_for(_preflight(client), timeout=3.0)
    except (BackendError, BackendUnavailableError, asyncio.TimeoutError):
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
async def healthz(client=Depends(get_client)) -> JSONResponse:
    """Return a combined shim/backend health payload."""
    backend_ok = True
    try:
        await asyncio.wait_for(_preflight(client), timeout=1.0)
    except (BackendError, BackendUnavailableError, asyncio.TimeoutError):
        backend_ok = False
    payload = {
        "shim": "ok",
        "backend": "ok" if backend_ok else "unavailable",
        "timestamp": now(),
    }
    status = 200 if backend_ok else 503
    return JSONResponse(payload, status_code=status)


@router.get("/metrics")
async def metrics(client=Depends(get_client)) -> Response:
    """Return Prometheus metrics when available."""
    if PROMETHEUS_AVAILABLE:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    up = 0
    try:
        await asyncio.wait_for(_preflight(client), timeout=1.0)
        up = 1
    except (BackendError, BackendUnavailableError, asyncio.TimeoutError):
        up = 0
    body = f"shim_up {up}\nshim_version{{version=\"{OLLAMA_VERSION}\"}} 1\n"
    return PlainTextResponse(body)
