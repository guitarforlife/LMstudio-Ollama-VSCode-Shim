"""Request middleware for authentication, request IDs, and metrics."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from logging_config import request_id_ctx
from state import settings

logger = logging.getLogger("lmstudio_shim")

try:
    from prometheus_client import Counter, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None


request_count: Optional["Counter"] = None
request_latency: Optional["Histogram"] = None

if PROMETHEUS_AVAILABLE:
    request_count = Counter(
        "shim_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    request_latency = Histogram(
        "shim_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "path"],
    )


def _scrub_path(path: str) -> str:
    if not path:
        return path
    if path.startswith("/v1/"):
        return "/v1/*"
    if path.startswith("/api/"):
        return "/api/*"
    return path


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach a request ID header to each request/response pair."""
    request_id = request.headers.get("X-Request-Id") or str(uuid4())
    token = request_id_ctx.set(request_id)
    request.state.logger = logging.LoggerAdapter(logger, {"request_id": request_id})
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    response.headers["X-Request-Id"] = request_id
    return response


PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/healthz",
    "/metrics",
    "/ping",
    "/version",
    "/api/version",
}


async def api_key_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Enforce API key authentication on non-public endpoints."""
    if not settings.api_key or request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if not auth:
        return JSONResponse(
            {"error": "unauthorized", "detail": "missing Authorization header"},
            401,
        )
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    else:
        token = auth.strip()
    if token != settings.api_key:
        return JSONResponse({"error": "unauthorized", "detail": "invalid API key"}, 401)
    return await call_next(request)


async def request_size_limit_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject requests that exceed the configured max size."""
    if settings.max_request_bytes:
        size_header = request.headers.get("Content-Length")
        if size_header:
            try:
                size = int(size_header)
            except ValueError:
                return JSONResponse(
                    {"error": "invalid_request", "detail": "invalid Content-Length"},
                    400,
                )
            if size > settings.max_request_bytes:
                return JSONResponse(
                    {
                        "error": "payload_too_large",
                        "detail": f"payload exceeds {settings.max_request_bytes} bytes",
                    },
                    413,
                )
    return await call_next(request)


async def prometheus_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Collect request metrics when Prometheus is available."""
    if not PROMETHEUS_AVAILABLE or request_count is None or request_latency is None:
        return await call_next(request)

    method = request.method
    path = _scrub_path(request.url.path)
    with request_latency.labels(method=method, path=path).time():
        response = await call_next(request)
    request_count.labels(method=method, path=path, status=response.status_code).inc()
    return response
