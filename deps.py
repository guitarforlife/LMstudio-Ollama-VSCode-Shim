"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Optional, cast

import httpx
from fastapi import Request

from backend import ModelCache


def get_client(request: Request) -> httpx.AsyncClient:
    """Return the shared HTTP client."""
    return cast(httpx.AsyncClient, request.app.state.client)


def get_model_cache(request: Request) -> Optional[ModelCache]:
    """Return the shared model cache."""
    return getattr(request.app.state, "model_cache", None)
