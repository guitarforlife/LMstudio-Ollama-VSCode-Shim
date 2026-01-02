"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request


def get_client(request: Request):
    """Return the shared HTTP client."""
    return request.app.state.client


def get_model_cache(request: Request):
    """Return the shared model cache."""
    return getattr(request.app.state, "model_cache", None)
