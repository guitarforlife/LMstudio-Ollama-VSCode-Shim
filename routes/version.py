"""Version endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from state import OLLAMA_VERSION, SHIM_VERSION, logger

router = APIRouter()


@router.get("/api/version")
def api_version() -> JSONResponse:
    """Return the shim's Ollama-compatible version payload."""
    logger.info("api_version")
    headers = {"X-Ollama-Version": OLLAMA_VERSION, "Server": "Ollama"}
    return JSONResponse({"version": OLLAMA_VERSION}, headers=headers)


@router.get("/version")
def version() -> JSONResponse:
    """Return the shim's semantic version payload."""
    return JSONResponse({"version": SHIM_VERSION})
