"""Routes module for the LM Studio shim."""

from routes.health import router as health_router
from routes.ollama import router as ollama_router
from routes.openai import router as openai_router
from routes.version import router as version_router

__all__ = ["health_router", "ollama_router", "openai_router", "version_router"]
