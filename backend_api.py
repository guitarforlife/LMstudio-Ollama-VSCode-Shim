"""Thin abstraction layer for backend calls."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

from backend import (
    ModelCache,
    lm_models,
    model_selector,
    post_openai_json,
    preflight_lmstudio,
    stream_post_raw,
)
from utils.types import BackendLike


class BackendAPI:
    """Facade for backend interactions to simplify testing."""

    async def models(
        self,
        client: BackendLike,
        model_cache: Optional[ModelCache],
    ) -> list[Dict[str, Any]]:
        """Fetch model metadata from the backend."""
        return await lm_models(client, model_cache=model_cache)

    async def ensure_selected(
        self,
        client: BackendLike,
        model_cache: Optional[ModelCache],
        requested: str,
    ) -> str:
        """Resolve and cache a selected model name."""
        return await model_selector.ensure_selected(client, model_cache, requested)

    async def post_openai_json(
        self,
        client: BackendLike,
        model_cache: Optional[ModelCache],
        path: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send a JSON payload to the OpenAI-compatible backend path."""
        return await post_openai_json(client, model_cache, path, payload)

    async def preflight(self, client: BackendLike) -> None:
        """Check backend availability before streaming."""
        await preflight_lmstudio(client)

    def stream_post_raw(  # pylint: disable=too-many-arguments
        self,
        client: BackendLike,
        url: str,
        payload: Dict[str, Any],
        *,
        log_label: str,
        on_error: Callable[[str], Iterable[bytes]],
    ):
        """Stream a raw POST response from the backend."""
        options = {"log_label": log_label, "on_error": on_error}
        return stream_post_raw(client, url, payload, **options)


api = BackendAPI()
