"""Utility functions for the LMStudio-Ollama VSCode Shim."""

from typing import Any, AsyncGenerator, Dict, Iterable

import httpx

import backend
from utils import json
from utils.ttl import inject_ttl
from utils.time import now


async def stream_from_backend(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    response_model: str,
) -> AsyncGenerator[bytes, None]:
    """Stream responses from the LM Studio backend.

    Args:
        client: HTTP client for making requests.
        url: URL to stream from.
        payload: Request payload.
        response_model: Model name for responses.

    Yields:
        Chunks of data from the streaming response.

    Raises:
        httpx.HTTPStatusError: If the backend returns an error.
        httpx.RequestError: If there is a connection issue.
    """
    def on_error(message: str) -> Iterable[bytes]:
        error_payload = {
            "error": message,
            "model": response_model,
            "created_at": now(),
            "done": True,
            "done_reason": "error",
        }
        return [json.dumps_bytes(error_payload) + b"\n"]

    async for chunk in backend.stream_post_raw(
        client,
        url,
        payload,
        log_label="helpers_stream",
        on_error=on_error,
    ):
        yield chunk


__all__ = ["inject_ttl", "stream_from_backend"]
