"""Streaming helper tests."""

import pytest
import httpx

import main


class DummyStream:
    """Async context manager for a fixed response."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def __aenter__(self) -> httpx.Response:
        return self._response

    async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
        return False


class ResponseClient:  # pylint: disable=too-few-public-methods
    """Client that returns a predefined response."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def stream(self, _method: str, _url: str, **_kwargs):  # type: ignore[override]
        """Return a dummy stream for the provided response."""
        return DummyStream(self._response)


class ErrorClient:  # pylint: disable=too-few-public-methods
    """Client that raises a connection error on stream."""

    def stream(self, _method: str, url: str, **_kwargs):  # type: ignore[override]
        """Raise a connection error to simulate backend failure."""
        raise httpx.ConnectError("boom", request=httpx.Request("POST", url))


async def _collect(stream) -> bytes:
    """Collect all stream chunks into a single bytes payload."""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_stream_post_raw_http_status_error() -> None:
    """Return error payload when LM Studio responds with an error status."""
    url = "http://example.test/completions"
    response = httpx.Response(500, text="boom", request=httpx.Request("POST", url))
    client = ResponseClient(response)

    def on_error(message: str):
        return [f"ERR:{message}".encode()]

    result = await _collect(
        main.stream_post_raw(client, url, {}, log_label="test", on_error=on_error)
    )
    assert result == b"ERR:boom"


@pytest.mark.asyncio
async def test_stream_post_raw_request_error() -> None:
    """Return error payload when LM Studio is unreachable."""
    url = "http://example.test/completions"
    client = ErrorClient()

    def on_error(message: str):
        return [f"ERR:{message}".encode()]

    result = await _collect(
        main.stream_post_raw(client, url, {}, log_label="test", on_error=on_error)
    )
    assert result == b"ERR:LMStudio backend unavailable"
