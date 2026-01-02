"""Tests for error handling paths."""

from typing import Any, cast

import httpx
import pytest
from fastapi import HTTPException
from pydantic_settings import BaseSettings

from utils.http import proxy_request
from utils.ttl import inject_ttl


class DummySettings(BaseSettings):  # pylint: disable=too-few-public-methods
    """Settings stub for TTL tests."""

    unload_ttl_seconds: int = 2
    default_ttl_seconds: int = 0


@pytest.mark.asyncio
async def test_proxy_request_post_error() -> None:
    """Ensure POST retries fail with HTTPException on backend error."""

    class FailingClient:  # pylint: disable=too-few-public-methods
        """Fake client that always raises RequestError."""

        async def post(  # type: ignore[override]
            self,
            request_url: str,
            **_kwargs: Any,
        ) -> httpx.Response:
            """Raise a request error for every POST."""
            raise httpx.RequestError("boom", request=httpx.Request("POST", request_url))

    with pytest.raises(HTTPException) as exc:
        await proxy_request(
            cast(Any, FailingClient()),
            "POST",
            "http://bad",
            json_body={},
            retries=1,
        )
    assert exc.value.status_code == 502


def test_inject_ttl_malformed_keep_alive() -> None:
    """Ensure malformed keep_alive does not inject ttl."""
    payload = {"model": "foo"}
    updated = inject_ttl(payload, keep_alive="nope", settings=DummySettings())
    assert "ttl" not in updated
    assert "ttl" not in payload
