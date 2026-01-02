"""Helper unit tests for backend utilities and TTL handling."""

# pylint: disable=unused-argument

# pylint: disable=duplicate-code,protected-access

from typing import Any, Optional, cast

import pytest
import httpx
from pydantic_settings import BaseSettings

import backend
import main
from utils import helpers
from utils.ttl import inject_ttl
from ttl_processor import TTLProcessor


class DummySettings(BaseSettings):  # pylint: disable=too-few-public-methods
    """Settings stub for TTL parsing tests."""

    unload_ttl_seconds: int = 2
    default_ttl_seconds: int = 0


@pytest.mark.asyncio
async def test_resolve_model_id_matches_tagged_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve tagged model names to known entries."""

    async def fake_models(_client, _model_cache=None, **_kwargs):  # type: ignore[override]
        return [{"id": "foo:latest"}, {"name": "bar"}]

    monkeypatch.setattr(backend, "lm_models", fake_models)
    model_id, entry = await backend._resolve_model_id(cast(Any, object()), None, "foo")
    assert model_id == "foo:latest"
    assert entry == {"id": "foo:latest"}


def test_ttl_processor_parsing_and_copy() -> None:
    """Verify TTL parsing and non-mutating injection."""
    processor = TTLProcessor(DummySettings())
    assert processor.parse_keep_alive_to_ttl_seconds(-1) == 2
    assert processor.parse_keep_alive_to_ttl_seconds("forever") is None
    assert processor.parse_keep_alive_to_ttl_seconds("2m") == 120
    assert processor.parse_keep_alive_to_ttl_seconds("500ms") == 1
    assert processor.parse_keep_alive_to_ttl_seconds("abc") is None

    payload = {"model": "foo"}
    updated = processor.inject_ttl(payload, keep_alive="10s")
    assert updated["ttl"] == 10
    assert payload is not updated
    assert "ttl" not in payload


def test_inject_ttl_if_missing() -> None:
    """Ensure inject_ttl_if_missing injects TTL without mutating input."""
    payload = {"model": "foo"}
    updated = helpers.inject_ttl_if_missing(payload, keep_alive="2m", settings=DummySettings())
    assert updated["ttl"] == 120
    assert "ttl" not in payload


def test_inject_ttl_malformed_keep_alive() -> None:
    """Ensure malformed keep_alive does not inject ttl."""
    payload = {"model": "foo"}
    updated = inject_ttl(payload, keep_alive="nope", settings=DummySettings())
    assert "ttl" not in updated


@pytest.mark.asyncio
async def test_proxy_get_retries_and_fails() -> None:
    """Ensure GET retries fail with HTTPException."""
    url = "http://example.test/models"

    class FailingClient:  # pylint: disable=too-few-public-methods
        """Fake client that always raises RequestError."""

        async def get(self, request_url: str):  # type: ignore[override]
            """Raise a request error for every GET."""
            raise httpx.RequestError("boom", request=httpx.Request("GET", request_url))

    with pytest.raises(main.HTTPException) as exc:
        await main._proxy_get(cast(Any, FailingClient()), url, retries=1)
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_proxy_post_unloaded_model_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return conflict when model exists but is unloaded."""
    url = "http://example.test/chat/completions"

    class UnloadedClient:  # pylint: disable=too-few-public-methods
        """Fake client that returns a 404 for unloaded models."""

        async def post(  # type: ignore[override]
            self,
            request_url: str,
            _json: Optional[dict] = None,
            **_kwargs,
        ):
            """Return a 404 response to simulate unloaded model."""
            return httpx.Response(
                404,
                text="not loaded",
                request=httpx.Request("POST", request_url),
            )

    async def fake_exists(_client, _cache, _model: str):  # type: ignore[override]
        return True, "not-loaded"

    monkeypatch.setattr(backend, "model_exists_and_state", fake_exists)

    with pytest.raises(main.HTTPException) as exc:
        await main._proxy_post_json(cast(Any, UnloadedClient()), url, {"model": "foo"}, retries=0)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_proxy_post_retries_and_fails() -> None:
    """Ensure POST retries fail with HTTPException."""
    url = "http://example.test/chat/completions"

    class FailingClient:  # pylint: disable=too-few-public-methods
        """Fake client that always raises RequestError."""

        async def post(self, request_url: str, **_kwargs):  # type: ignore[override]
            """Raise a request error for every POST."""
            raise httpx.RequestError("boom", request=httpx.Request("POST", request_url))

    with pytest.raises(main.HTTPException) as exc:
        await main._proxy_post_json(cast(Any, FailingClient()), url, {"model": "foo"}, retries=1)
    assert exc.value.status_code == 502
