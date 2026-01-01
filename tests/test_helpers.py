"""Helper unit tests for backend utilities and TTL handling."""

# pylint: disable=duplicate-code,protected-access

import pytest
import httpx
from pydantic_settings import BaseSettings

import main
from ttl_processor import TTLProcessor


class DummySettings(BaseSettings):  # pylint: disable=too-few-public-methods
    """Settings stub for TTL parsing tests."""

    unload_ttl_seconds: int = 2
    default_ttl_seconds: int = 0


@pytest.mark.asyncio
async def test_resolve_model_id_matches_tagged_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve tagged model names to known entries."""

    async def fake_models(_client):  # type: ignore[override]
        return [{"id": "foo:latest"}, {"name": "bar"}]

    monkeypatch.setattr(main, "lm_models", fake_models)
    model_id, entry = await main._resolve_model_id(object(), "foo")
    assert model_id == "foo:latest"
    assert entry == {"id": "foo:latest"}


def test_ttl_processor_parsing_and_copy() -> None:
    """Verify TTL parsing and non-mutating injection."""
    processor = TTLProcessor(DummySettings())
    assert processor.parse_keep_alive_to_ttl_seconds(-1) == 2
    assert processor.parse_keep_alive_to_ttl_seconds("forever") is None
    assert processor.parse_keep_alive_to_ttl_seconds("2m") == 120
    assert processor.parse_keep_alive_to_ttl_seconds("500ms") == 1

    payload = {"model": "foo"}
    updated = processor.inject_ttl(payload, keep_alive="10s")
    assert updated["ttl"] == 10
    assert payload is not updated
    assert "ttl" not in payload


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
        await main._proxy_get(FailingClient(), url, retries=1)
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
            _json: dict = None,
            **_kwargs,
        ):
            """Return a 404 response to simulate unloaded model."""
            return httpx.Response(
                404,
                text="not loaded",
                request=httpx.Request("POST", request_url),
            )

    async def fake_exists(_client, _model: str):  # type: ignore[override]
        return True, "not-loaded"

    monkeypatch.setattr(main, "_model_exists_and_state", fake_exists)

    with pytest.raises(main.HTTPException) as exc:
        await main._proxy_post_json(UnloadedClient(), url, {"model": "foo"}, retries=0)
    assert exc.value.status_code == 409
