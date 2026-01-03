"""Model and generate endpoint tests."""

# pylint: disable=unused-argument

# pylint: disable=duplicate-code

from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

import backend
import backend_api
import main


@pytest.mark.asyncio
async def test_openai_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure /v1/models returns a normalized list."""

    async def fake_models(
        _client: Any,
        _model_cache: Any | None = None,
        **_kwargs: Any,
    ) -> list[backend.ModelEntry]:
        return [backend.ModelEntry(id="model-a"), backend.ModelEntry(name="model-b")]

    monkeypatch.setattr(backend_api.api, "models", fake_models)

    async with main.lifespan(main.app):
        transport = ASGITransport(app=cast(Any, main.app))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")
    assert resp.status_code == 200
    payload = resp.json()
    ids = [entry["id"] for entry in payload["data"]]
    assert "model-a" in ids
    assert "model-b" in ids


@pytest.mark.asyncio
async def test_generate_non_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure non-streaming generate maps to LM Studio."""

    async def fake_select(_client: Any, _cache: Any, model: str) -> str:
        return model

    async def fake_post(
        _client: Any,
        _model_cache: Any,
        _path: str,
        _payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {"choices": [{"text": "hello"}]}

    monkeypatch.setattr(backend_api.api, "ensure_selected", fake_select)
    monkeypatch.setattr(backend_api.api, "post_openai_json", fake_post)

    async with main.lifespan(main.app):
        transport = ASGITransport(app=cast(Any, main.app))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/generate",
                json={"model": "model-a", "prompt": "hi", "stream": False},
            )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["model"] == "model-a"
    assert payload["response"] == "hello"


@pytest.mark.asyncio
async def test_model_selector_caches_resolved_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure model selector caches resolved model ids."""
    calls = {"count": 0}

    async def fake_resolve(_client: Any, _cache: Any, requested: str) -> tuple[str, None]:
        calls["count"] += 1
        return f"{requested}:resolved", None

    selector = main.ModelSelector()
    monkeypatch.setattr(backend, "_resolve_model_id", fake_resolve)

    dummy_client = cast(Any, object())
    assert await selector.ensure_selected(dummy_client, None, "model-a") == "model-a:resolved"
    assert await selector.ensure_selected(dummy_client, None, "model-a") == "model-a:resolved"
    assert calls["count"] == 1
