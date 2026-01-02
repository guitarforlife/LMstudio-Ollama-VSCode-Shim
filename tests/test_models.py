"""Model and generate endpoint tests."""

# pylint: disable=duplicate-code

import pytest
from httpx import ASGITransport, AsyncClient

import backend
import backend_api
import main


@pytest.mark.asyncio
async def test_openai_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure /v1/models returns a normalized list."""

    async def fake_models(_client, _model_cache=None, **_kwargs):  # type: ignore[override]
        return [{"id": "model-a"}, {"name": "model-b"}]

    monkeypatch.setattr(backend_api.api, "models", fake_models)

    async with main.lifespan(main.app):
        transport = ASGITransport(app=main.app)
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

    async def fake_select(_client, _cache, model: str) -> str:  # type: ignore[override]
        return model

    async def fake_post(  # type: ignore[override]
        _client, _model_cache, _path: str, _payload: dict
    ) -> dict:
        return {"choices": [{"text": "hello"}]}

    monkeypatch.setattr(backend_api.api, "ensure_selected", fake_select)
    monkeypatch.setattr(backend_api.api, "post_openai_json", fake_post)

    async with main.lifespan(main.app):
        transport = ASGITransport(app=main.app)
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

    async def fake_resolve(_client, _cache, requested: str):  # type: ignore[override]
        calls["count"] += 1
        return f"{requested}:resolved", None

    selector = main.ModelSelector()
    monkeypatch.setattr(backend, "_resolve_model_id", fake_resolve)

    dummy_client = object()
    assert await selector.ensure_selected(dummy_client, None, "model-a") == "model-a:resolved"
    assert await selector.ensure_selected(dummy_client, None, "model-a") == "model-a:resolved"
    assert calls["count"] == 1
