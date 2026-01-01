"""Model and generate endpoint tests."""

# pylint: disable=duplicate-code

import pytest
from httpx import ASGITransport, AsyncClient

import main


@pytest.mark.asyncio
async def test_openai_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure /v1/models returns a normalized list."""

    async def fake_models(_client):  # type: ignore[override]
        return [{"id": "model-a"}, {"name": "model-b"}]

    monkeypatch.setattr(main, "lm_models", fake_models)

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

    async def fake_select(_client, model: str) -> str:  # type: ignore[override]
        return model

    async def fake_post(  # type: ignore[override]
        _client, _url: str, _payload: dict, _retries: int = 0
    ) -> dict:
        return {"choices": [{"text": "hello"}]}

    monkeypatch.setattr(main.model_selector, "ensure_selected", fake_select)
    monkeypatch.setattr(main, "_proxy_post_json", fake_post)

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
