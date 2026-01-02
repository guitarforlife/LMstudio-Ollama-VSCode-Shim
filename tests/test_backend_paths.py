"""Integration-style backend path tests with mocked transport."""

# pylint: disable=duplicate-code,protected-access

import json
from typing import Any, cast

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import backend
import backend_api
import main
from client import BackendError


@pytest.mark.asyncio
async def test_resolve_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve model IDs via the REST API path."""
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v0/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "model-a", "state": "loaded"}]},
            )
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        monkeypatch.setattr(backend, "LMSTUDIO_REST_BASE", "http://test/api/v0")
        monkeypatch.setattr(backend, "LMSTUDIO_OPENAI_BASE", "http://test/v1")
        model_id, entry = await backend._resolve_model_id(client, None, "model-a:latest")
    assert model_id == "model-a"
    assert entry and entry["id"] == "model-a"


@pytest.mark.asyncio
async def test_openai_chat_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stream OpenAI chat responses through the shim."""
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v0/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "model-a", "state": "loaded"}]},
            )
        if request.url.path.endswith("/v1/chat/completions"):
            data = (
                b"data: "
                + json.dumps({"choices": [{"delta": {"content": "hi"}}]}).encode()
                + b"\n\n"
            )
            data += b"data: [DONE]\n\n"
            return httpx.Response(200, content=data)
        return httpx.Response(404, json={"error": "not found"})

    backend_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def client_factory() -> httpx.AsyncClient:
        return backend_client

    monkeypatch.setattr(backend, "LMSTUDIO_REST_BASE", "http://test/api/v0")
    monkeypatch.setattr(backend, "LMSTUDIO_OPENAI_BASE", "http://test/v1")

    try:
        async with main.lifespan(main.app, client_factory=client_factory):
            transport = ASGITransport(app=cast(Any, main.app))
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "model-a", "messages": [], "stream": True},
                )
                body = await resp.aread()
    finally:
        await backend_client.aclose()

    assert resp.status_code == 200
    assert b"data:" in body


@pytest.mark.asyncio
async def test_openai_chat_ttl_inject_keep_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject keep_alive into ttl for OpenAI chat payloads."""
    captured = {}

    async def fake_select(_client, _cache, model: str) -> str:  # type: ignore[override]
        return model

    async def fake_post(  # type: ignore[override]
        _client, _model_cache, _path: str, payload: dict
    ) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(backend_api.api, "ensure_selected", fake_select)
    monkeypatch.setattr(backend_api.api, "post_openai_json", fake_post)

    async with main.lifespan(main.app):
        transport = ASGITransport(app=cast(Any, main.app))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "model-a", "messages": [], "stream": False, "keep_alive": "2s"},
            )

    assert resp.status_code == 200
    assert captured["payload"]["ttl"] == 2
    assert "keep_alive" not in captured["payload"]


@pytest.mark.asyncio
async def test_openai_chat_ttl_preserves_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve explicit TTL when provided by the client."""
    captured = {}

    async def fake_select(_client, _cache, model: str) -> str:  # type: ignore[override]
        return model

    async def fake_post(  # type: ignore[override]
        _client, _model_cache, _path: str, payload: dict
    ) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(backend_api.api, "ensure_selected", fake_select)
    monkeypatch.setattr(backend_api.api, "post_openai_json", fake_post)

    async with main.lifespan(main.app):
        transport = ASGITransport(app=cast(Any, main.app))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "model-a", "messages": [], "stream": False, "ttl": 99},
            )

    assert resp.status_code == 200
    assert captured["payload"]["ttl"] == 99


@pytest.mark.asyncio
async def test_openai_models_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return 502 when backend model list is unavailable."""

    async def fake_models(_client, model_cache=None):  # type: ignore[override]
        raise BackendError(status_code=502, error="backend_error", detail="backend down")

    monkeypatch.setattr(backend_api.api, "models", fake_models)

    async with main.lifespan(main.app):
        transport = ASGITransport(app=cast(Any, main.app))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")

    assert resp.status_code == 502
