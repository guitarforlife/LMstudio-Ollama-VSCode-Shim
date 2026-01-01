"""Readiness endpoint tests."""

import pytest
from httpx import ASGITransport, AsyncClient

import main
from main import app, lifespan


@pytest.mark.asyncio
async def test_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the /ready endpoint returns status ok and includes timestamp."""
    async def fake_preflight(_client):
        return None

    monkeypatch.setattr(main, "_preflight_lmstudio", fake_preflight)
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ready")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["backend"] == "lmstudio"
        assert "version" in payload
        assert "timestamp" in payload
