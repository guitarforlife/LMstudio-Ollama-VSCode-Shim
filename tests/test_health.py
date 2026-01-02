"""Health endpoint tests."""

from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from main import app, lifespan


@pytest.mark.asyncio
async def test_health() -> None:
    """Test the /health endpoint returns status ok and includes timestamp."""
    async with lifespan(app):
        transport = ASGITransport(app=cast(Any, app))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["service"] == "shim"
        assert "timestamp" in payload
