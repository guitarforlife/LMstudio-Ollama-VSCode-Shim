"""Tests for the FastAPI app factory."""

import pytest
from starlette.routing import Route

from shim.main import create_app


@pytest.mark.asyncio
async def test_create_app_has_routers() -> None:
    """Ensure factory includes the expected routers."""
    app = create_app()
    route_paths = {route.path for route in app.routes if isinstance(route, Route)}
    assert "/v1/models" in route_paths
    assert "/api/version" in route_paths
