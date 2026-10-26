"""Tests for health and readiness endpoints."""

import pytest
from httpx import AsyncClient

from app.core.config import get_settings


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    """GET /health should always return 200 with status=healthy."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_ready_returns_200(client: AsyncClient):
    """GET /ready should return 200 with status and checks."""
    response = await client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "database" in data["checks"]
    assert "gemini" in data["checks"]


@pytest.mark.asyncio
async def test_ready_shows_not_configured_without_env(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    """Without env vars, /ready should show services as not_configured."""
    # conftest already ignores .env files; also drop shell-exported values.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    try:
        response = await client.get("/ready")
    finally:
        get_settings.cache_clear()
    data = response.json()
    assert data["checks"]["database"] == "not_configured"
    assert data["checks"]["gemini"] == "not_configured"


@pytest.mark.asyncio
async def test_nonexistent_route_returns_404(client: AsyncClient):
    """Unknown routes should return 404, not 500."""
    response = await client.get("/does-not-exist")
    assert response.status_code == 404
