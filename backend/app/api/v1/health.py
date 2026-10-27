"""
Health and readiness endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.core.deps import app_settings

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

# A readiness probe must answer quickly even when the database does not.
_DB_PING_TIMEOUT_SECONDS = 5.0


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "healthy"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    """
    Readiness probe: is the database reachable, and is an LLM key configured?

    ``database`` is ``ok`` or ``error`` once the app has a database,
    ``not_configured`` without ``DATABASE_URL``. ``runs`` counts this process's
    background runs.
    """
    settings = app_settings(request)
    services = getattr(request.app.state, "services", None)
    checks: dict[str, str] = {}
    result: dict[str, Any] = {}

    if services is None:
        checks["database"] = "configured" if settings.database_url else "not_configured"
    else:
        checks["database"] = await _ping(services.session_factory)
        result["runs"] = {
            "running": services.runner.running_count,
            "pending": services.runner.pending_count,
        }

    checks["gemini"] = "not_configured" if not settings.gemini_api_key else "configured"

    all_ready = all(v != "error" for v in checks.values())
    return {"status": "ready" if all_ready else "degraded", "checks": checks, **result}


async def _ping(session_factory: Any) -> str:
    try:
        async with asyncio.timeout(_DB_PING_TIMEOUT_SECONDS):
            async with session_factory() as session:
                await session.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        logger.warning("Readiness database check failed: %s", exc)
        return "error"
