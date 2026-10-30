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
    ``not_configured`` without ``DATABASE_URL``. ``runs`` counts queued and
    running runs across all workers; ``worker`` describes the one in this
    process, if any, and ``listening`` whether PostgreSQL notifications arrive
    (without them, streams and workers poll).
    """
    settings = app_settings(request)
    services = getattr(request.app.state, "services", None)
    checks: dict[str, str] = {}
    result: dict[str, Any] = {}

    if services is None:
        checks["database"] = "configured" if settings.database_url else "not_configured"
    else:
        checks["database"] = await _ping(services.session_factory)
        if checks["database"] == "ok":
            try:
                result["runs"] = await services.queue.counts()
            except Exception:  # noqa: BLE001 - the ping already reported the database
                logger.warning("Could not count runs", exc_info=True)
        if services.worker is not None:
            result["worker"] = {
                "id": services.worker.worker_id,
                "running": services.worker.running_count,
                "capacity": services.worker.capacity,
            }
        if services.listener is not None:
            result["listening"] = services.listener.listening.is_set()

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
