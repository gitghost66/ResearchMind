"""
Health and readiness endpoints.
"""

import logging
from typing import Any

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "healthy"}


@router.get("/ready")
async def ready() -> dict[str, Any]:
    """
    Readiness probe — checks that critical dependencies are reachable.

    For M0: just returns ready. In later milestones this will verify
    database and Supabase connectivity.
    """
    settings = get_settings()
    checks: dict[str, str] = {}

    # Database check placeholder (M1)
    checks["database"] = "not_configured" if not settings.database_url else "configured"

    # Gemini API check placeholder (M2)
    checks["gemini"] = "not_configured" if not settings.gemini_api_key else "configured"

    all_ready = all(v != "error" for v in checks.values())

    return {
        "status": "ready" if all_ready else "degraded",
        "checks": checks,
    }
