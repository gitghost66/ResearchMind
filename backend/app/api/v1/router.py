"""
V1 API router — assembles all v1 sub-routers.
"""

from fastapi import APIRouter

from app.api.v1.health import router as health_router

router = APIRouter()

# Health endpoints at root level (no /api/v1 prefix for health)
router.include_router(health_router)
