"""
ResearchMind — FastAPI application entry point.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown."""
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)
    logger.info(
        "ResearchMind API starting",
        extra={"llm_model": settings.llm_model, "debug": settings.debug},
    )
    yield
    logger.info("ResearchMind API shutting down")


app = FastAPI(
    title="ResearchMind API",
    description="Multi-agent AI research system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive for development, tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global error handler — consistent error envelope
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return a consistent error envelope."""
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "details": str(exc) if get_settings().debug else None,
            }
        },
    )


# Mount routers
# Health endpoints at root level
app.include_router(v1_router)

# API v1 endpoints (research, sessions, etc.) will be mounted here in M4:
# app.include_router(v1_research_router, prefix="/api/v1")
