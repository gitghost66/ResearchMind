"""
ResearchMind — FastAPI application entry point.

``create_app`` builds the app; the module-level ``app`` is what uvicorn serves.
Settings are resolved at startup, not import, so importing this module never
reads ``.env``.

The lifespan, when ``DATABASE_URL`` is set:

1. builds the shared services (event recorder, run queue, rate limiters, and
   with ``RUN_WORKER_IN_API`` a research worker);
2. starts listening for PostgreSQL notifications and starts the worker, which
   resumes runs whose worker died and claims queued ones;
3. on shutdown, the worker interrupts its runs and releases them for another
   worker to resume, waiting up to ``SHUTDOWN_GRACE_SECONDS``.

Runs live in the database, not in this process: any number of API processes
(``uvicorn --workers N``) and worker processes can share it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.v1.router import router as v1_router
from app.container import AppServices, build_services
from app.core.body_limit import BodySizeLimitMiddleware
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.core.logging import setup_logging
from app.core.tracing import configure_tracing
from app.db.session import get_engine, get_session_factory

logger = logging.getLogger(__name__)

__all__ = ["app", "create_app"]


class _SettingsCORS:
    """
    CORS from ``CORS_ALLOW_ORIGINS``, read when the first request arrives, so
    building the app never reads settings. Requests authenticate with the
    ``X-API-Key`` header, never cookies, so credentials are not allowed.
    """

    def __init__(self, app: ASGIApp, *, origins: Callable[[], list[str]]) -> None:
        self.app = app
        self._origins = origins
        self._cors: ASGIApp | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if self._cors is None:
            self._cors = CORSMiddleware(
                self.app,
                allow_origins=self._origins(),
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["Retry-After"],
            )
        await self._cors(scope, receive, send)


def create_app(
    *,
    settings: Settings | None = None,
    services: AppServices | None = None,
) -> FastAPI:
    """
    Build the API. Tests pass *services* (and *settings*) built around their
    own database; otherwise both come from the environment at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        resolved = app.state.settings or get_settings()
        app.state.settings = resolved
        setup_logging(level=resolved.log_level, fmt=resolved.log_format)
        configure_tracing(resolved)
        logger.info(
            "ResearchMind API starting",
            extra={"llm_model": resolved.llm_model, "debug": resolved.debug},
        )

        owns_engine = False
        if app.state.services is None and resolved.database_url:
            app.state.services = build_services(resolved, get_session_factory())
            owns_engine = True
        active: AppServices | None = app.state.services
        if active is None:
            logger.warning("DATABASE_URL is not set: research endpoints answer 503")
        else:
            await active.startup()

        yield

        if active is not None:
            await active.shutdown()
        if owns_engine:
            await get_engine().dispose()
        logger.info("ResearchMind API shutting down")

    app = FastAPI(
        title="ResearchMind API",
        description="Multi-agent AI research system",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.services = services

    def _max_body_bytes() -> int:
        return (app.state.settings or get_settings()).max_request_body_bytes

    def _cors_origins() -> list[str]:
        return (app.state.settings or get_settings()).cors_origins

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=_max_body_bytes)
    app.add_middleware(_SettingsCORS, origins=_cors_origins)
    install_error_handlers(app)

    # Health endpoints at root level; research under /api/v1/research.
    app.include_router(v1_router)
    return app


app = create_app()
