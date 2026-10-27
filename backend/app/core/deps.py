"""
FastAPI dependency injection.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Security

# Imported at runtime: FastAPI resolves these annotations to build dependencies.
from app.container import AppServices
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.rate_limit import SlidingWindowRateLimiter, client_key
from app.core.security import api_key_header, check_api_key

__all__ = [
    "SettingsDep",
    "app_settings",
    "get_services",
    "limit_requests",
    "limit_run_creation",
]

# Typed dependency for use in route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]


def app_settings(request: Request) -> Settings:
    """The app's settings: the ones it was created with, else the environment's."""
    settings = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


def get_services(request: Request) -> AppServices:
    services: AppServices | None = getattr(request.app.state, "services", None)
    if services is None:
        raise ApiError(
            503,
            "database_not_configured",
            "Research endpoints need a database; set DATABASE_URL",
        )
    return services


def _limit(limiter: SlidingWindowRateLimiter, key: str, what: str) -> None:
    retry_after = limiter.hit(key)
    if retry_after is not None:
        raise ApiError(
            429,
            "rate_limited",
            f"Too many {what}; retry in {int(retry_after)}s",
            {"retry_after_seconds": int(retry_after)},
            headers={"Retry-After": str(int(retry_after))},
        )


def _authenticate(request: Request, api_key: str | None) -> str:
    """
    Check the key before anything else (so a caller without one learns nothing
    about the server, not even whether it has a database). Failed attempts
    count against the caller's IP, so keys cannot be guessed at full speed.
    """
    try:
        return check_api_key(app_settings(request), api_key)
    except ApiError:
        services: AppServices | None = getattr(request.app.state, "services", None)
        if services is not None:
            _limit(services.request_limiter, client_key(request, None), "failed API key attempts")
        raise


async def limit_requests(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)],
) -> None:
    """Authenticate, then count the request against the caller's limit."""
    key = _authenticate(request, api_key)
    services = get_services(request)
    _limit(services.request_limiter, client_key(request, key), "requests")


async def limit_run_creation(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)],
) -> None:
    """The separate, tighter limit on starting runs."""
    key = _authenticate(request, api_key)
    services = get_services(request)
    _limit(services.run_limiter, client_key(request, key), "research runs created")
