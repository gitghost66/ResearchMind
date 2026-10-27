"""
API-key authentication (PRD s10, MVP).

Clients send ``X-API-Key``. The check is a constant-time comparison against
``Settings.api_key``; an empty setting disables auth for local development.
It is a dependency returning the caller's identity, so JWT or user accounts
can replace it later without touching the routes.

Browsers' ``EventSource`` cannot set headers, so a web client should read the
SSE stream with a fetch-based client that can.
"""

from __future__ import annotations

import hmac

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.errors import ApiError

API_KEY_HEADER = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def check_api_key(settings: Settings, api_key: str | None) -> str:
    """
    Validate *api_key* against the configured one.

    Returns the key if valid (``""`` when auth is disabled); raises 401 if it
    is missing and 403 if it is wrong.
    """
    if not settings.api_key:
        return ""

    if not api_key:
        raise ApiError(401, "missing_api_key", f"{API_KEY_HEADER} header required")

    if not hmac.compare_digest(api_key.encode(), settings.api_key.encode()):
        raise ApiError(403, "invalid_api_key", "Invalid API key")

    return api_key


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str:
    """Dependency form of :func:`check_api_key`, using the app's settings."""
    settings = getattr(request.app.state, "settings", None) or get_settings()
    return check_api_key(settings, api_key)
