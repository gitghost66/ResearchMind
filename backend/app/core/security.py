"""
Security utilities — API key validation, rate limiting helpers.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """
    Validate the API key from the X-API-Key header.

    Returns the key if valid; raises 401 if missing, 403 if wrong.
    For MVP: simple string comparison. Structured so JWT / user accounts
    can replace this later.
    """
    settings = get_settings()

    # If no API key is configured, skip auth (dev mode)
    if not settings.api_key:
        return ""

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_api_key", "message": "X-API-Key header required"},
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "invalid_api_key", "message": "Invalid API key"},
        )

    return api_key
