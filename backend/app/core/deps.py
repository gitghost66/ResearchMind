"""
FastAPI dependency injection.
"""

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings

# Typed dependency for use in route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
