"""
The API's error envelope (PRD s9): every error is

    {"error": {"code": "...", "message": "...", "details": ...}}

Routes raise :class:`ApiError`; FastAPI's own errors (404 for unknown paths,
405, request validation) and unhandled exceptions are converted here, so a
client never sees another shape.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings

logger = logging.getLogger(__name__)

__all__ = ["ApiError", "error_body", "install_error_handlers"]

_CODES_BY_STATUS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


class ApiError(Exception):
    """An error with a stable ``code`` clients can switch on."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.headers = dict(headers or {})


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


def _response(
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(error_body(code, message, details)),
        headers=dict(headers or {}),
    )


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return _response(exc.status_code, exc.code, exc.message, exc.details, exc.headers)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _CODES_BY_STATUS.get(exc.status_code, "error")
    message = str(exc.detail) if exc.detail else code.replace("_", " ")
    details = None
    # Dependencies raise HTTPException(detail={"code": ..., "message": ...}).
    if isinstance(exc.detail, Mapping):
        code = str(exc.detail.get("code", code))
        message = str(exc.detail.get("message", code))
        details = exc.detail.get("details")
    return _response(exc.status_code, code, message, details, exc.headers)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return _response(422, "validation_error", "Invalid request", details)


async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", exc_info=exc)
    return _response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred",
        str(exc) if get_settings().debug else None,
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)
