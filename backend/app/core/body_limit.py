"""
Request body size limit (PRD s10, input length limits).

A pure ASGI middleware: a declared ``Content-Length`` over the limit is
rejected before the body is read; a body sent without one (chunked) is read
up to the limit and rejected as soon as it goes over, then replayed to the
app unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import error_body

__all__ = ["BodySizeLimitMiddleware"]

_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: Callable[[], int]) -> None:
        self.app = app
        # Read per request, so the limit follows the app's settings.
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return
        limit = self._max_bytes()
        if limit <= 0:
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None:
            if declared > limit:
                await _reject(send, limit)
                return
            await self.app(scope, receive, send)
            return

        # No Content-Length: read (bounded) before handing over.
        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            total += len(message.get("body", b""))
            if total > limit:
                await _reject(send, limit)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            return buffered.pop(0) if buffered else await receive()

        await self.app(scope, replay, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _reject(send: Send, limit: int) -> None:
    body: dict[str, Any] = error_body(
        "payload_too_large",
        f"Request body is larger than {limit} bytes",
        {"max_bytes": limit},
    )
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})
