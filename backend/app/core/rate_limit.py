"""
Per-client rate limiting (PRD s10), in process memory.

A sliding-window log per key: a request is allowed if fewer than ``limit``
requests from that key fell inside the last ``window`` seconds. Keys are the
API key's fingerprint when one is sent, else the client IP (behind a proxy,
run uvicorn with ``--proxy-headers`` so that is the real client).

State lives in this process, which matches the single-worker deployment
until M8; several workers would each enforce the limit separately.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from fastapi import Request

__all__ = ["SlidingWindowRateLimiter", "client_key"]


class SlidingWindowRateLimiter:
    """Allows ``limit`` hits per key in any ``window_seconds``; 0 disables it."""

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = float(window_seconds)
        self._max_keys = max(1, max_keys)
        self._clock = clock
        # Least recently seen first, so the oldest keys are evicted at the cap.
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.limit > 0 and self.window > 0

    def hit(self, key: str) -> float | None:
        """
        Count one request for *key*.

        Returns ``None`` if it is allowed, otherwise the seconds until it would
        be (rounded up, for ``Retry-After``). A rejected request is not counted.
        """
        if not self.enabled:
            return None
        now = self._clock()
        hits = self._hits.get(key)
        if hits is None:
            hits = deque()
            self._hits[key] = hits
            while len(self._hits) > self._max_keys:
                self._hits.popitem(last=False)
        else:
            self._hits.move_to_end(key)

        cutoff = now - self.window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(1.0, math.ceil(hits[0] + self.window - now))
        hits.append(now)
        return None


def client_key(request: Request, api_key: str | None) -> str:
    """Who to count a request against: the API key if present, else the IP."""
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"
