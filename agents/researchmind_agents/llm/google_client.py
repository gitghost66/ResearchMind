"""
Shared google-genai client.

Constructing ``genai.Client`` costs roughly half a second and builds a fresh
connection pool each time. The chat provider and the embedding service both
need one, and M4 creates a runtime per research run, so clients are cached per
API key instead of rebuilt.

The client is only used through ``asyncio.to_thread``, and google-genai's
client is safe to share across those calls.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from google import genai

__all__ = ["shared_genai_client", "reset_genai_clients"]


@lru_cache(maxsize=8)
def _cached_client(api_key: str) -> Any:
    return genai.Client(api_key=api_key)


def shared_genai_client(api_key: str) -> Any:
    """
    Return a cached ``genai.Client`` for *api_key*.

    Raises ``ValueError`` from the SDK when the key is empty, which is the
    desired fail-fast behaviour at the point of first real use.
    """
    return _cached_client(api_key)


def reset_genai_clients() -> None:
    """Drop cached clients. Intended for tests and key rotation."""
    _cached_client.cache_clear()
