"""
Gemini LLM client wrapper.

Responsibilities:
- Provider-agnostic interface (model name from Settings / injected config).
- Structured output: every call either returns a validated Pydantic model or plain text.
- Automatic retry with exponential back-off on transient errors.
- Per-call token usage tracking returned as LLMUsage.

No HTTP server code, no DB sessions — pure agent-layer logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Data classes returned by the client
# ---------------------------------------------------------------------------


@dataclass
class LLMUsage:
    """Token usage for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0

    @classmethod
    def from_metadata(cls, metadata: Any, model: str, latency_ms: float) -> LLMUsage:
        """Build from google-genai UsageMetadata (may be None in mocks)."""
        if metadata is None:
            return cls(model=model, latency_ms=latency_ms)
        return cls(
            prompt_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
            total_tokens=getattr(metadata, "total_token_count", 0) or 0,
            model=model,
            latency_ms=latency_ms,
        )


@dataclass
class LLMResponse(Generic[T]):
    """Wrapper around an LLM response — text or validated Pydantic model."""

    text: str = ""
    parsed: T | None = None  # set when structured output was requested
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_RETRYABLE_MESSAGES = (
    "quota",
    "rate",
    "resource exhausted",
    "503",
    "500",
    "overloaded",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _RETRYABLE_MESSAGES)


async def _retry_async(
    coro_factory: Any,
    *,
    max_attempts: int,
    base_delay: float,
) -> Any:
    """
    Call `coro_factory()` up to *max_attempts* times, with exponential back-off.
    Only retries on transient/rate-limit errors.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc) or attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                max_attempts,
                delay,
                exc,
            )
            last_exc = exc
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class GeminiClient:
    """
    Thin async wrapper around google-genai's GenerativeModel.

    Usage::

        client = GeminiClient(api_key="...", model="gemini-3.1-flash-lite")

        # Plain text
        resp = await client.generate("Summarise: ...")

        # Structured output
        class Answer(BaseModel):
            score: int
            reason: str

        resp = await client.generate_structured("Rate this: ...", Answer)
        assert isinstance(resp.parsed, Answer)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        max_attempts: int = 3,
        base_delay: float = 1.0,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> None:
        self._model_name = model
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        self._client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse[Any]:
        """Generate a plain-text response."""
        return await self._call(
            prompt=prompt,
            system=system,
            temperature=temperature or self._temperature,
            max_output_tokens=max_output_tokens or self._max_output_tokens,
            response_schema=None,
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse[T]:
        """
        Generate a response and validate it against *schema*.

        The response is requested as JSON and parsed by Pydantic.
        Raises ValidationError if the model output cannot be validated.
        """
        resp: LLMResponse[T] = await self._call(
            prompt=prompt,
            system=system,
            temperature=temperature or self._temperature,
            max_output_tokens=max_output_tokens or self._max_output_tokens,
            response_schema=schema,
        )
        # Parse text → Pydantic model
        try:
            resp.parsed = schema.model_validate_json(resp.text)
        except (ValidationError, ValueError):
            # Fall back: try model_validate on raw dict if genai returned an object
            if hasattr(resp.raw, "parsed") and resp.raw.parsed is not None:
                resp.parsed = schema.model_validate(
                    resp.raw.parsed
                    if isinstance(resp.raw.parsed, dict)
                    else resp.raw.parsed.__dict__
                )
            else:
                raise
        return resp

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        max_output_tokens: int,
        response_schema: type[BaseModel] | None,
    ) -> LLMResponse[Any]:
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        contents: list[Any] = [prompt]

        async def _do_call() -> Any:
            return await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    **config_kwargs,
                ),
            )

        t0 = time.monotonic()
        raw = await _retry_async(
            _do_call,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        text = ""
        if raw.candidates:
            parts = raw.candidates[0].content.parts or []
            text = "".join(p.text for p in parts if hasattr(p, "text"))

        usage = LLMUsage.from_metadata(
            getattr(raw, "usage_metadata", None),
            model=self._model_name,
            latency_ms=latency_ms,
        )

        logger.debug(
            "LLM call complete model=%s tokens=%d latency_ms=%.0f",
            self._model_name,
            usage.total_tokens,
            latency_ms,
        )

        return LLMResponse(text=text, usage=usage, raw=raw)
