"""
Gemini provider.

Implements the single provider-specific method required by
:class:`~researchmind_agents.llm.base.BaseLLMProvider`; retries, schema
validation, cost, and usage recording all live in the base class.

No HTTP server code, no DB sessions — pure agent-layer logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.genai import types as genai_types
from pydantic import BaseModel

from researchmind_agents.config import LLMConfig
from researchmind_agents.llm.base import (
    BaseLLMProvider,
    LLMResponse,
    LLMUsage,
    RawCompletion,
    TransientLLMError,
    UsageSink,
)
from researchmind_agents.llm.google_client import shared_genai_client

logger = logging.getLogger(__name__)

__all__ = ["GeminiProvider", "GeminiClient", "LLMResponse", "LLMUsage"]


class GeminiProvider(BaseLLMProvider):
    """
    Async wrapper around google-genai.

    Usage::

        config = LLMConfig(api_key="...", model="gemini-3.1-flash-lite")
        provider = GeminiProvider(config, usage_sink=sink)

        resp = await provider.generate("Summarise: ...", agent="planner", run_id=run_id)

        class Answer(BaseModel):
            score: int
            reason: str

        resp = await provider.generate_structured("Rate this: ...", Answer, agent="judge")
        assert isinstance(resp.parsed, Answer)
    """

    def __init__(
        self,
        config: LLMConfig,
        usage_sink: UsageSink | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(config, usage_sink)
        self._client = client or shared_genai_client(config.api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def _complete(
        self,
        *,
        prompt: str,
        system: str | None,
        temperature: float,
        max_output_tokens: int,
        json_schema: type[BaseModel] | None,
    ) -> RawCompletion:
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if json_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = json_schema

        def call() -> Any:
            return self._client.models.generate_content(
                model=self._config.model,
                contents=[prompt],
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    **config_kwargs,
                ),
            )

        raw = await asyncio.to_thread(call)
        return _to_completion(raw)


def _to_completion(raw: Any) -> RawCompletion:
    """Normalise a google-genai response into a :class:`RawCompletion`."""
    text = _response_text(raw)

    metadata = getattr(raw, "usage_metadata", None)
    prompt_tokens = int(getattr(metadata, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(metadata, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(metadata, "total_token_count", 0) or 0)

    if not text:
        # A blocked or truncated candidate yields no text. Surface it as
        # transient so the retry path gets one more shot before the node fails.
        reason = _finish_reason(raw)
        feedback = getattr(raw, "prompt_feedback", None)
        raise TransientLLMError(
            f"Gemini returned no text (finish_reason={reason!r}, prompt_feedback={feedback!r})"
        )

    return RawCompletion(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw=raw,
    )


def _response_text(raw: Any) -> str:
    """Concatenate the text parts of the first candidate."""
    candidates = getattr(raw, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        value = getattr(part, "text", None)
        if isinstance(value, str) and value:
            chunks.append(value)
    return "".join(chunks)


def _finish_reason(raw: Any) -> str:
    candidates = getattr(raw, "candidates", None) or []
    if not candidates:
        return "no_candidates"
    return str(getattr(candidates[0], "finish_reason", "unknown"))


# Pre-refactor name, kept so existing imports keep working.
GeminiClient = GeminiProvider
