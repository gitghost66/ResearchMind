"""
OpenAI-compatible chat-completions provider.

Satisfies the PRD s4 requirement for at least two LLM providers, and gives the
project a fallback when the Gemini free tier is rate-limited. Any endpoint
speaking the ``/chat/completions`` API works — OpenRouter, Groq, Together,
a local vLLM or Ollama server — selected purely by ``LLM_PROVIDER`` and
``LLM_BASE_URL``.

Built on httpx, which the project already depends on, so this adds no new
dependency (AGENTS.md rule 8).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel

from researchmind_agents.config import LLMConfig
from researchmind_agents.llm.base import (
    BaseLLMProvider,
    RawCompletion,
    TransientLLMError,
    UsageSink,
)

logger = logging.getLogger(__name__)

__all__ = ["OpenAICompatibleProvider"]

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Chat-completions provider for any OpenAI-compatible endpoint.

    Usage::

        config = LLMConfig(
            provider="openai_compatible",
            model="meta-llama/llama-3.3-70b-instruct:free",
            base_url="https://openrouter.ai/api/v1",
            api_key="...",
        )
        provider = OpenAICompatibleProvider(config, usage_sink=sink)

    Structured output is requested via ``response_format={"type": "json_object"}``
    with the schema embedded in the system message. Strict ``json_schema`` mode
    is deliberately not used because most free endpoints do not implement it;
    the base class validates and repairs instead.
    """

    def __init__(
        self,
        config: LLMConfig,
        usage_sink: UsageSink | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config, usage_sink)
        self._base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.request_timeout)

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this provider created it."""
        if self._owns_client:
            await self._client.aclose()

    async def _complete(
        self,
        *,
        prompt: str,
        system: str | None,
        temperature: float,
        max_output_tokens: int,
        json_schema: type[BaseModel] | None,
    ) -> RawCompletion:
        messages: list[dict[str, str]] = []

        system_text = system or ""
        if json_schema is not None:
            schema_text = json.dumps(json_schema.model_json_schema(), indent=2)
            system_text = (
                f"{system_text}\n\n" if system_text else ""
            ) + (
                "Reply with a single JSON object only — no prose, no markdown fences.\n"
                f"It must validate against this JSON schema:\n{schema_text}"
            )
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        # HTTPStatusError carries .response.status_code, which the base class
        # uses to decide whether to retry.
        response.raise_for_status()
        return _to_completion(response.json())


def _to_completion(body: dict[str, Any]) -> RawCompletion:
    """Normalise an OpenAI-shaped response body into a :class:`RawCompletion`."""
    choices = body.get("choices") or []
    text = ""
    if choices:
        message = choices[0].get("message") or {}
        text = message.get("content") or ""

    usage = body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)

    if not text:
        finish_reason = choices[0].get("finish_reason") if choices else "no_choices"
        raise TransientLLMError(
            f"Provider returned no content (finish_reason={finish_reason!r})"
        )

    return RawCompletion(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw=body,
    )
