"""
Provider selection.

Keeps the ``LLM_PROVIDER`` -> class mapping in one place so nodes never name a
concrete provider and the choice stays a config value (AGENTS.md rule 4).
"""

from __future__ import annotations

from collections.abc import Callable

from researchmind_agents.config import LLMConfig
from researchmind_agents.llm.base import BaseLLMProvider, UsageSink
from researchmind_agents.llm.client import GeminiProvider
from researchmind_agents.llm.openai_compatible import OpenAICompatibleProvider

__all__ = ["PROVIDERS", "build_llm_provider", "available_providers"]

ProviderFactory = Callable[[LLMConfig, UsageSink | None], BaseLLMProvider]

PROVIDERS: dict[str, ProviderFactory] = {
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "openrouter": OpenAICompatibleProvider,
    "groq": OpenAICompatibleProvider,
}


def available_providers() -> list[str]:
    """Provider ids accepted by :func:`build_llm_provider`."""
    return sorted(PROVIDERS)


def build_llm_provider(
    config: LLMConfig,
    usage_sink: UsageSink | None = None,
) -> BaseLLMProvider:
    """
    Instantiate the provider named by ``config.provider``.

    Raises ``ValueError`` on an unknown provider so a typo in ``.env`` fails at
    startup rather than mid-run.
    """
    key = (config.provider or "").strip().lower()
    factory = PROVIDERS.get(key)
    if factory is None:
        raise ValueError(
            f"Unknown LLM provider {config.provider!r}. "
            f"Available: {', '.join(available_providers())}"
        )
    return factory(config, usage_sink)
