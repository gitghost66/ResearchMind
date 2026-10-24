"""
Agent-layer configuration.

The agents package must not import the backend's ``Settings`` (AGENTS.md rule 2),
but it must also not hard-code model names, limits, or thresholds (rule 4).
This module is the bridge: plain frozen dataclasses that the backend populates
from ``Settings`` and passes in.

Build one with :func:`AgentConfig.from_mapping` so the backend only has to hand
over a flat dict of its settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_USER_AGENT = (
    "ResearchMind/1.0 (+https://github.com/researchmind; research bot; respectful crawler)"
)


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for a single LLM provider."""

    provider: str = "gemini"
    model: str = "gemini-3.1-flash-lite"
    api_key: str = ""
    # Only used by the openai-compatible provider (OpenRouter, Groq, vLLM, ...).
    base_url: str = ""
    temperature: float = 0.2
    max_output_tokens: int = 4096
    # Transport-level retries (rate limits, 5xx).
    max_attempts: int = 3
    base_delay: float = 1.0
    # Schema-validation retries, fed the validation error (PRD s6.3).
    structured_retries: int = 2
    request_timeout: float = 120.0
    # USD per 1M tokens. 0.0 is correct for free tiers; a non-zero value here
    # always wins over the built-in pricing table.
    price_per_1m_input: float = 0.0
    price_per_1m_output: float = 0.0


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding service."""

    model: str = "text-embedding-004"
    dimension: int = 768
    # Texts per request to the provider's batch endpoint.
    batch_size: int = 32
    # Concurrent in-flight batch requests. Keep low on free tiers.
    max_concurrency: int = 4
    max_attempts: int = 3
    base_delay: float = 1.0
    api_key: str = ""


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for the web search tool."""

    max_results: int = 10
    timeout: int = 30
    region: str = "wt-wt"
    safe_search: str = "off"
    # Concurrent queries in search_multi. The free endpoint rate-limits hard,
    # so this is deliberately separate from max_results.
    max_concurrency: int = 4
    # Redis was dropped from the stack, so the cache is in-process.
    cache_ttl_seconds: int = 900
    cache_max_entries: int = 512


@dataclass(frozen=True)
class PageExtractConfig:
    """Configuration for the page extraction tool."""

    timeout: int = 15
    max_size_bytes: int = 1_048_576
    max_redirects: int = 5
    user_agent: str = DEFAULT_USER_AGENT
    respect_robots: bool = True
    robots_cache_ttl_seconds: int = 3600
    allowed_content_types: tuple[str, ...] = (
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        "application/xml",
        "text/xml",
    )


@dataclass(frozen=True)
class AgentConfig:
    """Top-level configuration handed to the graph and its nodes."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    page_extract: PageExtractConfig = field(default_factory=PageExtractConfig)

    concurrency_cap: int = 4
    agent_timeout: int = 120
    max_iterations: int = 3
    max_sources_per_question: int = 10
    max_tokens_per_run: int = 100_000
    max_cost_per_run_usd: float = 1.00

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> AgentConfig:
        """
        Build a config from a flat mapping of backend setting names.

        Unknown keys are ignored, and any key that is absent falls back to the
        dataclass default, so the backend can pass ``Settings.model_dump()``
        straight through.
        """

        def pick(key: str, default: Any) -> Any:
            value = values.get(key)
            return default if value is None else value

        llm = LLMConfig(
            provider=pick("llm_provider", LLMConfig.provider),
            model=pick("llm_model", LLMConfig.model),
            api_key=pick("gemini_api_key", LLMConfig.api_key),
            base_url=pick("llm_base_url", LLMConfig.base_url),
            temperature=pick("llm_temperature", LLMConfig.temperature),
            max_output_tokens=pick("llm_max_output_tokens", LLMConfig.max_output_tokens),
            max_attempts=pick("llm_max_attempts", LLMConfig.max_attempts),
            base_delay=pick("llm_retry_base_delay", LLMConfig.base_delay),
            structured_retries=pick("llm_structured_retries", LLMConfig.structured_retries),
            request_timeout=pick("agent_timeout", LLMConfig.request_timeout),
            price_per_1m_input=pick("llm_price_per_1m_input", LLMConfig.price_per_1m_input),
            price_per_1m_output=pick("llm_price_per_1m_output", LLMConfig.price_per_1m_output),
        )
        # An openai-compatible endpoint uses its own key, not the Gemini one.
        if llm.provider != "gemini" and values.get("llm_api_key"):
            llm = replace(llm, api_key=values["llm_api_key"])

        embedding = EmbeddingConfig(
            model=pick("embedding_model", EmbeddingConfig.model),
            dimension=pick("embedding_dimension", EmbeddingConfig.dimension),
            batch_size=pick("embedding_batch_size", EmbeddingConfig.batch_size),
            max_concurrency=pick("embedding_max_concurrency", EmbeddingConfig.max_concurrency),
            max_attempts=pick("llm_max_attempts", EmbeddingConfig.max_attempts),
            base_delay=pick("llm_retry_base_delay", EmbeddingConfig.base_delay),
            api_key=pick("gemini_api_key", EmbeddingConfig.api_key),
        )

        search = SearchConfig(
            max_results=pick("max_sources_per_question", SearchConfig.max_results),
            timeout=pick("search_timeout", SearchConfig.timeout),
            region=pick("search_region", SearchConfig.region),
            safe_search=pick("search_safe_search", SearchConfig.safe_search),
            max_concurrency=pick("concurrency_cap", SearchConfig.max_concurrency),
            cache_ttl_seconds=pick("search_cache_ttl_seconds", SearchConfig.cache_ttl_seconds),
            cache_max_entries=pick("search_cache_max_entries", SearchConfig.cache_max_entries),
        )

        page_extract = PageExtractConfig(
            timeout=pick("page_extract_timeout", PageExtractConfig.timeout),
            max_size_bytes=pick("page_max_size_bytes", PageExtractConfig.max_size_bytes),
            max_redirects=pick("page_max_redirects", PageExtractConfig.max_redirects),
            user_agent=pick("page_user_agent", PageExtractConfig.user_agent),
            respect_robots=pick("page_respect_robots", PageExtractConfig.respect_robots),
            robots_cache_ttl_seconds=pick(
                "page_robots_cache_ttl_seconds",
                PageExtractConfig.robots_cache_ttl_seconds,
            ),
        )

        return cls(
            llm=llm,
            embedding=embedding,
            search=search,
            page_extract=page_extract,
            concurrency_cap=pick("concurrency_cap", cls.concurrency_cap),
            agent_timeout=pick("agent_timeout", cls.agent_timeout),
            max_iterations=pick("max_iterations", cls.max_iterations),
            max_sources_per_question=pick(
                "max_sources_per_question", cls.max_sources_per_question
            ),
            max_tokens_per_run=pick("max_tokens_per_run", cls.max_tokens_per_run),
            max_cost_per_run_usd=pick("max_cost_per_run_usd", cls.max_cost_per_run_usd),
        )
