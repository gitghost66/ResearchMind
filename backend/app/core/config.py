"""
ResearchMind configuration.

All settings are loaded from environment variables / .env file.
Never hard-code model names, API keys, or thresholds.
"""

from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings — single source of truth for all configuration."""

    # --- Supabase ---
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    database_url: str = ""

    # --- Gemini ---
    gemini_api_key: str = ""

    # --- LLM provider ---
    # "gemini" (google-genai) or "openai_compatible" / "openrouter" / "groq"
    # for any endpoint speaking the OpenAI chat-completions API.
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.1-flash-lite"
    # Only used by the openai-compatible provider.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 4096
    llm_max_attempts: int = 3
    llm_retry_base_delay: float = 1.0
    # Schema-validation repair attempts for structured output (PRD s6.3).
    llm_structured_retries: int = 2
    # USD per 1M tokens; 0.0 is correct for free tiers. Non-zero overrides the
    # built-in pricing table when computing llm_usage.cost_usd.
    llm_price_per_1m_input: float = 0.0
    llm_price_per_1m_output: float = 0.0

    # --- Embeddings ---
    embedding_model: str = "text-embedding-004"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32
    embedding_max_concurrency: int = 4

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    debug: bool = False

    # --- Research defaults ---
    max_iterations: int = 3
    max_sources_per_question: int = 10
    default_mode: str = "quick"
    concurrency_cap: int = 4

    # --- Cost control ---
    max_tokens_per_run: int = 100_000
    max_cost_per_run_usd: float = 1.00

    # --- Search ---
    search_region: str = "wt-wt"
    search_safe_search: str = "off"
    # In-process result cache (Redis was dropped from the stack).
    search_cache_ttl_seconds: int = 900
    search_cache_max_entries: int = 512

    # --- Page extraction ---
    page_max_redirects: int = 5
    page_respect_robots: bool = True
    page_robots_cache_ttl_seconds: int = 3600
    page_user_agent: str = (
        "ResearchMind/1.0 (+https://github.com/researchmind; research bot; respectful crawler)"
    )

    # --- Timeouts (seconds) ---
    agent_timeout: int = 120
    search_timeout: int = 30
    page_extract_timeout: int = 15
    page_max_size_bytes: int = 1_048_576  # 1 MB

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    def agent_config_mapping(self) -> dict[str, Any]:
        """
        Flat mapping consumed by ``AgentConfig.from_mapping``.

        The agents package must not import this class (AGENTS.md rule 2), so the
        handover is a plain dict. Passing the whole dump keeps the two sides from
        drifting as settings are added.
        """
        return self.model_dump()


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
