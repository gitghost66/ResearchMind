"""
ResearchMind configuration.

All settings are loaded from environment variables / .env file.
Never hard-code model names, API keys, or thresholds.

``.env`` is looked up by absolute path, so the same file is used whether
you run from the repo root, ``backend/``, or a script: first ``<repo>/.env``,
then ``backend/.env``, which wins where both set a value. Real environment
variables override both. The agents package reads no ``.env``; it gets its
config from these settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings

_BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILES = (_BACKEND_DIR.parent / ".env", _BACKEND_DIR / ".env")


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
    # Cap on one retry wait; rate-limit errors naming a wait are honoured up to it.
    llm_max_retry_delay: float = 60.0
    # First wait after a 503 "high demand" that names no wait (LLM and
    # embeddings); doubles per attempt.
    llm_overload_retry_delay: float = 10.0
    # Schema-validation repair attempts for structured output (PRD s6.3).
    llm_structured_retries: int = 2
    # USD per 1M tokens; 0.0 is correct for free tiers. Non-zero overrides the
    # built-in pricing table when computing llm_usage.cost_usd.
    llm_price_per_1m_input: float = 0.0
    llm_price_per_1m_output: float = 0.0

    # --- Embeddings ---
    # text-embedding-004 was shut down on 2026-01-14. gemini-embedding-001 is
    # asked for embedding_dimension outputs so the vector(768) column still fits.
    embedding_model: str = "gemini-embedding-001"
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

    # --- Planner ---
    planner_min_sub_questions: int = 2
    planner_max_sub_questions: int = 5
    planner_max_queries_per_sub_question: int = 3

    # --- Web research ---
    research_pages_per_sub_question: int = 4
    research_max_fetch_attempts: int = 8
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100
    max_chunks_per_source: int = 8
    min_source_chars: int = 400

    # --- Synthesizer ---
    synth_top_k_per_sub_question: int = 5
    synth_max_passages: int = 24
    synth_max_output_tokens: int = 8192
    synth_min_similarity: float = 0.0

    # --- Cost control ---
    # 0 disables a limit.
    max_tokens_per_run: int = 100_000
    max_cost_per_run_usd: float = 1.00
    # Share of the budget held back so the synthesizer can still write a report.
    budget_synthesis_reserve: float = 0.2

    # --- Search ---
    # ddgs region; "wt-wt" breaks non-DuckDuckGo engines in its metasearch.
    search_region: str = "us-en"
    search_safe_search: str = "off"
    # ddgs backend: "auto" (several engines) or e.g. "duckduckgo,brave".
    search_backend: str = "auto"
    search_max_attempts: int = 2
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
    # Per-node safety net; long enough to wait out a free-tier rate limit.
    agent_timeout: int = 240
    search_timeout: int = 30
    page_extract_timeout: int = 15
    page_max_size_bytes: int = 1_048_576  # 1 MB

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "json"

    model_config = {
        "env_file": ENV_FILES,
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @field_validator("database_url")
    @classmethod
    def _use_async_driver(cls, value: str) -> str:
        """
        Supabase's dashboard hands out ``postgresql://`` URLs, which SQLAlchemy
        maps to a sync driver. The app is async throughout, so pick asyncpg
        unless a driver is already named.
        """
        url = value.strip()
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+asyncpg://" + url[len(prefix) :]
        return url

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
