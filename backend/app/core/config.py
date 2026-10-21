"""
ResearchMind configuration.

All settings are loaded from environment variables / .env file.
Never hard-code model names, API keys, or thresholds.
"""

from functools import lru_cache

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

    # --- Models ---
    llm_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "text-embedding-004"
    embedding_dimension: int = 768

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


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
