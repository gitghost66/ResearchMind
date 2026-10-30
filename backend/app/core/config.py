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
    # Connection for LISTEN/NOTIFY (live events and run signals across
    # processes). Empty uses DATABASE_URL. Supabase's transaction pooler
    # (port 6543) cannot LISTEN: point this at the session pooler (port 5432)
    # or the direct connection, or processes fall back to polling.
    database_listen_url: str = ""

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
    # Clients send it as X-API-Key. Empty disables auth (local development only).
    api_key: str = ""
    debug: bool = False
    # Browser origins allowed to call the API (comma-separated, or "*").
    # Requests carry X-API-Key, not cookies, so credentials are never allowed.
    cors_allow_origins: str = "*"

    # --- API limits ---
    max_question_chars: int = 2000
    # Size of user_context once serialised to JSON.
    max_user_context_bytes: int = 8192
    # Requests declaring a larger Content-Length are rejected with 413.
    max_request_body_bytes: int = 65_536
    # Upper bound for a request's max_sources.
    max_sources_limit: int = 50
    # Per API key (or client IP without one), over a sliding window. 0 disables.
    rate_limit_window_seconds: int = 60
    rate_limit_requests: int = 120
    # Run creation is expensive, so it has its own, tighter limit.
    rate_limit_runs: int = 10

    # --- Background runs ---
    # Runs are queued in the database and executed by workers
    # (python -m app.workers.research_worker), each claiming runs under a lease.
    # Runs one worker executes at once; further runs wait in the queue.
    max_concurrent_runs: int = 2
    # Queued plus running, across all workers; beyond this, new runs get 503.
    max_pending_runs: int = 20
    # Also run a worker inside the API process (one container is enough).
    # Turn off when separate worker processes run.
    run_worker_in_api: bool = True
    # A worker renews its runs' lease every third of this. A run whose lease
    # lapses (its worker died) is resumed from its last checkpoint by any worker.
    run_lease_seconds: float = 60.0
    # How often an idle worker looks for queued runs without being notified.
    worker_poll_seconds: float = 5.0
    # Times a worker may start a run before an interrupted run is failed
    # instead of resumed (it may be what kills its worker).
    max_run_attempts: int = 3
    # How long cancel/delete wait for a run in this process to stop.
    cancel_wait_seconds: float = 10.0
    # On shutdown, a worker interrupts its runs and hands them back (released
    # for another worker to resume), waiting this long for them to stop.
    shutdown_grace_seconds: float = 10.0

    # --- Event stream (SSE) ---
    # Events are pushed as they are written; this is the fallback poll for
    # events written by another process.
    sse_poll_interval_seconds: float = 2.0
    # Reconnect delay suggested to EventSource clients.
    sse_retry_ms: int = 3000

    # --- Research defaults ---
    # Research rounds in deep, academic, and decision modes: the first plus up
    # to max_iterations - 1 gap-filling rounds. Quick mode always runs one.
    max_iterations: int = 3
    # Search results requested per query.
    max_sources_per_question: int = 10
    # Sources stored per run across all sub-questions (0 = no cap). A request's
    # max_sources overrides it for that run.
    max_sources_per_run: int = 0
    default_mode: str = "quick"
    concurrency_cap: int = 4

    # --- Planner ---
    # Quick mode.
    planner_min_sub_questions: int = 2
    planner_max_sub_questions: int = 5
    planner_max_queries_per_sub_question: int = 3
    # Deep, academic, and decision modes (PRD: 4-8 sub-questions).
    planner_verified_min_sub_questions: int = 4
    planner_verified_max_sub_questions: int = 8
    # Gap-filling rounds: most new sub-questions per round.
    planner_max_gap_sub_questions: int = 3

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
    # Verified modes: claims shown to the model, and the lowest claim
    # confidence a report may use.
    synth_max_claims: int = 40
    synth_min_claim_confidence: float = 0.0

    # --- Academic research (arXiv, Semantic Scholar) ---
    academic_search_providers: str = "arxiv,semantic_scholar"
    # Papers requested per query from each provider.
    academic_max_results: int = 8
    academic_search_timeout: float = 20.0
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    # arXiv asks for three seconds between API requests.
    arxiv_min_interval_seconds: float = 3.0
    semantic_scholar_api_url: str = "https://api.semanticscholar.org/graph/v1/paper/search"
    # Optional; without it requests share a public, heavily rate-limited pool.
    semantic_scholar_api_key: str = ""
    semantic_scholar_min_interval_seconds: float = 1.0
    # Papers whose abstract is shorter than this are skipped.
    min_abstract_chars: int = 200

    # --- Claim extraction ---
    claims_passages_per_sub_question: int = 5
    claims_max_passages: int = 24
    claims_max_per_sub_question: int = 8
    claims_max_output_tokens: int = 4096

    # --- Fact checking ---
    # Passages per claim: its own, then similar passages from other sources.
    fact_check_evidence_per_claim: int = 4
    fact_check_min_similarity: float = 0.0
    # Characters of each passage the fact checker reads.
    fact_check_excerpt_chars: int = 700
    # Claims per model call.
    fact_check_batch_size: int = 6
    fact_check_max_output_tokens: int = 4096

    # --- Contradiction detection ---
    # Claim pairs at least this similar (cosine) are compared by the model;
    # pairs the fact checker linked are always compared.
    contradiction_min_similarity: float = 0.7
    # Most pairs compared per round (most similar first), claims per topic,
    # and topics per model call.
    contradiction_max_pairs: int = 40
    contradiction_max_group_size: int = 6
    contradiction_groups_per_call: int = 4
    contradiction_excerpts_per_claim: int = 2
    contradiction_max_output_tokens: int = 4096

    # --- Sufficiency check and the gap-filling loop ---
    # A sub-question is covered with this many usable claims from this many
    # independent sites; otherwise it is a gap.
    sufficiency_min_claims_per_sub_question: int = 2
    sufficiency_min_sites_per_sub_question: int = 2
    # When every sub-question is covered, the model judges what is missing.
    sufficiency_llm_check: bool = True
    sufficiency_max_gaps: int = 3
    sufficiency_claims_shown_per_sub_question: int = 6
    sufficiency_max_output_tokens: int = 2048
    # Seconds into a run after which no new round starts, per mode, as JSON;
    # merged over the defaults ({"deep": 150, "academic": 180, "decision": 150}).
    # 0 turns a mode's deadline off.
    loop_deadline_seconds: dict[str, float] = {}

    # --- Scoring (PRD s8; formulas in agents/researchmind_agents/reliability.py) ---
    # Base reliability per domain type, as JSON; merged over the defaults,
    # e.g. {"forum": 0.25, "news": 0.75}.
    reliability_type_scores: dict[str, float] = {}
    reliability_recency_full_years: float = 3.0
    reliability_recency_penalty_per_year: float = 0.05
    reliability_recency_floor: float = 0.75
    reliability_unknown_date_factor: float = 0.95
    reliability_citation_bonus_max: float = 0.10
    reliability_citation_saturation: int = 1000
    reliability_methodology_bonus: float = 0.05
    # Claim confidence multiplier per verdict, as JSON; merged over the defaults.
    confidence_verdict_weights: dict[str, float] = {}
    # Most one fully reliable source establishes alone (noisy-OR across sources).
    confidence_source_weight: float = 0.8
    confidence_uncovered_penalty: float = 0.5
    confidence_contradiction_penalty: float = 0.10
    confidence_contradiction_penalty_cap: float = 0.5
    # Likely causes under which a contradiction is resolved (the claims differ
    # in scope), comma-separated; other contradictions lower the confidence.
    confidence_resolved_causes: str = "conditions,date,version,dataset,definition"

    # --- Evaluation (M7; python -m researchmind_agents.evals) ---
    # LLM-as-judge model; empty uses llm_model.
    eval_judge_model: str = ""
    # Precision@K / Recall@K cut-offs, comma-separated.
    eval_retrieval_k: str = "5,10"
    # Single-agent RAG baseline: pages read for its one query, passages retrieved.
    eval_rag_pages: int = 6
    eval_rag_top_k: int = 12
    eval_judge_batch_size: int = 6
    eval_judge_excerpt_chars: int = 700
    eval_max_statements: int = 40
    # Closed-book baseline: distinct cited URLs fetched per answer.
    eval_baseline_max_urls: int = 20
    eval_spot_check_sample: int = 30
    eval_max_output_tokens: int = 4096

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

    # --- Tracing (optional) ---
    # When enabled, these are exported as LANGSMITH_* environment variables at
    # startup; each run is then one trace with a span per graph node.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "researchmind"
    langsmith_endpoint: str = ""

    model_config = {
        "env_file": ENV_FILES,
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @field_validator("database_url", "database_listen_url")
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

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

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
