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
    # Longest single wait between retries. A rate-limit error that names its
    # own wait ("retry in 46s") is honoured up to this cap.
    max_retry_delay: float = 60.0
    # First wait after an overload (503 "high demand") that names no wait;
    # doubles per attempt. A 1s retry usually meets the same spike.
    overload_delay: float = 10.0
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

    # text-embedding-004 was shut down on 2026-01-14. gemini-embedding-001
    # defaults to 3072 dims, so the service always requests ``dimension``.
    model: str = "gemini-embedding-001"
    dimension: int = 768
    # Texts per request to the provider's batch endpoint.
    batch_size: int = 32
    # Concurrent in-flight batch requests. Keep low on free tiers.
    max_concurrency: int = 4
    max_attempts: int = 3
    base_delay: float = 1.0
    max_retry_delay: float = 60.0
    overload_delay: float = 10.0
    api_key: str = ""


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for the web search tool."""

    max_results: int = 10
    timeout: int = 30
    # ddgs region code. DuckDuckGo's "wt-wt" (no region) breaks other engines
    # in ddgs's metasearch, e.g. it becomes the host wt.wikipedia.org.
    region: str = "us-en"
    safe_search: str = "off"
    # ddgs backend: "auto" (several engines, merged) or a comma-separated list
    # such as "duckduckgo,brave".
    backend: str = "auto"
    # Total tries per query; each "auto" retry draws a different engine set.
    max_attempts: int = 2
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
class PlannerConfig:
    """Bounds for the planner's decomposition."""

    # Quick mode.
    min_sub_questions: int = 2
    max_sub_questions: int = 5
    max_queries_per_sub_question: int = 3
    # Verified modes (deep, academic, decision); PRD s6.2: 4-8.
    verified_min_sub_questions: int = 4
    verified_max_sub_questions: int = 8
    # Gap-filling rounds: most new sub-questions per round.
    max_gap_sub_questions: int = 3

    def bounds(self, *, verified: bool) -> tuple[int, int]:
        """(min, max) sub-questions for a first plan."""
        if verified:
            return self.verified_min_sub_questions, self.verified_max_sub_questions
        return self.min_sub_questions, self.max_sub_questions


@dataclass(frozen=True)
class ResearchConfig:
    """Web research node: how many pages to read and how to chunk them."""

    pages_per_sub_question: int = 4
    # Failed fetches are replaced from the remaining candidates, up to this
    # many fetch attempts per sub-question in total.
    max_fetch_attempts: int = 8
    # Token counts are estimated (~4 chars per token); no tokenizer dependency.
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100
    max_chunks_per_source: int = 8
    # Pages with less readable text than this are skipped as boilerplate.
    min_source_chars: int = 400
    # Sources stored per run, across all branches (0 = no cap). The API's
    # per-run ``max_sources`` arrives here as an override.
    max_sources_per_run: int = 0


@dataclass(frozen=True)
class SynthesizerConfig:
    """Retrieval and generation limits for the synthesizer."""

    top_k_per_sub_question: int = 5
    max_passages: int = 24
    max_output_tokens: int = 8192
    # Retrieved passages below this cosine similarity are ignored (0 = keep all).
    min_similarity: float = 0.0
    # Verified modes: most claims shown to the model, and the lowest claim
    # confidence that may appear in a report.
    max_claims: int = 40
    min_claim_confidence: float = 0.0


@dataclass(frozen=True)
class AcademicSearchConfig:
    """Scholarly search for the academic researcher (arXiv, Semantic Scholar)."""

    # Providers queried for every search, merged round-robin.
    providers: tuple[str, ...] = ("arxiv", "semantic_scholar")
    # Papers requested per query from each provider.
    max_results: int = 8
    timeout: float = 20.0
    # Total tries per provider and query; failures degrade to no results.
    max_attempts: int = 2
    arxiv_url: str = "https://export.arxiv.org/api/query"
    # arXiv asks API clients to wait three seconds between requests.
    arxiv_min_interval: float = 3.0
    semantic_scholar_url: str = "https://api.semanticscholar.org/graph/v1/paper/search"
    # Optional; without a key requests share a public, heavily rate-limited pool.
    semantic_scholar_api_key: str = ""
    semantic_scholar_min_interval: float = 1.0
    # Largest API response read, in bytes.
    max_response_bytes: int = 2_097_152
    cache_ttl_seconds: int = 900
    cache_max_entries: int = 256
    user_agent: str = DEFAULT_USER_AGENT
    # Papers whose abstract is shorter than this are skipped.
    min_abstract_chars: int = 200


@dataclass(frozen=True)
class ClaimsConfig:
    """Claim extraction: which passages are read and how many claims are kept."""

    # Passages retrieved per sub-question and shown to the extractor.
    passages_per_sub_question: int = 5
    # Passages read across the whole run (one extraction call per sub-question).
    max_passages: int = 24
    max_claims_per_sub_question: int = 8
    max_output_tokens: int = 4096


@dataclass(frozen=True)
class FactCheckConfig:
    """Fact checking: the evidence each claim is checked against."""

    # Passages per claim: the ones it was extracted from, then the most
    # similar passages from other sources (independent corroboration).
    evidence_per_claim: int = 4
    # Retrieved passages below this cosine similarity are not used as evidence.
    min_similarity: float = 0.0
    # The fact checker sees the part of each passage most relevant to the claim.
    excerpt_chars: int = 700
    # Claims per model call.
    batch_size: int = 6
    max_output_tokens: int = 4096


@dataclass(frozen=True)
class ContradictionConfig:
    """Contradiction detection: which claim pairs the model compares."""

    # Claim pairs at least this similar (cosine) are compared. Pairs the fact
    # checker already linked (one claim's passage contradicts the other) are
    # always compared.
    min_similarity: float = 0.7
    # Most claim pairs compared per round, most similar first.
    max_pairs: int = 40
    # Similar claims are grouped by topic; most claims per group.
    max_group_size: int = 6
    # Topic groups per model call.
    groups_per_call: int = 4
    # Supporting excerpts shown per claim.
    excerpts_per_claim: int = 2
    max_output_tokens: int = 4096


@dataclass(frozen=True)
class SufficiencyConfig:
    """When a verified run is complete enough, and when it researches more."""

    # A sub-question is covered with at least this many usable claims ...
    min_claims_per_sub_question: int = 2
    # ... supported by at least this many independent sites.
    min_sites_per_sub_question: int = 2
    # When every sub-question is covered, ask the model whether the claims
    # answer the question (the rules always run first).
    llm_check: bool = True
    # Most gaps the model may name.
    max_gaps: int = 3
    # Claims per sub-question shown to the model.
    claims_shown_per_sub_question: int = 6
    max_output_tokens: int = 2048
    # Seconds into a run after which no new research round starts, per mode
    # (absent or 0: no deadline). Keeps a looping mode near its latency target.
    loop_deadlines: tuple[tuple[str, float], ...] = (
        ("deep", 150.0),
        ("academic", 180.0),
        ("decision", 150.0),
    )

    def loop_deadline(self, mode: str) -> float:
        return float(dict(self.loop_deadlines).get(mode, 0.0))


# Why two claims disagree (contradiction records' likely_cause). Mirrored by
# ``state.ContradictionCause``.
CONTRADICTION_CAUSES: tuple[str, ...] = (
    "conditions",
    "date",
    "version",
    "dataset",
    "methodology",
    "definition",
    "unexplained",
)

# Causes under which both claims can hold once their scope is stated, so the
# contradiction is resolved and does not lower the report's confidence.
DEFAULT_RESOLVED_CAUSES: tuple[str, ...] = ("conditions", "date", "version", "dataset", "definition")


# Base reliability per domain type, highest first (PRD s8: peer-reviewed,
# official docs, standards bodies > reputable publications > company blogs >
# forums/unknown). See reliability.py for how scores are combined.
DEFAULT_TYPE_SCORES: tuple[tuple[str, float], ...] = (
    ("peer_reviewed", 0.90),
    ("standards_body", 0.90),
    ("official_docs", 0.85),
    ("government", 0.85),
    ("academic", 0.80),
    ("preprint", 0.70),
    ("news", 0.70),
    ("reference", 0.65),
    ("code_repository", 0.60),
    ("company_blog", 0.55),
    ("blog", 0.40),
    ("unknown", 0.45),
    ("forum", 0.30),
)

# Claim confidence multiplier per fact-check verdict.
DEFAULT_VERDICT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("supported", 1.0),
    ("weak", 0.6),
    ("mixed", 0.35),
    ("unsupported", 0.0),
    ("unverified", 0.0),
)


@dataclass(frozen=True)
class ScoringConfig:
    """
    Weights for source reliability and claim/report confidence.

    The formulas, and what each weight does, are documented in
    ``researchmind_agents.reliability``. Mappings are tuples of pairs so the
    config stays hashable.
    """

    type_scores: tuple[tuple[str, float], ...] = DEFAULT_TYPE_SCORES
    # Recency: full score up to ``recency_full_years`` old, then minus
    # ``recency_penalty_per_year`` per year, never below ``recency_floor``.
    recency_full_years: float = 3.0
    recency_penalty_per_year: float = 0.05
    recency_floor: float = 0.75
    # Multiplier when a source has no known publication date.
    unknown_date_factor: float = 0.95
    # Papers: up to this bonus for citations (log scale, full at saturation).
    citation_bonus_max: float = 0.10
    citation_saturation: int = 1000
    # Papers whose text describes data or methods.
    methodology_bonus: float = 0.05

    verdict_weights: tuple[tuple[str, float], ...] = DEFAULT_VERDICT_WEIGHTS
    # The most one source of reliability 1.0 establishes on its own;
    # independent sources combine by noisy-OR, so agreement compounds.
    source_weight: float = 0.8
    # Report confidence: penalty at zero sub-question coverage (scaled by the
    # uncovered share), and per unresolved contradiction up to a cap.
    uncovered_penalty: float = 0.5
    contradiction_penalty: float = 0.10
    contradiction_penalty_cap: float = 0.5
    # A contradiction with one of these likely causes is resolved: the claims
    # differ in scope, not in fact. Others count against the report.
    resolved_causes: tuple[str, ...] = DEFAULT_RESOLVED_CAUSES

    def __post_init__(self) -> None:
        unknown = set(self.resolved_causes) - set(CONTRADICTION_CAUSES)
        if unknown:
            raise ValueError(f"unknown contradiction cause(s) {sorted(unknown)} in resolved_causes")
        bounded = [
            *self.type_scores,
            *self.verdict_weights,
            ("source_weight", self.source_weight),
            ("recency_floor", self.recency_floor),
            ("unknown_date_factor", self.unknown_date_factor),
            ("uncovered_penalty", self.uncovered_penalty),
            ("contradiction_penalty_cap", self.contradiction_penalty_cap),
        ]
        for name, value in bounded:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"scoring weight {name!r} must be between 0 and 1, got {value}")

    def type_score(self, domain_type: str) -> float:
        scores = dict(self.type_scores)
        return float(scores.get(domain_type, scores.get("unknown", 0.45)))

    def verdict_weight(self, verdict: str) -> float:
        return float(dict(self.verdict_weights).get(verdict, 0.0))

    def is_resolved(self, cause: str) -> bool:
        return cause in self.resolved_causes


def _merged_pairs(
    defaults: tuple[tuple[str, float], ...], overrides: Any
) -> tuple[tuple[str, float], ...]:
    """Defaults with *overrides* (a mapping, possibly empty) applied on top."""
    merged = dict(defaults)
    for key, value in dict(overrides or {}).items():
        merged[str(key)] = float(value)
    return tuple(merged.items())


def _names(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    """A comma-separated string or a sequence of names; *default* when absent."""
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(part.strip().lower() for part in value.split(",") if part.strip())
    return tuple(str(part).strip().lower() for part in value)


@dataclass(frozen=True)
class AgentConfig:
    """Top-level configuration handed to the graph and its nodes."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    page_extract: PageExtractConfig = field(default_factory=PageExtractConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    synthesizer: SynthesizerConfig = field(default_factory=SynthesizerConfig)
    academic: AcademicSearchConfig = field(default_factory=AcademicSearchConfig)
    claims: ClaimsConfig = field(default_factory=ClaimsConfig)
    fact_check: FactCheckConfig = field(default_factory=FactCheckConfig)
    contradictions: ContradictionConfig = field(default_factory=ContradictionConfig)
    sufficiency: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    concurrency_cap: int = 4
    # Per-node safety net, not a latency target: on free tiers a node may wait
    # out a ~60s rate limit, and a timeout discards that branch's unstored work.
    agent_timeout: int = 240
    # Research rounds in a looping mode (deep, academic, decision): the first
    # plus up to max_iterations - 1 gap-filling rounds. Quick mode runs one.
    max_iterations: int = 3
    max_sources_per_question: int = 10
    # A limit of 0 disables that limit.
    max_tokens_per_run: int = 100_000
    max_cost_per_run_usd: float = 1.00
    # Share of the run budget held back for synthesis: research stops once
    # (1 - reserve) of either limit is spent, so the report still gets written.
    budget_synthesis_reserve: float = 0.2

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
            max_retry_delay=pick("llm_max_retry_delay", LLMConfig.max_retry_delay),
            overload_delay=pick("llm_overload_retry_delay", LLMConfig.overload_delay),
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
            max_retry_delay=pick("llm_max_retry_delay", EmbeddingConfig.max_retry_delay),
            overload_delay=pick("llm_overload_retry_delay", EmbeddingConfig.overload_delay),
            api_key=pick("gemini_api_key", EmbeddingConfig.api_key),
        )

        search = SearchConfig(
            max_results=pick("max_sources_per_question", SearchConfig.max_results),
            timeout=pick("search_timeout", SearchConfig.timeout),
            region=pick("search_region", SearchConfig.region),
            safe_search=pick("search_safe_search", SearchConfig.safe_search),
            backend=pick("search_backend", SearchConfig.backend),
            max_attempts=pick("search_max_attempts", SearchConfig.max_attempts),
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

        planner = PlannerConfig(
            min_sub_questions=pick("planner_min_sub_questions", PlannerConfig.min_sub_questions),
            max_sub_questions=pick("planner_max_sub_questions", PlannerConfig.max_sub_questions),
            max_queries_per_sub_question=pick(
                "planner_max_queries_per_sub_question",
                PlannerConfig.max_queries_per_sub_question,
            ),
            verified_min_sub_questions=pick(
                "planner_verified_min_sub_questions", PlannerConfig.verified_min_sub_questions
            ),
            verified_max_sub_questions=pick(
                "planner_verified_max_sub_questions", PlannerConfig.verified_max_sub_questions
            ),
            max_gap_sub_questions=pick(
                "planner_max_gap_sub_questions", PlannerConfig.max_gap_sub_questions
            ),
        )

        research = ResearchConfig(
            pages_per_sub_question=pick(
                "research_pages_per_sub_question", ResearchConfig.pages_per_sub_question
            ),
            max_fetch_attempts=pick(
                "research_max_fetch_attempts", ResearchConfig.max_fetch_attempts
            ),
            chunk_size_tokens=pick("chunk_size_tokens", ResearchConfig.chunk_size_tokens),
            chunk_overlap_tokens=pick("chunk_overlap_tokens", ResearchConfig.chunk_overlap_tokens),
            max_chunks_per_source=pick(
                "max_chunks_per_source", ResearchConfig.max_chunks_per_source
            ),
            min_source_chars=pick("min_source_chars", ResearchConfig.min_source_chars),
            max_sources_per_run=pick("max_sources_per_run", ResearchConfig.max_sources_per_run),
        )

        synthesizer = SynthesizerConfig(
            top_k_per_sub_question=pick(
                "synth_top_k_per_sub_question", SynthesizerConfig.top_k_per_sub_question
            ),
            max_passages=pick("synth_max_passages", SynthesizerConfig.max_passages),
            max_output_tokens=pick(
                "synth_max_output_tokens", SynthesizerConfig.max_output_tokens
            ),
            min_similarity=pick("synth_min_similarity", SynthesizerConfig.min_similarity),
            max_claims=pick("synth_max_claims", SynthesizerConfig.max_claims),
            min_claim_confidence=pick(
                "synth_min_claim_confidence", SynthesizerConfig.min_claim_confidence
            ),
        )

        providers = pick("academic_search_providers", None)
        if isinstance(providers, str):
            providers = tuple(p.strip().lower() for p in providers.split(",") if p.strip())
        academic = AcademicSearchConfig(
            providers=tuple(providers) if providers is not None else AcademicSearchConfig.providers,
            max_results=pick("academic_max_results", AcademicSearchConfig.max_results),
            timeout=pick("academic_search_timeout", AcademicSearchConfig.timeout),
            max_attempts=pick("search_max_attempts", AcademicSearchConfig.max_attempts),
            arxiv_url=pick("arxiv_api_url", AcademicSearchConfig.arxiv_url),
            arxiv_min_interval=pick(
                "arxiv_min_interval_seconds", AcademicSearchConfig.arxiv_min_interval
            ),
            semantic_scholar_url=pick(
                "semantic_scholar_api_url", AcademicSearchConfig.semantic_scholar_url
            ),
            semantic_scholar_api_key=pick(
                "semantic_scholar_api_key", AcademicSearchConfig.semantic_scholar_api_key
            ),
            semantic_scholar_min_interval=pick(
                "semantic_scholar_min_interval_seconds",
                AcademicSearchConfig.semantic_scholar_min_interval,
            ),
            cache_ttl_seconds=pick("search_cache_ttl_seconds", AcademicSearchConfig.cache_ttl_seconds),
            user_agent=pick("page_user_agent", AcademicSearchConfig.user_agent),
            min_abstract_chars=pick("min_abstract_chars", AcademicSearchConfig.min_abstract_chars),
        )

        claims = ClaimsConfig(
            passages_per_sub_question=pick(
                "claims_passages_per_sub_question", ClaimsConfig.passages_per_sub_question
            ),
            max_passages=pick("claims_max_passages", ClaimsConfig.max_passages),
            max_claims_per_sub_question=pick(
                "claims_max_per_sub_question", ClaimsConfig.max_claims_per_sub_question
            ),
            max_output_tokens=pick("claims_max_output_tokens", ClaimsConfig.max_output_tokens),
        )

        fact_check = FactCheckConfig(
            evidence_per_claim=pick(
                "fact_check_evidence_per_claim", FactCheckConfig.evidence_per_claim
            ),
            min_similarity=pick("fact_check_min_similarity", FactCheckConfig.min_similarity),
            excerpt_chars=pick("fact_check_excerpt_chars", FactCheckConfig.excerpt_chars),
            batch_size=pick("fact_check_batch_size", FactCheckConfig.batch_size),
            max_output_tokens=pick(
                "fact_check_max_output_tokens", FactCheckConfig.max_output_tokens
            ),
        )

        contradictions = ContradictionConfig(
            min_similarity=pick(
                "contradiction_min_similarity", ContradictionConfig.min_similarity
            ),
            max_pairs=pick("contradiction_max_pairs", ContradictionConfig.max_pairs),
            max_group_size=pick(
                "contradiction_max_group_size", ContradictionConfig.max_group_size
            ),
            groups_per_call=pick(
                "contradiction_groups_per_call", ContradictionConfig.groups_per_call
            ),
            excerpts_per_claim=pick(
                "contradiction_excerpts_per_claim", ContradictionConfig.excerpts_per_claim
            ),
            max_output_tokens=pick(
                "contradiction_max_output_tokens", ContradictionConfig.max_output_tokens
            ),
        )

        sufficiency = SufficiencyConfig(
            min_claims_per_sub_question=pick(
                "sufficiency_min_claims_per_sub_question",
                SufficiencyConfig.min_claims_per_sub_question,
            ),
            min_sites_per_sub_question=pick(
                "sufficiency_min_sites_per_sub_question",
                SufficiencyConfig.min_sites_per_sub_question,
            ),
            llm_check=pick("sufficiency_llm_check", SufficiencyConfig.llm_check),
            max_gaps=pick("sufficiency_max_gaps", SufficiencyConfig.max_gaps),
            claims_shown_per_sub_question=pick(
                "sufficiency_claims_shown_per_sub_question",
                SufficiencyConfig.claims_shown_per_sub_question,
            ),
            max_output_tokens=pick(
                "sufficiency_max_output_tokens", SufficiencyConfig.max_output_tokens
            ),
            loop_deadlines=_merged_pairs(
                SufficiencyConfig.loop_deadlines, values.get("loop_deadline_seconds")
            ),
        )

        scoring = ScoringConfig(
            type_scores=_merged_pairs(
                DEFAULT_TYPE_SCORES, values.get("reliability_type_scores")
            ),
            recency_full_years=pick(
                "reliability_recency_full_years", ScoringConfig.recency_full_years
            ),
            recency_penalty_per_year=pick(
                "reliability_recency_penalty_per_year", ScoringConfig.recency_penalty_per_year
            ),
            recency_floor=pick("reliability_recency_floor", ScoringConfig.recency_floor),
            unknown_date_factor=pick(
                "reliability_unknown_date_factor", ScoringConfig.unknown_date_factor
            ),
            citation_bonus_max=pick(
                "reliability_citation_bonus_max", ScoringConfig.citation_bonus_max
            ),
            citation_saturation=pick(
                "reliability_citation_saturation", ScoringConfig.citation_saturation
            ),
            methodology_bonus=pick(
                "reliability_methodology_bonus", ScoringConfig.methodology_bonus
            ),
            verdict_weights=_merged_pairs(
                DEFAULT_VERDICT_WEIGHTS, values.get("confidence_verdict_weights")
            ),
            source_weight=pick("confidence_source_weight", ScoringConfig.source_weight),
            uncovered_penalty=pick("confidence_uncovered_penalty", ScoringConfig.uncovered_penalty),
            contradiction_penalty=pick(
                "confidence_contradiction_penalty", ScoringConfig.contradiction_penalty
            ),
            contradiction_penalty_cap=pick(
                "confidence_contradiction_penalty_cap", ScoringConfig.contradiction_penalty_cap
            ),
            resolved_causes=_names(
                values.get("confidence_resolved_causes"), ScoringConfig.resolved_causes
            ),
        )

        return cls(
            llm=llm,
            embedding=embedding,
            search=search,
            page_extract=page_extract,
            planner=planner,
            research=research,
            synthesizer=synthesizer,
            academic=academic,
            claims=claims,
            fact_check=fact_check,
            contradictions=contradictions,
            sufficiency=sufficiency,
            scoring=scoring,
            concurrency_cap=pick("concurrency_cap", cls.concurrency_cap),
            agent_timeout=pick("agent_timeout", cls.agent_timeout),
            max_iterations=pick("max_iterations", cls.max_iterations),
            max_sources_per_question=pick(
                "max_sources_per_question", cls.max_sources_per_question
            ),
            max_tokens_per_run=pick("max_tokens_per_run", cls.max_tokens_per_run),
            max_cost_per_run_usd=pick("max_cost_per_run_usd", cls.max_cost_per_run_usd),
            budget_synthesis_reserve=pick(
                "budget_synthesis_reserve", cls.budget_synthesis_reserve
            ),
        )
