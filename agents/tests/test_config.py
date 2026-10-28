"""
M2 tests — agent configuration.

AGENTS.md rule 4 forbids hard-coded model names and thresholds, so the mapping
from backend settings into the agents package needs to actually carry every
value it claims to.
"""

from __future__ import annotations

import pytest

from researchmind_agents.config import (
    CONTRADICTION_CAUSES,
    DEFAULT_RESOLVED_CAUSES,
    DEFAULT_TYPE_SCORES,
    DEFAULT_VERDICT_WEIGHTS,
    AcademicSearchConfig,
    AgentConfig,
    ClaimsConfig,
    ContradictionConfig,
    EmbeddingConfig,
    FactCheckConfig,
    LLMConfig,
    PageExtractConfig,
    PlannerConfig,
    ResearchConfig,
    ScoringConfig,
    SearchConfig,
    SufficiencyConfig,
    SynthesizerConfig,
)


class TestDefaults:
    def test_empty_mapping_yields_defaults(self) -> None:
        config = AgentConfig.from_mapping({})

        assert config.llm.provider == "gemini"
        assert config.embedding.dimension == 768
        assert config.search.max_results == SearchConfig.max_results
        assert config.page_extract.respect_robots is True

    def test_unknown_keys_are_ignored(self) -> None:
        config = AgentConfig.from_mapping({"totally_unrelated": 42})
        assert config.llm.model == LLMConfig.model

    def test_none_values_fall_back_to_defaults(self) -> None:
        """Settings fields left unset must not overwrite defaults with None."""
        config = AgentConfig.from_mapping({"llm_model": None, "embedding_dimension": None})

        assert config.llm.model == LLMConfig.model
        assert config.embedding.dimension == EmbeddingConfig.dimension

    def test_configs_are_immutable(self) -> None:
        import dataclasses

        import pytest

        config = AgentConfig.from_mapping({})
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.llm.model = "other"  # type: ignore[misc]


class TestMapping:
    def test_carries_llm_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "llm_provider": "openrouter",
                "llm_model": "some/free-model:free",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_api_key": "or-key",
                "llm_temperature": 0.7,
                "llm_max_output_tokens": 2048,
                "llm_max_attempts": 5,
                "llm_retry_base_delay": 0.5,
                "llm_structured_retries": 3,
                "llm_price_per_1m_input": 0.1,
                "llm_price_per_1m_output": 0.4,
            }
        )

        assert config.llm.provider == "openrouter"
        assert config.llm.model == "some/free-model:free"
        assert config.llm.base_url == "https://openrouter.ai/api/v1"
        assert config.llm.temperature == 0.7
        assert config.llm.max_output_tokens == 2048
        assert config.llm.max_attempts == 5
        assert config.llm.base_delay == 0.5
        assert config.llm.structured_retries == 3
        assert config.llm.price_per_1m_input == 0.1
        assert config.llm.price_per_1m_output == 0.4

    def test_gemini_provider_uses_gemini_key(self) -> None:
        config = AgentConfig.from_mapping(
            {"llm_provider": "gemini", "gemini_api_key": "g-key", "llm_api_key": "other"}
        )
        assert config.llm.api_key == "g-key"

    def test_non_gemini_provider_prefers_its_own_key(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "llm_provider": "openai_compatible",
                "gemini_api_key": "g-key",
                "llm_api_key": "or-key",
            }
        )
        assert config.llm.api_key == "or-key"

    def test_carries_embedding_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "embedding_model": "some-embedding-model",
                "embedding_dimension": 1536,
                "embedding_batch_size": 16,
                "embedding_max_concurrency": 2,
                "gemini_api_key": "g-key",
            }
        )

        assert config.embedding.model == "some-embedding-model"
        assert config.embedding.dimension == 1536
        assert config.embedding.batch_size == 16
        assert config.embedding.max_concurrency == 2
        assert config.embedding.api_key == "g-key"

    def test_carries_search_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "max_sources_per_question": 7,
                "search_timeout": 12,
                "search_region": "uk-en",
                "search_safe_search": "moderate",
                "search_backend": "duckduckgo,brave",
                "search_max_attempts": 4,
                "search_cache_ttl_seconds": 60,
                "search_cache_max_entries": 32,
                "concurrency_cap": 3,
            }
        )

        assert config.search.max_results == 7
        assert config.search.timeout == 12
        assert config.search.region == "uk-en"
        assert config.search.safe_search == "moderate"
        assert config.search.backend == "duckduckgo,brave"
        assert config.search.max_attempts == 4
        assert config.search.cache_ttl_seconds == 60
        assert config.search.cache_max_entries == 32
        # Parallelism is its own knob, not the result count.
        assert config.search.max_concurrency == 3

    def test_carries_page_extract_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "page_extract_timeout": 9,
                "page_max_size_bytes": 2048,
                "page_max_redirects": 2,
                "page_user_agent": "TestBot/1.0",
                "page_respect_robots": False,
                "page_robots_cache_ttl_seconds": 30,
            }
        )

        assert config.page_extract.timeout == 9
        assert config.page_extract.max_size_bytes == 2048
        assert config.page_extract.max_redirects == 2
        assert config.page_extract.user_agent == "TestBot/1.0"
        assert config.page_extract.respect_robots is False
        assert config.page_extract.robots_cache_ttl_seconds == 30

    def test_carries_run_level_limits(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "concurrency_cap": 8,
                "agent_timeout": 45,
                "max_iterations": 2,
                "max_sources_per_question": 15,
                "max_tokens_per_run": 50_000,
                "max_cost_per_run_usd": 0.25,
            }
        )

        assert config.concurrency_cap == 8
        assert config.agent_timeout == 45
        assert config.max_iterations == 2
        assert config.max_sources_per_question == 15
        assert config.max_tokens_per_run == 50_000
        assert config.max_cost_per_run_usd == 0.25

    def test_agent_timeout_becomes_llm_request_timeout(self) -> None:
        config = AgentConfig.from_mapping({"agent_timeout": 30})
        assert config.llm.request_timeout == 30

    def test_carries_planner_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "planner_min_sub_questions": 3,
                "planner_max_sub_questions": 7,
                "planner_max_queries_per_sub_question": 2,
            }
        )

        assert config.planner.min_sub_questions == 3
        assert config.planner.max_sub_questions == 7
        assert config.planner.max_queries_per_sub_question == 2

    def test_carries_research_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "research_pages_per_sub_question": 6,
                "research_max_fetch_attempts": 12,
                "chunk_size_tokens": 500,
                "chunk_overlap_tokens": 50,
                "max_chunks_per_source": 4,
                "min_source_chars": 250,
            }
        )

        assert config.research.pages_per_sub_question == 6
        assert config.research.max_fetch_attempts == 12
        assert config.research.chunk_size_tokens == 500
        assert config.research.chunk_overlap_tokens == 50
        assert config.research.max_chunks_per_source == 4
        assert config.research.min_source_chars == 250

    def test_carries_synthesizer_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "synth_top_k_per_sub_question": 3,
                "synth_max_passages": 12,
                "synth_max_output_tokens": 2048,
                "synth_min_similarity": 0.35,
            }
        )

        assert config.synthesizer.top_k_per_sub_question == 3
        assert config.synthesizer.max_passages == 12
        assert config.synthesizer.max_output_tokens == 2048
        assert config.synthesizer.min_similarity == 0.35

    def test_carries_budget_reserve(self) -> None:
        config = AgentConfig.from_mapping({"budget_synthesis_reserve": 0.35})
        assert config.budget_synthesis_reserve == 0.35


class TestM3Defaults:
    def test_absent_m3_keys_fall_back_to_defaults(self) -> None:
        config = AgentConfig.from_mapping({})

        assert config.planner == PlannerConfig()
        assert config.research == ResearchConfig()
        assert config.synthesizer == SynthesizerConfig()
        assert config.budget_synthesis_reserve == AgentConfig.budget_synthesis_reserve

    def test_none_m3_values_fall_back_to_defaults(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "planner_max_sub_questions": None,
                "research_max_fetch_attempts": None,
                "synth_max_passages": None,
                "synth_min_similarity": None,
                "budget_synthesis_reserve": None,
            }
        )

        assert config.planner.max_sub_questions == PlannerConfig.max_sub_questions
        assert config.research.max_fetch_attempts == ResearchConfig.max_fetch_attempts
        assert config.synthesizer.max_passages == SynthesizerConfig.max_passages
        assert config.synthesizer.min_similarity == SynthesizerConfig.min_similarity
        assert config.budget_synthesis_reserve == AgentConfig.budget_synthesis_reserve

    def test_quick_mode_defaults_are_sane(self) -> None:
        planner = PlannerConfig()
        research = ResearchConfig()

        assert 1 <= planner.min_sub_questions <= planner.max_sub_questions
        assert research.chunk_overlap_tokens < research.chunk_size_tokens
        assert research.max_fetch_attempts >= research.pages_per_sub_question
        assert 0.0 <= AgentConfig().budget_synthesis_reserve < 1.0

    def test_default_embedding_model_is_not_the_retired_one(self) -> None:
        """text-embedding-004 was shut down on 2026-01-14."""
        assert EmbeddingConfig().model != "text-embedding-004"


class TestSearchDefaults:
    def test_default_region_is_a_real_region_code(self) -> None:
        """'wt-wt' only means something to DuckDuckGo; ddgs's other engines break on it."""
        config = AgentConfig.from_mapping({})
        assert config.search.region != "wt-wt"
        assert config.search.backend == "auto"
        assert config.search.max_attempts >= 1


class TestAllowedContentTypes:
    def test_defaults_are_text_only(self) -> None:
        allowed = PageExtractConfig().allowed_content_types
        assert "text/html" in allowed
        assert "application/pdf" not in allowed


# ===========================================================================
# M5: academic search, claims, fact checking, scoring
# ===========================================================================


class TestM5Defaults:
    def test_absent_m5_keys_fall_back_to_defaults(self) -> None:
        config = AgentConfig.from_mapping({})

        assert config.academic == AcademicSearchConfig()
        assert config.claims == ClaimsConfig()
        assert config.fact_check == FactCheckConfig()
        assert config.scoring == ScoringConfig()
        assert config.synthesizer.max_claims == SynthesizerConfig.max_claims
        assert config.synthesizer.min_claim_confidence == SynthesizerConfig.min_claim_confidence

    def test_none_m5_values_fall_back_to_defaults(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "academic_search_providers": None,
                "claims_max_per_sub_question": None,
                "fact_check_batch_size": None,
                "reliability_type_scores": None,
                "confidence_verdict_weights": None,
                "confidence_source_weight": None,
            }
        )

        assert config.academic.providers == AcademicSearchConfig.providers
        assert config.claims.max_claims_per_sub_question == ClaimsConfig.max_claims_per_sub_question
        assert config.fact_check.batch_size == FactCheckConfig.batch_size
        assert config.scoring.type_scores == DEFAULT_TYPE_SCORES
        assert config.scoring.verdict_weights == DEFAULT_VERDICT_WEIGHTS
        assert config.scoring.source_weight == ScoringConfig.source_weight

    def test_defaults_are_sane(self) -> None:
        claims = ClaimsConfig()
        check = FactCheckConfig()
        academic = AcademicSearchConfig()

        assert claims.passages_per_sub_question <= claims.max_passages
        assert check.evidence_per_claim >= 1
        assert check.batch_size >= 1
        assert check.excerpt_chars > 0
        assert set(academic.providers) == {"arxiv", "semantic_scholar"}
        # The arXiv API asks for three seconds between requests.
        assert academic.arxiv_min_interval >= 3.0
        assert academic.arxiv_url.startswith("https://")
        assert academic.semantic_scholar_url.startswith("https://")

    def test_config_stays_hashable(self) -> None:
        """Mappings are tuples of pairs so the frozen config can be hashed."""
        assert hash(AgentConfig()) == hash(AgentConfig())


class TestM5Mapping:
    def test_academic_providers_from_a_comma_string(self) -> None:
        config = AgentConfig.from_mapping({"academic_search_providers": " arxiv, Semantic_Scholar ,, "})
        assert config.academic.providers == ("arxiv", "semantic_scholar")

    def test_academic_providers_from_a_list(self) -> None:
        config = AgentConfig.from_mapping({"academic_search_providers": ["arxiv"]})
        assert config.academic.providers == ("arxiv",)

    def test_carries_academic_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "academic_max_results": 3,
                "academic_search_timeout": 7.5,
                "arxiv_api_url": "https://arxiv.test/api",
                "arxiv_min_interval_seconds": 5.0,
                "semantic_scholar_api_url": "https://s2.test/search",
                "semantic_scholar_api_key": "s2-key",
                "semantic_scholar_min_interval_seconds": 2.0,
                "min_abstract_chars": 120,
                "search_max_attempts": 4,
                "search_cache_ttl_seconds": 60,
                "page_user_agent": "UA/1",
            }
        )

        academic = config.academic
        assert academic.max_results == 3
        assert academic.timeout == 7.5
        assert academic.arxiv_url == "https://arxiv.test/api"
        assert academic.arxiv_min_interval == 5.0
        assert academic.semantic_scholar_url == "https://s2.test/search"
        assert academic.semantic_scholar_api_key == "s2-key"
        assert academic.semantic_scholar_min_interval == 2.0
        assert academic.min_abstract_chars == 120
        assert academic.max_attempts == 4
        assert academic.cache_ttl_seconds == 60
        assert academic.user_agent == "UA/1"

    def test_carries_claim_and_fact_check_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "claims_passages_per_sub_question": 3,
                "claims_max_passages": 9,
                "claims_max_per_sub_question": 4,
                "claims_max_output_tokens": 1024,
                "fact_check_evidence_per_claim": 2,
                "fact_check_min_similarity": 0.4,
                "fact_check_excerpt_chars": 300,
                "fact_check_batch_size": 3,
                "fact_check_max_output_tokens": 2048,
                "synth_max_claims": 12,
                "synth_min_claim_confidence": 0.2,
            }
        )

        assert config.claims == ClaimsConfig(
            passages_per_sub_question=3,
            max_passages=9,
            max_claims_per_sub_question=4,
            max_output_tokens=1024,
        )
        assert config.fact_check == FactCheckConfig(
            evidence_per_claim=2,
            min_similarity=0.4,
            excerpt_chars=300,
            batch_size=3,
            max_output_tokens=2048,
        )
        assert config.synthesizer.max_claims == 12
        assert config.synthesizer.min_claim_confidence == 0.2

    def test_carries_scalar_scoring_settings(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "reliability_recency_full_years": 1.5,
                "reliability_recency_penalty_per_year": 0.02,
                "reliability_recency_floor": 0.6,
                "reliability_unknown_date_factor": 0.9,
                "reliability_citation_bonus_max": 0.2,
                "reliability_citation_saturation": 50,
                "reliability_methodology_bonus": 0.03,
                "confidence_source_weight": 0.7,
                "confidence_uncovered_penalty": 0.4,
                "confidence_contradiction_penalty": 0.05,
                "confidence_contradiction_penalty_cap": 0.3,
            }
        )

        scoring = config.scoring
        assert scoring.recency_full_years == 1.5
        assert scoring.recency_penalty_per_year == 0.02
        assert scoring.recency_floor == 0.6
        assert scoring.unknown_date_factor == 0.9
        assert scoring.citation_bonus_max == 0.2
        assert scoring.citation_saturation == 50
        assert scoring.methodology_bonus == 0.03
        assert scoring.source_weight == 0.7
        assert scoring.uncovered_penalty == 0.4
        assert scoring.contradiction_penalty == 0.05
        assert scoring.contradiction_penalty_cap == 0.3

    def test_type_score_overrides_merge_over_defaults(self) -> None:
        """Settings parse RELIABILITY_TYPE_SCORES as a JSON object; only named types change."""
        config = AgentConfig.from_mapping({"reliability_type_scores": {"forum": 0.2, "news": "0.75"}})

        scores = dict(config.scoring.type_scores)
        assert scores["forum"] == 0.2
        assert scores["news"] == 0.75
        assert scores["peer_reviewed"] == dict(DEFAULT_TYPE_SCORES)["peer_reviewed"]
        assert set(scores) == set(dict(DEFAULT_TYPE_SCORES))
        assert config.scoring.type_score("forum") == 0.2

    def test_verdict_weight_overrides_merge_over_defaults(self) -> None:
        config = AgentConfig.from_mapping({"confidence_verdict_weights": {"weak": 0.5}})

        assert config.scoring.verdict_weight("weak") == 0.5
        assert config.scoring.verdict_weight("supported") == 1.0
        assert config.scoring.verdict_weight("unverified") == 0.0

    def test_empty_overrides_keep_defaults(self) -> None:
        config = AgentConfig.from_mapping(
            {"reliability_type_scores": {}, "confidence_verdict_weights": {}}
        )
        assert config.scoring == ScoringConfig()


class TestScoringConfig:
    def test_lookups(self) -> None:
        scoring = ScoringConfig()
        assert scoring.type_score("official_docs") == dict(DEFAULT_TYPE_SCORES)["official_docs"]
        # An unknown type scores as "unknown"; an unknown verdict carries no weight.
        assert scoring.type_score("tabloid") == scoring.type_score("unknown")
        assert scoring.verdict_weight("probably") == 0.0

    @pytest.mark.parametrize(
        "overrides",
        [
            {"reliability_type_scores": {"forum": 1.5}},
            {"reliability_type_scores": {"news": -0.1}},
            {"confidence_verdict_weights": {"supported": 2.0}},
            {"confidence_source_weight": 1.2},
            {"reliability_recency_floor": -0.5},
            {"reliability_unknown_date_factor": 1.1},
            {"confidence_uncovered_penalty": 1.5},
            {"confidence_contradiction_penalty_cap": 2.0},
        ],
    )
    def test_out_of_range_weights_fail_at_startup(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            AgentConfig.from_mapping(overrides)

    def test_direct_construction_is_validated_too(self) -> None:
        with pytest.raises(ValueError, match="'forum'"):
            ScoringConfig(type_scores=(("forum", 3.0),))


# ===========================================================================
# M6: planner bounds, contradictions, sufficiency, resolved causes
# ===========================================================================


class TestM6Config:
    def test_defaults(self) -> None:
        config = AgentConfig.from_mapping({})
        assert config.planner.bounds(verified=False) == (2, 5)
        assert config.planner.bounds(verified=True) == (4, 8), "PRD s6.2: 4-8 sub-questions"
        assert config.planner.max_gap_sub_questions == 3
        assert config.contradictions == ContradictionConfig()
        assert config.sufficiency == SufficiencyConfig()
        assert config.scoring.resolved_causes == DEFAULT_RESOLVED_CAUSES
        assert config.max_iterations == 3

    def test_mapping_carries_every_m6_setting(self) -> None:
        config = AgentConfig.from_mapping(
            {
                "planner_verified_min_sub_questions": 3,
                "planner_verified_max_sub_questions": 6,
                "planner_max_gap_sub_questions": 2,
                "contradiction_min_similarity": 0.8,
                "contradiction_max_pairs": 10,
                "contradiction_max_group_size": 4,
                "contradiction_groups_per_call": 2,
                "contradiction_excerpts_per_claim": 1,
                "contradiction_max_output_tokens": 1000,
                "sufficiency_min_claims_per_sub_question": 3,
                "sufficiency_min_sites_per_sub_question": 1,
                "sufficiency_llm_check": False,
                "sufficiency_max_gaps": 5,
                "sufficiency_claims_shown_per_sub_question": 2,
                "sufficiency_max_output_tokens": 500,
                "loop_deadline_seconds": {"deep": 0, "quick": 10},
                "confidence_resolved_causes": "date, VERSION",
                "max_iterations": 4,
            }
        )
        assert config.planner.bounds(verified=True) == (3, 6)
        assert config.planner.max_gap_sub_questions == 2
        assert config.contradictions == ContradictionConfig(
            min_similarity=0.8,
            max_pairs=10,
            max_group_size=4,
            groups_per_call=2,
            excerpts_per_claim=1,
            max_output_tokens=1000,
        )
        suff = config.sufficiency
        assert (suff.min_claims_per_sub_question, suff.min_sites_per_sub_question) == (3, 1)
        assert (suff.llm_check, suff.max_gaps, suff.claims_shown_per_sub_question) == (False, 5, 2)
        assert suff.max_output_tokens == 500
        # Deadlines merge over the defaults; 0 turns one off.
        assert suff.loop_deadline("deep") == 0.0
        assert suff.loop_deadline("academic") == 180.0
        assert suff.loop_deadline("quick") == 10.0
        assert suff.loop_deadline("nonsense") == 0.0
        assert config.scoring.resolved_causes == ("date", "version")
        assert config.max_iterations == 4

    def test_resolved_causes_from_a_list(self) -> None:
        config = AgentConfig.from_mapping({"confidence_resolved_causes": ["conditions"]})
        assert config.scoring.is_resolved("conditions")
        assert not config.scoring.is_resolved("date")

    def test_unknown_resolved_cause_fails_at_startup(self) -> None:
        with pytest.raises(ValueError, match="unknown contradiction cause"):
            AgentConfig.from_mapping({"confidence_resolved_causes": "date,guesswork"})

    def test_default_resolution(self) -> None:
        scoring = ScoringConfig()
        for cause in ("conditions", "date", "version", "dataset", "definition"):
            assert scoring.is_resolved(cause)
        for cause in ("methodology", "unexplained"):
            assert not scoring.is_resolved(cause)

    def test_cause_list_matches_the_state_schema(self) -> None:
        from typing import get_args

        from researchmind_agents.prompts.contradiction_detector import CAUSE_DESCRIPTIONS
        from researchmind_agents.state import ContradictionCause

        assert set(get_args(ContradictionCause)) == set(CONTRADICTION_CAUSES)
        assert set(CAUSE_DESCRIPTIONS) == set(CONTRADICTION_CAUSES)
