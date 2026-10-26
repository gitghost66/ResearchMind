"""
M2 tests — agent configuration.

AGENTS.md rule 4 forbids hard-coded model names and thresholds, so the mapping
from backend settings into the agents package needs to actually carry every
value it claims to.
"""

from __future__ import annotations

from researchmind_agents.config import (
    AgentConfig,
    EmbeddingConfig,
    LLMConfig,
    PageExtractConfig,
    PlannerConfig,
    ResearchConfig,
    SearchConfig,
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
