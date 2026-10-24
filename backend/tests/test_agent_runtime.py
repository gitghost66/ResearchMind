"""
M2 tests — the backend/agents seam.

These are the tests that would have caught both M2 integration bugs: the usage
sink that nothing called, and the vector-search adapter whose keyword arguments
did not match the repository. Both are exercised here against the real
repositories on SQLite, not against mocks.
"""

from __future__ import annotations

import uuid

import pytest
from researchmind_agents.llm.base import LLMUsage
from researchmind_agents.llm.client import GeminiProvider
from researchmind_agents.llm.openai_compatible import OpenAICompatibleProvider
from researchmind_agents.tools.vector_search import VectorSearchTool, make_vector_search_fn
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.ddl import CREATE_VECTOR_EXTENSION, ensure_pgvector
from app.models.research import LLMUsage as LLMUsageRow
from app.repositories.evidence_repository import PassageRepository, SourceRepository
from app.repositories.research_repository import (
    LLMUsageRepository,
    ResearchRunRepository,
    SubQuestionRepository,
)
from app.services.agent_runtime import (
    build_agent_config,
    build_agent_runtime,
    make_usage_sink,
    record_usage,
)

# ===========================================================================
# Settings -> AgentConfig
# ===========================================================================


class TestBuildAgentConfig:
    def test_translates_settings(self) -> None:
        settings = Settings(
            llm_provider="gemini",
            llm_model="gemini-3.1-flash-lite",
            gemini_api_key="g-key",
            embedding_dimension=768,
            max_sources_per_question=6,
            concurrency_cap=3,
        )

        config = build_agent_config(settings)

        assert config.llm.provider == "gemini"
        assert config.llm.model == "gemini-3.1-flash-lite"
        assert config.llm.api_key == "g-key"
        assert config.embedding.dimension == 768
        assert config.search.max_results == 6
        assert config.concurrency_cap == 3

    def test_settings_mapping_is_a_plain_dict(self) -> None:
        """The agents package must receive data, not a Settings instance."""
        mapping = Settings().agent_config_mapping()
        assert isinstance(mapping, dict)
        assert "llm_model" in mapping


class TestBuildAgentRuntime:
    def test_builds_gemini_runtime_without_a_database(self) -> None:
        runtime = build_agent_runtime(settings=Settings(gemini_api_key="k"))

        assert isinstance(runtime.llm, GeminiProvider)
        assert runtime.vector_search is None
        assert runtime.embeddings.dimension == 768
        assert runtime.search.config.max_results == runtime.config.search.max_results

    def test_provider_choice_follows_settings(self) -> None:
        runtime = build_agent_runtime(
            settings=Settings(
                llm_provider="openrouter",
                llm_base_url="https://openrouter.ai/api/v1",
                llm_api_key="or-key",
            )
        )
        assert isinstance(runtime.llm, OpenAICompatibleProvider)

    def test_unknown_provider_fails_at_build_time(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_agent_runtime(settings=Settings(llm_provider="nope"))

    async def test_vector_search_is_wired_when_a_session_is_given(
        self, db_session: AsyncSession
    ) -> None:
        runtime = build_agent_runtime(
            settings=Settings(gemini_api_key="k"), session=db_session
        )
        assert isinstance(runtime.vector_search, VectorSearchTool)
        await runtime.aclose()


# ===========================================================================
# Usage recording — M2: "wrapper records usage rows"
# ===========================================================================


class TestUsageRecording:
    async def test_record_usage_writes_a_row(self, db_session: AsyncSession) -> None:
        run = await ResearchRunRepository(db_session).create(question="Is RAG better?")

        await record_usage(
            db_session,
            LLMUsage(
                prompt_tokens=120,
                completion_tokens=45,
                total_tokens=165,
                model="gemini-3.1-flash-lite",
                provider="gemini",
                agent="planner",
                run_id=str(run.id),
                latency_ms=812.6,
                cost_usd=0.000123,
            ),
        )

        rows = (await db_session.execute(select(LLMUsageRow))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.run_id == run.id
        assert row.agent == "planner"
        assert row.model == "gemini-3.1-flash-lite"
        assert row.tokens_in == 120
        assert row.tokens_out == 45
        assert row.cost_usd == pytest.approx(0.000123)
        assert row.latency_ms == 812

    async def test_usage_without_run_id_is_dropped_not_fatal(
        self, db_session: AsyncSession
    ) -> None:
        """An unattributable call must not raise; the FK column is NOT NULL."""
        await record_usage(db_session, LLMUsage(prompt_tokens=5, agent="ad-hoc"))

        rows = (await db_session.execute(select(LLMUsageRow))).scalars().all()
        assert rows == []

    async def test_sink_aggregates_into_the_run_summary(
        self, db_session: AsyncSession, async_engine
    ) -> None:
        """End-to-end: provider sink -> llm_usage -> /usage endpoint payload."""
        run = await ResearchRunRepository(db_session).create(question="Postgres or Mongo?")
        await db_session.commit()

        factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
        sink = make_usage_sink(factory)

        await sink(
            LLMUsage(
                prompt_tokens=100,
                completion_tokens=20,
                model="m",
                agent="planner",
                run_id=str(run.id),
                cost_usd=0.001,
                latency_ms=10,
            )
        )
        await sink(
            LLMUsage(
                prompt_tokens=200,
                completion_tokens=50,
                model="m",
                agent="synthesizer",
                run_id=str(run.id),
                cost_usd=0.002,
                latency_ms=20,
            )
        )

        async with factory() as session:
            summary = await LLMUsageRepository(session).get_summary_by_run(run.id)

        assert summary["total_tokens_in"] == 300
        assert summary["total_tokens_out"] == 70
        assert summary["total_cost_usd"] == pytest.approx(0.003)
        assert set(summary["per_agent"]) == {"planner", "synthesizer"}
        assert summary["per_agent"]["planner"]["tokens_in"] == 100

    async def test_sink_records_usage_from_a_real_provider_call(
        self, db_session: AsyncSession, async_engine
    ) -> None:
        """The wrapper, not the caller, is what writes the row."""
        from unittest.mock import MagicMock, patch

        run = await ResearchRunRepository(db_session).create(question="Q")
        await db_session.commit()

        factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
        runtime = build_agent_runtime(
            settings=Settings(gemini_api_key="k"), session_factory=factory
        )

        part = MagicMock()
        part.text = "an answer"
        candidate = MagicMock()
        candidate.content.parts = [part]
        usage_metadata = MagicMock()
        usage_metadata.prompt_token_count = 11
        usage_metadata.candidates_token_count = 7
        usage_metadata.total_token_count = 18
        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata = usage_metadata

        with patch.object(
            runtime.llm._client.models, "generate_content", return_value=response
        ):
            await runtime.llm.generate("hello", agent="web_research", run_id=str(run.id))

        async with factory() as session:
            rows = (await session.execute(select(LLMUsageRow))).scalars().all()

        assert len(rows) == 1
        assert rows[0].agent == "web_research"
        assert rows[0].tokens_in == 11
        assert rows[0].tokens_out == 7
        await runtime.aclose()


# ===========================================================================
# Vector search adapter against the real repository
# ===========================================================================


class TestVectorSearchAdapter:
    async def test_adapter_works_against_the_real_repository(
        self, db_session: AsyncSession
    ) -> None:
        """
        Regression: the adapter used to call search_similar(embedding=...) and
        read row._similarity_score, neither of which the repository provides.
        A mock repo accepted both, so only a real one catches it.
        """
        run = await ResearchRunRepository(db_session).create(question="RAG vs fine-tuning")
        source = await SourceRepository(db_session).create(
            run_id=run.id,
            url="https://example.com/rag",
            title="RAG Explained",
            reliability_score=0.9,
        )
        passage_repo = PassageRepository(db_session)

        target_vec = [1.0] + [0.0] * 767
        other_vec = [0.0, 1.0] + [0.0] * 766

        near = await passage_repo.create(
            source_id=source.id, text="near passage", embedding=target_vec, chunk_index=0
        )
        await passage_repo.create(
            source_id=source.id, text="far passage", embedding=other_vec, chunk_index=1
        )

        search = make_vector_search_fn(passage_repo)
        results = await search(target_vec, top_k=2, run_id=str(run.id))

        assert len(results) == 2
        assert results[0].passage_id == str(near.id)
        assert results[0].text == "near passage"
        assert results[0].score == pytest.approx(1.0, abs=1e-6)
        assert results[0].score > results[1].score
        assert results[0].metadata["source_url"] == "https://example.com/rag"
        assert results[0].metadata["source_title"] == "RAG Explained"
        assert results[0].metadata["reliability_score"] == pytest.approx(0.9)

    async def test_sub_question_scope_reaches_the_repository(
        self, db_session: AsyncSession
    ) -> None:
        run = await ResearchRunRepository(db_session).create(question="Scoped")
        sq_repo = SubQuestionRepository(db_session)
        sq1 = await sq_repo.create(run_id=run.id, text="SQ1")
        sq2 = await sq_repo.create(run_id=run.id, text="SQ2")
        source = await SourceRepository(db_session).create(
            run_id=run.id, url="https://example.com"
        )
        passage_repo = PassageRepository(db_session)

        vec = [0.5] * 768
        await passage_repo.create(
            source_id=source.id, sub_question_id=sq1.id, text="under SQ1", embedding=vec
        )
        await passage_repo.create(
            source_id=source.id, sub_question_id=sq2.id, text="under SQ2", embedding=vec
        )

        search = make_vector_search_fn(passage_repo)
        results = await search(vec, top_k=5, sub_question_id=str(sq1.id))

        assert [r.text for r in results] == ["under SQ1"]
        assert results[0].metadata["sub_question_id"] == str(sq1.id)

    async def test_tool_wraps_the_adapter(self, db_session: AsyncSession) -> None:
        run = await ResearchRunRepository(db_session).create(question="Q")
        source = await SourceRepository(db_session).create(
            run_id=run.id, url="https://example.com"
        )
        passage_repo = PassageRepository(db_session)
        await passage_repo.create(
            source_id=source.id, text="hit", embedding=[1.0] + [0.0] * 767
        )

        tool = VectorSearchTool(make_vector_search_fn(passage_repo))
        results = await tool.search([1.0] + [0.0] * 767, top_k=3, run_id=str(run.id))

        assert [r.text for r in results] == ["hit"]

    async def test_empty_store_returns_no_results(self, db_session: AsyncSession) -> None:
        passage_repo = PassageRepository(db_session)
        tool = VectorSearchTool(make_vector_search_fn(passage_repo))

        assert await tool.search([0.1] * 768, run_id=str(uuid.uuid4())) == []


# ===========================================================================
# pgvector DDL helper (M1 regression)
# ===========================================================================


class TestEnsurePgvector:
    def test_statement_is_executable_not_a_string(self) -> None:
        """
        Regression: alembic/env.py passed a bare string to Connection.execute,
        which SQLAlchemy 2.x rejects with ObjectNotExecutableError.
        """
        from sqlalchemy.sql.elements import TextClause

        assert isinstance(CREATE_VECTOR_EXTENSION, TextClause)
        assert "CREATE EXTENSION" in str(CREATE_VECTOR_EXTENSION)

    def test_issues_statement_on_postgresql(self) -> None:
        from unittest.mock import MagicMock

        connection = MagicMock()
        connection.dialect.name = "postgresql"

        assert ensure_pgvector(connection) is True
        connection.execute.assert_called_once_with(CREATE_VECTOR_EXTENSION)

    def test_no_op_on_other_dialects(self) -> None:
        from unittest.mock import MagicMock

        connection = MagicMock()
        connection.dialect.name = "sqlite"

        assert ensure_pgvector(connection) is False
        connection.execute.assert_not_called()

    async def test_runs_against_a_live_connection(self, async_engine) -> None:
        """Proves the statement is accepted by Connection.execute, not just typed."""
        async with async_engine.begin() as conn:
            await conn.run_sync(ensure_pgvector)
