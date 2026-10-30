"""
M3 tests — ``execute_research`` end to end through the backend seam.

The graph runs against fakes for the LLM, search, pages, and embeddings, but
persists through the real ``SqlResearchStore`` and repositories on SQLite. This
is where the M3 acceptance criterion is checked: every citation in the stored
report points at a source row stored for that run.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import pytest
from researchmind_agents.config import LLMConfig
from researchmind_agents.llm.base import UsageSink
from researchmind_agents.state import SubQuestionItem
from researchmind_agents.testing import (
    FakePageExtract,
    FakeSearchProvider,
    FakeWeb,
    ScriptedLLM,
    fake_embedding_service,
)
from researchmind_agents.tools.search import SearchTool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models.evidence import Passage, Source
from app.models.research import LLMUsage, Report, ResearchRun, RunEvent, SubQuestion
from app.services.agent_runtime import build_agent_config
from app.services.research_runner import (
    execute_research,
    execute_run,
    settings_for_run,
)
from app.services.research_store import SqlResearchStore

QUESTION = "RAG versus fine-tuning"


def page_text(seed: str) -> str:
    return " ".join(
        f"The {seed} study sentence {k} compares rag benefits with fine-tuning costs and risks."
        for k in range(10)
    )


def make_web() -> FakeWeb:
    return FakeWeb(
        pages={
            "https://a.example/rag": ("RAG benefits", page_text("alpha")),
            "https://b.example/ft": ("Fine-tuning costs", page_text("beta")),
            "https://c.example/compare": ("RAG vs fine-tuning risks", page_text("gamma")),
        }
    )


def settings(**overrides: Any) -> Settings:
    overrides.setdefault("gemini_api_key", "test-key")
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def runtime_kwargs(config_settings: Settings, web: FakeWeb, **extra: Any) -> dict[str, Any]:
    config = build_agent_config(config_settings)

    def llm_factory(llm_config: LLMConfig, sink: UsageSink | None) -> ScriptedLLM:
        return ScriptedLLM(dataclasses.replace(llm_config, base_delay=0.0), sink)

    return {
        "llm_factory": llm_factory,
        "embeddings": fake_embedding_service(config.embedding),
        "search": SearchTool(config.search, provider=FakeSearchProvider(web)),
        "page_extract": FakePageExtract(web),
        **extra,
    }


@pytest.fixture
def factory(async_engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=async_engine, expire_on_commit=False)


async def rows(factory: async_sessionmaker[AsyncSession], model: Any, run_id: str) -> list[Any]:
    async with factory() as session:
        result = await session.execute(select(model).where(model.run_id == uuid.UUID(run_id)))
        return list(result.scalars().all())


class TestPersistedRun:
    async def test_end_to_end(self, factory: async_sessionmaker[AsyncSession]) -> None:
        cfg = settings()
        outcome = await execute_research(
            QUESTION,
            settings=cfg,
            session_factory=factory,
            runtime_kwargs=runtime_kwargs(cfg, make_web()),
        )

        assert outcome.persisted is True
        run_id = outcome.run_id

        async with factory() as session:
            run = await session.get(ResearchRun, uuid.UUID(run_id))
        assert run is not None
        assert run.status == "completed"
        assert run.started_at is not None
        assert run.finished_at is not None
        assert run.error is None
        assert run.plan["objective"]
        assert len(run.plan["sub_questions"]) == 2

        sub_questions = await rows(factory, SubQuestion, run_id)
        assert {str(sq.id) for sq in sub_questions} == {sq["id"] for sq in run.plan["sub_questions"]}
        assert all(sq.status == "completed" for sq in sub_questions)

        sources = await rows(factory, Source, run_id)
        assert sources
        source_ids = {str(s.id) for s in sources}
        assert outcome.stored_source_ids == source_ids

        async with factory() as session:
            passages = list(
                (
                    await session.execute(
                        select(Passage).join(Source).where(Source.run_id == uuid.UUID(run_id))
                    )
                )
                .scalars()
                .all()
            )
        assert passages
        expected_model = build_agent_config(cfg).embedding.model
        assert all(p.embedding_model == expected_model for p in passages)
        assert all(p.embedding is not None for p in passages)

    async def test_report_citations_point_only_at_stored_sources(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """M3 acceptance: citations resolve to sources stored for this run."""
        cfg = settings()
        outcome = await execute_research(
            QUESTION,
            settings=cfg,
            session_factory=factory,
            runtime_kwargs=runtime_kwargs(cfg, make_web()),
        )

        reports = await rows(factory, Report, outcome.run_id)
        assert len(reports) == 1
        report = reports[0]
        source_ids = {str(s.id) for s in await rows(factory, Source, outcome.run_id)}

        citations = report.content["sources"]
        assert citations
        assert {c["source_id"] for c in citations} <= source_ids
        labels = {c["label"] for c in citations}
        statements = [
            *report.content["executive_summary"],
            *(s for section in report.content["sections"] for s in section["statements"]),
        ]
        assert statements
        assert all(set(s["citations"]) <= labels and s["citations"] for s in statements)
        assert report.markdown == outcome.markdown
        assert report.content["overall_confidence"] is None
        assert report.overall_confidence == 0.0
        assert outcome.report["generation"] == "llm"

    async def test_events_and_usage_are_recorded(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        cfg = settings()
        outcome = await execute_research(
            QUESTION,
            settings=cfg,
            session_factory=factory,
            runtime_kwargs=runtime_kwargs(cfg, make_web()),
        )

        events = await rows(factory, RunEvent, outcome.run_id)
        types = {(e.agent, e.type) for e in events}
        assert ("planner", "agent_started") in types
        assert ("web_researcher", "source_found") in types
        assert ("synthesizer", "report_ready") in types

        usage = await rows(factory, LLMUsage, outcome.run_id)
        assert usage
        assert {u.agent for u in usage} >= {"planner", "synthesizer"}
        assert sum(u.tokens_in + u.tokens_out for u in usage) == outcome.usage["tokens_used"]


class _BrokenPlanStore(SqlResearchStore):
    async def save_plan(
        self,
        *,
        run_id: str,
        plan: dict[str, Any],
        sub_questions: list[SubQuestionItem],
        iteration: int = 0,
    ) -> None:
        raise RuntimeError("database unavailable")


class TestFailures:
    async def test_graph_failure_marks_run_failed(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        cfg = settings()
        kwargs = runtime_kwargs(cfg, make_web(), store=_BrokenPlanStore(factory))

        with pytest.raises(RuntimeError, match="database unavailable"):
            await execute_research(
                QUESTION, settings=cfg, session_factory=factory, runtime_kwargs=kwargs
            )

        async with factory() as session:
            runs = list((await session.execute(select(ResearchRun))).scalars().all())
        assert len(runs) == 1
        assert runs[0].status == "failed"
        assert runs[0].error is not None
        assert "database unavailable" in runs[0].error
        assert runs[0].finished_at is not None

    async def test_unsupported_mode_marks_run_failed(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        cfg = settings()

        with pytest.raises(ValueError, match="not available yet"):
            await execute_research(
                QUESTION,
                mode="nonsense",
                settings=cfg,
                session_factory=factory,
                runtime_kwargs=runtime_kwargs(cfg, make_web()),
            )

        async with factory() as session:
            run = (await session.execute(select(ResearchRun))).scalars().one()
        assert run.status == "failed"
        assert run.mode == "nonsense"
        assert "ValueError" in (run.error or "")


class TestMemoryRun:
    async def test_runs_without_a_database(self) -> None:
        cfg = settings()
        outcome = await execute_research(
            QUESTION, settings=cfg, runtime_kwargs=runtime_kwargs(cfg, make_web())
        )

        assert outcome.persisted is False
        uuid.UUID(outcome.run_id)
        assert outcome.mode == cfg.default_mode
        assert outcome.stored_source_ids
        assert {c["source_id"] for c in outcome.report["sources"]} <= outcome.stored_source_ids
        assert outcome.markdown.startswith("# ")
        assert outcome.usage["calls"] >= 2
        assert outcome.seconds >= 0
        assert outcome.state["research_results"]


class TestLifecycle:
    async def test_status_changes_are_events_in_order(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        cfg = settings()
        outcome = await execute_research(
            QUESTION,
            settings=cfg,
            session_factory=factory,
            runtime_kwargs=runtime_kwargs(cfg, make_web()),
        )

        events = sorted(await rows(factory, RunEvent, outcome.run_id), key=lambda e: e.seq)
        assert [e.seq for e in events] == list(range(1, len(events) + 1))
        lifecycle = [e.payload["status"] for e in events if e.type == "run_status"]
        assert lifecycle == ["queued", "running", "completed"]
        assert events[0].type == events[-1].type == "run_status"

    async def test_failure_is_the_final_event(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        cfg = settings()
        kwargs = runtime_kwargs(cfg, make_web(), store=_BrokenPlanStore(factory))
        with pytest.raises(RuntimeError):
            await execute_research(
                QUESTION, settings=cfg, session_factory=factory, runtime_kwargs=kwargs
            )

        async with factory() as session:
            last = (
                await session.execute(select(RunEvent).order_by(RunEvent.seq.desc()).limit(1))
            ).scalar_one()
        assert last.type == "run_status"
        assert last.payload["status"] == "failed"
        assert "database unavailable" in last.payload["error"]


class TestRunOptions:
    def test_max_sources_becomes_the_run_cap(self) -> None:
        base = settings(max_sources_per_run=0)
        assert settings_for_run(base, {"max_sources": 7}).max_sources_per_run == 7
        assert settings_for_run(base, {}) is base
        assert settings_for_run(base, None) is base

    async def test_options_are_stored_and_applied(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        cfg = settings()
        outcome = await execute_research(
            QUESTION,
            settings=cfg,
            session_factory=factory,
            options={"max_sources": 1},
            runtime_kwargs=runtime_kwargs(cfg, make_web()),
        )

        assert len(outcome.stored_source_ids) == 1
        async with factory() as session:
            run = (await session.execute(select(ResearchRun))).scalars().one()
        assert run.options == {"max_sources": 1}


class TestM8:
    async def test_a_script_run_holds_a_lease_and_leaves_no_checkpoints(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        from app.models.checkpoints import GraphCheckpoint

        cfg = settings()
        outcome = await execute_research(
            QUESTION, settings=cfg, session_factory=factory, runtime_kwargs=runtime_kwargs(cfg, make_web())
        )

        async with factory() as session:
            run = (await session.execute(select(ResearchRun))).scalars().one()
            checkpoints = (await session.execute(select(GraphCheckpoint))).scalars().all()
        assert run.status == "completed" and run.attempts == 1
        assert run.worker_id is not None and run.worker_id.startswith("script:")
        assert run.lease_expires_at is None
        assert checkpoints == []
        assert outcome.persisted

    async def test_a_resumed_run_counts_what_earlier_attempts_spent(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        async with factory() as session:
            run = ResearchRun(question=QUESTION, status="running")
            session.add(run)
            await session.flush()
            session.add(
                LLMUsage(run_id=run.id, agent="planner", model="m", tokens_in=900, tokens_out=100)
            )
            await session.commit()
            run_id = str(run.id)
        # A budget the first attempt already spent: research is skipped.
        cfg = settings(max_tokens_per_run=1000)

        outcome = await execute_run(
            run_id,
            question=QUESTION,
            mode="quick",
            settings=cfg,
            session_factory=factory,
            runtime_kwargs=runtime_kwargs(cfg, make_web()),
            resume=True,
        )

        assert outcome.usage["tokens_used"] >= 1000
        assert outcome.usage["calls"] >= 1
        assert outcome.stored_source_ids == set()
