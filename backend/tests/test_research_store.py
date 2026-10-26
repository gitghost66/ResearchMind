"""
M3 tests: the SQL ResearchStore the graph persists through.

Run against the real repositories on SQLite, because the store is the seam
between the packages and a mock would accept any contract drift.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from researchmind_agents.state import (
    PassageItem,
    ReportSection,
    ReportStatement,
    ResearchReport,
    SourceItem,
    SubQuestionItem,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.evidence import Passage, Source
from app.models.research import Report, ResearchRun, RunEvent, SubQuestion
from app.services.research_store import SqlResearchStore, make_session_vector_search_fn

DIM = 768


def unit(index: int) -> list[float]:
    """A 768-d one-hot vector; distinct indexes are orthogonal."""
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


@pytest.fixture
def factory(async_engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=async_engine, expire_on_commit=False)


@pytest.fixture
def store(factory) -> SqlResearchStore:
    return SqlResearchStore(factory)


@pytest.fixture
async def run_id(factory) -> str:
    async with factory() as session:
        run = ResearchRun(question="Is RAG better than fine-tuning?")
        session.add(run)
        await session.commit()
        return str(run.id)


def sub_question(text: str = "What does RAG cost?") -> SubQuestionItem:
    return SubQuestionItem(id=str(uuid.uuid4()), text=text, queries=["rag cost"])


def source(url: str = "https://example.com/rag", content_hash: str = "hash-a") -> SourceItem:
    return SourceItem(
        id=str(uuid.uuid4()),
        url=url,
        title="RAG Explained",
        domain="example.com",
        content_hash=content_hash,
    )


async def all_rows(factory, model):  # type: ignore[no-untyped-def]
    async with factory() as session:
        return list((await session.execute(select(model))).scalars().all())


# ===========================================================================
# Plan and sub-questions
# ===========================================================================


class TestPlan:
    async def test_save_plan_writes_run_plan_and_given_ids(self, store, factory, run_id) -> None:
        items = [sub_question("SQ one"), sub_question("SQ two")]
        plan = {"objective": "Compare RAG and fine-tuning", "sub_questions": ["SQ one", "SQ two"]}

        await store.save_plan(run_id=run_id, plan=plan, sub_questions=items)

        runs = await all_rows(factory, ResearchRun)
        assert runs[0].plan == plan

        rows = await all_rows(factory, SubQuestion)
        assert {str(r.id) for r in rows} == {item.id for item in items}
        assert {r.text for r in rows} == {"SQ one", "SQ two"}
        assert all(r.run_id == uuid.UUID(run_id) for r in rows)
        assert all(r.status == "pending" and r.domain == "web" for r in rows)

    async def test_set_sub_question_status(self, store, factory, run_id) -> None:
        item = sub_question()
        await store.save_plan(run_id=run_id, plan={}, sub_questions=[item])

        await store.set_sub_question_status(
            run_id=run_id, sub_question_id=item.id, status="completed"
        )

        rows = await all_rows(factory, SubQuestion)
        assert rows[0].status == "completed"

    async def test_status_update_is_scoped_to_the_run(self, store, factory, run_id) -> None:
        item = sub_question()
        await store.save_plan(run_id=run_id, plan={}, sub_questions=[item])

        await store.set_sub_question_status(
            run_id=str(uuid.uuid4()), sub_question_id=item.id, status="failed"
        )

        rows = await all_rows(factory, SubQuestion)
        assert rows[0].status == "pending"


# ===========================================================================
# Sources
# ===========================================================================


class TestSources:
    async def test_creates_a_source_with_the_given_id(self, store, factory, run_id) -> None:
        item = source()

        source_id, created = await store.save_source(run_id=run_id, source=item)

        assert created is True
        assert source_id == item.id
        rows = await all_rows(factory, Source)
        assert len(rows) == 1
        row = rows[0]
        assert row.url == item.url
        assert row.title == "RAG Explained"
        assert row.content_hash == "hash-a"
        # Scoring is M5: nothing is invented here.
        assert row.domain_type == "unknown"
        assert row.reliability_score == pytest.approx(0.5)
        assert row.published_at is None

    async def test_dedupes_by_url(self, store, factory, run_id) -> None:
        first_id, _ = await store.save_source(run_id=run_id, source=source())

        second_id, created = await store.save_source(
            run_id=run_id, source=source(content_hash="different")
        )

        assert created is False
        assert second_id == first_id
        assert len(await all_rows(factory, Source)) == 1

    async def test_dedupes_by_content_hash(self, store, factory, run_id) -> None:
        first_id, _ = await store.save_source(run_id=run_id, source=source())

        second_id, created = await store.save_source(
            run_id=run_id, source=source(url="https://mirror.example.org/rag")
        )

        assert created is False
        assert second_id == first_id

    async def test_same_url_in_another_run_is_a_new_source(self, store, factory, run_id) -> None:
        async with factory() as session:
            other = ResearchRun(question="Another run")
            session.add(other)
            await session.commit()
            other_id = str(other.id)

        await store.save_source(run_id=run_id, source=source())
        _, created = await store.save_source(run_id=other_id, source=source())

        assert created is True
        assert len(await all_rows(factory, Source)) == 2

    async def test_concurrent_saves_create_exactly_one_row(self, store, factory, run_id) -> None:
        """Parallel graph branches can find the same page at the same time."""
        results = await asyncio.gather(
            *(store.save_source(run_id=run_id, source=source()) for _ in range(5))
        )

        assert sum(created for _, created in results) == 1
        assert len({source_id for source_id, _ in results}) == 1
        assert len(await all_rows(factory, Source)) == 1

    async def test_list_source_ids(self, store, run_id) -> None:
        a, _ = await store.save_source(run_id=run_id, source=source())
        b, _ = await store.save_source(
            run_id=run_id, source=source(url="https://b.example", content_hash="hash-b")
        )

        assert await store.list_source_ids(run_id=run_id) == {a, b}
        assert await store.list_source_ids(run_id=str(uuid.uuid4())) == set()


# ===========================================================================
# Passages and vector search
# ===========================================================================


class TestPassages:
    async def _seed(self, store, run_id):  # type: ignore[no-untyped-def]
        item = sub_question()
        await store.save_plan(run_id=run_id, plan={}, sub_questions=[item])
        source_id, _ = await store.save_source(run_id=run_id, source=source())
        passages = [
            PassageItem(
                id=str(uuid.uuid4()),
                source_id=source_id,
                sub_question_id=item.id,
                text=f"passage {i}",
                chunk_index=i,
            )
            for i in range(3)
        ]
        return item, source_id, passages

    async def test_stores_embeddings_and_model(self, store, factory, run_id) -> None:
        item, source_id, passages = await self._seed(store, run_id)

        await store.save_passages(
            run_id=run_id,
            passages=passages,
            embeddings=[unit(0), unit(1), None],
            embedding_model="gemini-embedding-001",
        )

        rows = sorted(await all_rows(factory, Passage), key=lambda r: r.chunk_index)
        assert [str(r.id) for r in rows] == [p.id for p in passages]
        assert all(r.embedding_model == "gemini-embedding-001" for r in rows)
        assert all(str(r.source_id) == source_id for r in rows)
        assert all(str(r.sub_question_id) == item.id for r in rows)
        assert rows[2].embedding is None

    async def test_length_mismatch_is_rejected(self, store, run_id) -> None:
        _, _, passages = await self._seed(store, run_id)

        with pytest.raises(ValueError, match="same length"):
            await store.save_passages(
                run_id=run_id,
                passages=passages,
                embeddings=[unit(0)],
                embedding_model="m",
            )

    async def test_empty_batch_is_a_no_op(self, store, factory, run_id) -> None:
        await store.save_passages(run_id=run_id, passages=[], embeddings=[], embedding_model="m")
        assert await all_rows(factory, Passage) == []

    async def test_session_vector_search_finds_stored_passages(
        self, store, factory, run_id
    ) -> None:
        _, source_id, passages = await self._seed(store, run_id)
        await store.save_passages(
            run_id=run_id,
            passages=passages,
            embeddings=[unit(0), unit(1), None],
            embedding_model="gemini-embedding-001",
        )

        search = make_session_vector_search_fn(factory)
        query = [0.9, 0.1] + [0.0] * (DIM - 2)
        results = await search(query, top_k=5, run_id=run_id)

        # The passage without an embedding is not searchable.
        assert [r.passage_id for r in results] == [passages[0].id, passages[1].id]
        assert results[0].score > results[1].score
        assert results[0].source_id == source_id
        assert results[0].metadata["source_url"] == "https://example.com/rag"
        assert results[0].metadata["source_title"] == "RAG Explained"

    async def test_session_vector_search_is_scoped_to_the_run(
        self, store, factory, run_id
    ) -> None:
        _, _, passages = await self._seed(store, run_id)
        await store.save_passages(
            run_id=run_id,
            passages=passages,
            embeddings=[unit(0), unit(1), unit(2)],
            embedding_model="m",
        )

        search = make_session_vector_search_fn(factory)
        assert await search(unit(0), top_k=5, run_id=str(uuid.uuid4())) == []


# ===========================================================================
# Report and events
# ===========================================================================


class TestReportAndEvents:
    async def test_save_report(self, store, factory, run_id) -> None:
        report = ResearchReport(
            title="RAG vs fine-tuning",
            question="Is RAG better than fine-tuning?",
            mode="quick",
            sections=[
                ReportSection(
                    heading="Cost",
                    statements=[ReportStatement(text="RAG is cheaper.", citations=["S1"])],
                )
            ],
        )

        report_id = await store.save_report(
            run_id=run_id, report=report, markdown="# RAG vs fine-tuning"
        )

        rows = await all_rows(factory, Report)
        assert len(rows) == 1
        row = rows[0]
        assert str(row.id) == report_id
        assert row.markdown == "# RAG vs fine-tuning"
        assert row.version == 1
        # Unscored until M5: the column gets 0.0, the JSON keeps null.
        assert row.overall_confidence == 0.0
        assert row.content["overall_confidence"] is None
        assert row.content["sections"][0]["statements"][0]["citations"] == ["S1"]

    async def test_log_event(self, store, factory, run_id) -> None:
        await store.log_event(
            run_id=run_id,
            agent="planner",
            event_type="agent_started",
            payload={"question": "Is RAG better?"},
        )

        rows = await all_rows(factory, RunEvent)
        assert len(rows) == 1
        assert rows[0].agent == "planner"
        assert rows[0].type == "agent_started"
        assert rows[0].payload == {"question": "Is RAG better?"}
