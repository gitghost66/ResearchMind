"""
M3/M5/M6 tests: the SQL ResearchStore the graph persists through.

Run against the real repositories on SQLite, because the store is the seam
between the packages and a mock would accept any contract drift.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from researchmind_agents.state import (
    ClaimItem,
    ContradictionItem,
    EvidenceItem,
    PassageItem,
    ReportSection,
    ReportStatement,
    ResearchReport,
    SourceItem,
    SubQuestionItem,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.claims import Claim, ClaimEvidence, Contradiction, ContradictionClaim
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
        # Quick-mode reports are unscored: the column gets 0.0, the JSON keeps null.
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


# ===========================================================================
# M5: source scores, claims, and scored reports
# ===========================================================================


class TestSourceScores:
    async def test_scores_and_date_are_stored(self, store, factory, run_id) -> None:
        item = source().model_copy(
            update={
                "domain_type": "official_docs",
                "reliability_score": 0.8075,
                "published_at": "2024-03-01T00:00:00+00:00",
            }
        )
        await store.save_source(run_id=run_id, source=item)

        row = (await all_rows(factory, Source))[0]
        assert row.domain_type == "official_docs"
        assert row.reliability_score == pytest.approx(0.8075)
        assert row.published_at is not None
        assert (row.published_at.year, row.published_at.month) == (2024, 3)

    async def test_unknown_or_bad_date_is_null(self, store, factory, run_id) -> None:
        await store.save_source(run_id=run_id, source=source(url="https://a.example/1"))
        bad = source(url="https://b.example/2", content_hash="hash-b").model_copy(
            update={"published_at": "not a date"}
        )
        await store.save_source(run_id=run_id, source=bad)

        rows = await all_rows(factory, Source)
        assert [r.published_at for r in rows] == [None, None]
        # Defaults when nothing was scored.
        assert {(r.domain_type, r.reliability_score) for r in rows} == {("unknown", 0.5)}


class _ClaimRows:
    """Seeds a sub-question with two passages, and builds claims on them."""

    async def _seed(self, store, run_id):  # type: ignore[no-untyped-def]
        item = sub_question()
        await store.save_plan(run_id=run_id, plan={}, sub_questions=[item])
        first, _ = await store.save_source(run_id=run_id, source=source())
        second, _ = await store.save_source(
            run_id=run_id, source=source(url="https://other.example/x", content_hash="hash-b")
        )
        passages = [
            PassageItem(id=str(uuid.uuid4()), source_id=sid, sub_question_id=item.id, text=f"p{i}")
            for i, sid in enumerate([first, second])
        ]
        await store.save_passages(
            run_id=run_id, passages=passages, embeddings=[None, None], embedding_model="m"
        )
        return item, passages

    def claim(self, item, passages, **update):  # type: ignore[no-untyped-def]
        values = {
            "id": str(uuid.uuid4()),
            "text": "RAG costs less to update than fine-tuning.",
            "sub_question_id": item.id,
            "passage_ids": [passages[0].id],
            "evidence": [
                EvidenceItem(passage_id=passages[0].id, source_id=passages[0].source_id),
                EvidenceItem(
                    passage_id=passages[1].id, source_id=passages[1].source_id, stance="contradicts"
                ),
            ],
            "verdict": "mixed",
            "confidence": 0.21,
        }
        values.update(update)
        return ClaimItem(**values)


class TestClaims(_ClaimRows):
    async def test_claims_and_evidence_links_are_stored(self, store, factory, run_id) -> None:
        item, passages = await self._seed(store, run_id)
        claim = self.claim(item, passages)

        await store.save_claims(run_id=run_id, claims=[claim])

        rows = await all_rows(factory, Claim)
        assert len(rows) == 1
        row = rows[0]
        assert str(row.id) == claim.id
        assert str(row.run_id) == run_id and str(row.sub_question_id) == item.id
        assert (row.verdict, row.confidence) == ("mixed", pytest.approx(0.21))
        links = {(str(link.passage_id), link.stance) for link in await all_rows(factory, ClaimEvidence)}
        assert links == {(passages[0].id, "supports"), (passages[1].id, "contradicts")}

    async def test_a_passage_is_linked_once_per_claim(self, store, factory, run_id) -> None:
        item, passages = await self._seed(store, run_id)
        link = EvidenceItem(passage_id=passages[0].id, source_id=passages[0].source_id)
        claim = self.claim(item, passages, evidence=[link, link], verdict="supported")

        await store.save_claims(run_id=run_id, claims=[claim])

        assert len(await all_rows(factory, ClaimEvidence)) == 1

    async def test_a_claim_without_evidence_is_rejected(self, store, factory, run_id) -> None:
        item, passages = await self._seed(store, run_id)
        good = self.claim(item, passages)
        bad = self.claim(item, passages, evidence=[])

        with pytest.raises(ValueError, match="no evidence"):
            await store.save_claims(run_id=run_id, claims=[good, bad])
        # One transaction: nothing was stored.
        assert await all_rows(factory, Claim) == []

    async def test_unknown_passage_fails_the_whole_batch(self, store, factory, run_id) -> None:
        item, passages = await self._seed(store, run_id)
        ghost = EvidenceItem(passage_id=str(uuid.uuid4()), source_id=passages[0].source_id)
        bad = self.claim(item, passages, evidence=[ghost])

        with pytest.raises(Exception):  # noqa: B017 - the FK violation's type is driver-specific
            await store.save_claims(run_id=run_id, claims=[self.claim(item, passages), bad])
        assert await all_rows(factory, Claim) == []

    async def test_empty_batch_is_a_no_op(self, store, factory, run_id) -> None:
        await store.save_claims(run_id=run_id, claims=[])
        assert await all_rows(factory, Claim) == []

    async def test_scored_report_stores_its_confidence(self, store, factory, run_id) -> None:
        report = ResearchReport(title="t", question="q", mode="deep", overall_confidence=0.62)
        await store.save_report(run_id=run_id, report=report, markdown="# t")

        row = (await all_rows(factory, Report))[0]
        assert row.overall_confidence == pytest.approx(0.62)
        assert row.content["overall_confidence"] == pytest.approx(0.62)


# ===========================================================================
# M6: gap-filling rounds and contradictions
# ===========================================================================


class TestRounds:
    async def test_a_gap_round_adds_its_sub_questions_and_sets_the_iteration(
        self, store, factory, run_id
    ) -> None:
        first = sub_question()
        await store.save_plan(run_id=run_id, plan={"sub_questions": [first.model_dump()]}, sub_questions=[first])
        follow_up = SubQuestionItem(
            id=str(uuid.uuid4()), text="What does RAG hosting cost?", iteration=1, parent_id=first.id
        )
        plan = {"sub_questions": [first.model_dump(), follow_up.model_dump()], "rounds": [{"iteration": 1}]}
        await store.save_plan(run_id=run_id, plan=plan, sub_questions=[follow_up], iteration=1)

        [run] = await all_rows(factory, ResearchRun)
        assert run.iteration == 1
        assert run.plan["rounds"] == [{"iteration": 1}]
        assert [sq["parent_id"] for sq in run.plan["sub_questions"]] == [None, first.id]
        rows = await all_rows(factory, SubQuestion)
        assert {str(r.id) for r in rows} == {first.id, follow_up.id}


class TestContradictions(_ClaimRows):
    async def test_contradictions_and_their_claims_are_stored(self, store, factory, run_id) -> None:
        item, passages = await self._seed(store, run_id)
        a = self.claim(item, passages, verdict="supported")
        b = self.claim(item, passages, text="RAG costs more to update.", passage_ids=[passages[1].id])
        await store.save_claims(run_id=run_id, claims=[a, b])
        contradiction = ContradictionItem(
            id=str(uuid.uuid4()),
            claim_ids=[a.id, b.id],
            explanation="Measured on different datasets.",
            likely_cause="dataset",
            resolved=True,
        )
        await store.save_contradictions(run_id=run_id, contradictions=[contradiction])

        [row] = await all_rows(factory, Contradiction)
        assert str(row.id) == contradiction.id
        assert str(row.run_id) == run_id
        assert row.likely_cause == "dataset"
        assert row.explanation == "Measured on different datasets."
        links = await all_rows(factory, ContradictionClaim)
        assert {str(link.claim_id) for link in links} == {a.id, b.id}

    async def test_a_contradiction_needs_two_claims(self, store, run_id) -> None:
        item, passages = await self._seed(store, run_id)
        a = self.claim(item, passages)
        await store.save_claims(run_id=run_id, claims=[a])
        bad = ContradictionItem(id=str(uuid.uuid4()), claim_ids=[a.id, a.id], explanation="e", likely_cause="date")
        with pytest.raises(ValueError, match="two claims"):
            await store.save_contradictions(run_id=run_id, contradictions=[bad])

    async def test_empty_batch_is_a_no_op(self, store, factory, run_id) -> None:
        await store.save_contradictions(run_id=run_id, contradictions=[])
        assert await all_rows(factory, Contradiction) == []

    async def test_deleting_the_run_removes_them(self, store, factory, run_id) -> None:
        item, passages = await self._seed(store, run_id)
        a = self.claim(item, passages)
        b = self.claim(item, passages, passage_ids=[passages[1].id])
        await store.save_claims(run_id=run_id, claims=[a, b])
        await store.save_contradictions(
            run_id=run_id,
            contradictions=[
                ContradictionItem(id=str(uuid.uuid4()), claim_ids=[a.id, b.id], explanation="e", likely_cause="unexplained")
            ],
        )
        async with factory() as session:
            await session.delete(await session.get(ResearchRun, uuid.UUID(run_id)))
            await session.commit()
        assert await all_rows(factory, Contradiction) == []
        assert await all_rows(factory, ContradictionClaim) == []
