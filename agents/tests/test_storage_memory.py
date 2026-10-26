"""M3 tests — the in-memory ResearchStore (and its vector search)."""

from __future__ import annotations

import asyncio

import pytest

from researchmind_agents.state import PassageItem, ResearchReport, SourceItem, SubQuestionItem
from researchmind_agents.storage import InMemoryResearchStore


def source(sid: str, url: str, content_hash: str = "") -> SourceItem:
    return SourceItem(id=sid, url=url, title=f"Title {sid}", content_hash=content_hash)


def passage(pid: str, source_id: str, text: str = "text", sq: str | None = None) -> PassageItem:
    return PassageItem(id=pid, source_id=source_id, sub_question_id=sq, text=text)


class TestPlan:
    async def test_save_plan_and_status(self) -> None:
        store = InMemoryResearchStore()
        items = [SubQuestionItem(id="sq-1", text="A?"), SubQuestionItem(id="sq-2", text="B?")]

        await store.save_plan(run_id="run-1", plan={"objective": "o"}, sub_questions=items)
        await store.set_sub_question_status(run_id="run-1", sub_question_id="sq-1", status="completed")

        assert store.plans["run-1"] == {"objective": "o"}
        assert store.sub_questions["sq-1"]["status"] == "completed"
        assert store.sub_questions["sq-2"]["status"] == "pending"
        assert store.sub_questions["sq-1"]["run_id"] == "run-1"

    async def test_status_ignores_other_runs_and_unknown_ids(self) -> None:
        store = InMemoryResearchStore()
        await store.save_plan(
            run_id="run-1", plan={}, sub_questions=[SubQuestionItem(id="sq-1", text="A?")]
        )
        await store.set_sub_question_status(run_id="run-2", sub_question_id="sq-1", status="failed")
        await store.set_sub_question_status(run_id="run-1", sub_question_id="nope", status="failed")
        assert store.sub_questions["sq-1"]["status"] == "pending"


class TestSources:
    async def test_new_source_is_created(self) -> None:
        store = InMemoryResearchStore()
        assert await store.save_source(run_id="r", source=source("s1", "https://a")) == ("s1", True)

    async def test_dedupes_by_url(self) -> None:
        store = InMemoryResearchStore()
        await store.save_source(run_id="r", source=source("s1", "https://a", "h1"))
        assert await store.save_source(run_id="r", source=source("s2", "https://a", "h2")) == (
            "s1",
            False,
        )
        assert set(store.sources) == {"s1"}

    async def test_dedupes_by_content_hash(self) -> None:
        store = InMemoryResearchStore()
        await store.save_source(run_id="r", source=source("s1", "https://a", "same"))
        assert await store.save_source(run_id="r", source=source("s2", "https://b", "same")) == (
            "s1",
            False,
        )

    async def test_empty_hash_never_matches(self) -> None:
        store = InMemoryResearchStore()
        await store.save_source(run_id="r", source=source("s1", "https://a", ""))
        assert await store.save_source(run_id="r", source=source("s2", "https://b", "")) == (
            "s2",
            True,
        )

    async def test_runs_are_isolated(self) -> None:
        store = InMemoryResearchStore()
        await store.save_source(run_id="r1", source=source("s1", "https://a", "h"))
        assert await store.save_source(run_id="r2", source=source("s2", "https://a", "h")) == (
            "s2",
            True,
        )
        assert await store.list_source_ids(run_id="r1") == {"s1"}
        assert await store.list_source_ids(run_id="r2") == {"s2"}

    async def test_concurrent_saves_create_one_row(self) -> None:
        store = InMemoryResearchStore()
        results = await asyncio.gather(
            *(store.save_source(run_id="r", source=source(f"s{i}", "https://a")) for i in range(5))
        )
        assert sum(created for _, created in results) == 1
        assert len({sid for sid, _ in results}) == 1
        assert len(store.sources) == 1


class TestPassages:
    async def test_saves_with_embedding_and_model(self) -> None:
        store = InMemoryResearchStore()
        await store.save_source(run_id="r", source=source("s1", "https://a"))
        await store.save_passages(
            run_id="r",
            passages=[passage("p1", "s1"), passage("p2", "s1")],
            embeddings=[[1.0, 0.0], None],
            embedding_model="model-x",
        )

        assert store.passages["p1"]["embedding"] == [1.0, 0.0]
        assert store.passages["p2"]["embedding"] is None
        assert store.passages["p1"]["embedding_model"] == "model-x"

    async def test_length_mismatch_raises(self) -> None:
        store = InMemoryResearchStore()
        await store.save_source(run_id="r", source=source("s1", "https://a"))
        with pytest.raises(ValueError, match="same length"):
            await store.save_passages(
                run_id="r", passages=[passage("p1", "s1")], embeddings=[], embedding_model="m"
            )

    async def test_unknown_source_raises(self) -> None:
        store = InMemoryResearchStore()
        with pytest.raises(ValueError, match="unknown source_id"):
            await store.save_passages(
                run_id="r", passages=[passage("p1", "ghost")], embeddings=[None], embedding_model="m"
            )


class TestVectorSearch:
    async def make_store(self) -> InMemoryResearchStore:
        store = InMemoryResearchStore()
        await store.save_source(run_id="r1", source=source("s1", "https://a.example/1"))
        await store.save_source(run_id="r2", source=source("s2", "https://b.example/2"))
        await store.save_passages(
            run_id="r1",
            passages=[
                passage("near", "s1", "near text", sq="sq-1"),
                passage("mid", "s1", "mid text", sq="sq-2"),
                passage("far", "s1", "far text", sq="sq-1"),
                passage("none", "s1", "unembedded"),
            ],
            embeddings=[[1.0, 0.0], [0.7, 0.7], [0.0, 1.0], None],
            embedding_model="m",
        )
        await store.save_passages(
            run_id="r2",
            passages=[passage("other-run", "s2", "other")],
            embeddings=[[1.0, 0.0]],
            embedding_model="m",
        )
        return store

    async def test_ranks_by_similarity(self) -> None:
        store = await self.make_store()
        hits = await store.vector_search([1.0, 0.0], top_k=10, run_id="r1")

        assert [h.passage_id for h in hits] == ["near", "mid", "far"]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].score > hits[1].score > hits[2].score

    async def test_top_k(self) -> None:
        store = await self.make_store()
        hits = await store.vector_search([1.0, 0.0], top_k=1, run_id="r1")
        assert [h.passage_id for h in hits] == ["near"]

    async def test_run_scoping(self) -> None:
        store = await self.make_store()
        r2 = await store.vector_search([1.0, 0.0], top_k=10, run_id="r2")
        assert [h.passage_id for h in r2] == ["other-run"]

        unscoped = await store.vector_search([1.0, 0.0], top_k=10)
        assert {h.passage_id for h in unscoped} == {"near", "mid", "far", "other-run"}

    async def test_sub_question_scoping(self) -> None:
        store = await self.make_store()
        hits = await store.vector_search([1.0, 0.0], top_k=10, run_id="r1", sub_question_id="sq-1")
        assert [h.passage_id for h in hits] == ["near", "far"]

    async def test_skips_passages_without_embeddings(self) -> None:
        store = await self.make_store()
        hits = await store.vector_search([0.0, 1.0], top_k=10, run_id="r1")
        assert "none" not in {h.passage_id for h in hits}

    async def test_metadata(self) -> None:
        store = await self.make_store()
        hit = (await store.vector_search([1.0, 0.0], top_k=1, run_id="r1"))[0]

        assert hit.source_id == "s1"
        assert hit.text == "near text"
        assert hit.metadata["source_url"] == "https://a.example/1"
        assert hit.metadata["source_title"] == "Title s1"
        assert hit.metadata["sub_question_id"] == "sq-1"
        assert hit.metadata["chunk_index"] == 0

    async def test_zero_query_vector_scores_zero(self) -> None:
        store = await self.make_store()
        hits = await store.vector_search([0.0, 0.0], top_k=10, run_id="r1")
        assert all(h.score == 0.0 for h in hits)


class TestReportsEventsIds:
    async def test_save_report(self) -> None:
        store = InMemoryResearchStore()
        report = ResearchReport(title="T", question="Q", mode="quick")

        report_id = await store.save_report(run_id="r", report=report, markdown="# T")

        row = store.reports[report_id]
        assert row["run_id"] == "r"
        assert row["markdown"] == "# T"
        assert row["content"]["title"] == "T"
        assert row["content"]["overall_confidence"] is None

    async def test_report_ids_are_unique(self) -> None:
        store = InMemoryResearchStore()
        report = ResearchReport(title="T", question="Q", mode="quick")
        first = await store.save_report(run_id="r", report=report, markdown="")
        second = await store.save_report(run_id="r", report=report, markdown="")
        assert first != second

    async def test_log_event(self) -> None:
        store = InMemoryResearchStore()
        payload = {"k": 1}
        await store.log_event(run_id="r", agent="planner", event_type="agent_started", payload=payload)
        payload["k"] = 2

        assert store.events == [
            {"run_id": "r", "agent": "planner", "type": "agent_started", "payload": {"k": 1}}
        ]

    async def test_list_source_ids_empty(self) -> None:
        assert await InMemoryResearchStore().list_source_ids(run_id="r") == set()
