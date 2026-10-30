"""
M3 tests — web researcher node.

Driven against the fakes in ``researchmind_agents.testing``: a FakeWeb corpus
for search and fetches, the real EmbeddingService over a fake client, a
scripted LLM, and the in-memory store.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from researchmind_agents.config import AgentConfig, ResearchConfig
from researchmind_agents.nodes.web_researcher import (
    MARKET_PROFILE,
    TECHNICAL_PROFILE,
    Candidate,
    make_market_researcher_node,
    make_technical_researcher_node,
    make_web_researcher_node,
    merge_candidates,
    prefer_types,
)
from researchmind_agents.prompts.web_researcher import PROMPT_VERSION
from researchmind_agents.reliability import SourceSignals, source_reliability
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import PassageItem, SubQuestionItem, WebResearchTask
from researchmind_agents.storage import InMemoryResearchStore
from researchmind_agents.testing import (
    FakePageExtract,
    FakeWeb,
    ScriptedLLM,
    build_fake_runtime,
)
from researchmind_agents.tools.page_extract import PageContent
from researchmind_agents.tools.search import SearchResult, SearchTool
from researchmind_agents.tools.urls import normalize_url

RUN_ID = "run-1"


def page_text(seed: str) -> str:
    """Comfortably above the default min_source_chars (400), unique per seed."""
    return " ".join(
        f"The {seed} report sentence {k} explains retrieval augmented generation." for k in range(12)
    )


def url(i: int) -> str:
    return f"https://site{i}.example/page"


def make_web(count: int, **kwargs: Any) -> FakeWeb:
    """Pages that all match the query "rag" equally, so search order is by URL."""
    return FakeWeb(
        pages={url(i): (f"rag page {i}", page_text(f"unique{i}")) for i in range(count)}, **kwargs
    )


def config(**research: Any) -> AgentConfig:
    return AgentConfig(research=ResearchConfig(**research))


def sub_question() -> SubQuestionItem:
    return SubQuestionItem(id="sq-1", text="What is RAG?", queries=["rag"])


async def run_node(runtime: AgentRuntime, sq: SubQuestionItem | None = None) -> dict[str, Any]:
    sq = sq or sub_question()
    await runtime.store.save_plan(run_id=RUN_ID, plan={}, sub_questions=[sq])
    node = make_web_researcher_node(runtime)
    return await node(WebResearchTask(run_id=RUN_ID, question="Q", sub_question=sq.model_dump()))


def store_of(runtime: AgentRuntime) -> InMemoryResearchStore:
    assert isinstance(runtime.store, InMemoryResearchStore)
    return runtime.store


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    assert isinstance(runtime.llm, ScriptedLLM)
    return runtime.llm


def result_of(update: dict[str, Any]) -> dict[str, Any]:
    assert len(update["research_results"]) == 1
    return update["research_results"][0]


def event_types(update: dict[str, Any]) -> list[str]:
    return [e["type"] for e in update["events"]]


# ===========================================================================
# Happy path
# ===========================================================================


class TestStoresEvidence:
    async def test_sources_and_passages_are_stored(self) -> None:
        web = make_web(2)
        runtime = build_fake_runtime(web=web)

        update = await run_node(runtime)

        store = store_of(runtime)
        result = result_of(update)
        assert result["status"] == "completed"
        assert result["error"] is None
        assert result["sources"] == 2
        assert len(store.sources) == 2
        assert {s["url"] for s in store.sources.values()} == {url(0), url(1)}
        assert len(update["sources"]) == 2
        assert {s["id"] for s in update["sources"]} == set(store.sources)

        assert store.passages
        assert {p["id"] for p in update["passages"]} == set(store.passages)
        for row in store.passages.values():
            assert row["sub_question_id"] == "sq-1"
            assert row["embedding_model"] == runtime.embeddings.model
            assert len(row["embedding"]) == runtime.embeddings.dimension

        assert store.sub_questions["sq-1"]["status"] == "completed"

    async def test_events(self) -> None:
        runtime = build_fake_runtime(web=make_web(2))

        update = await run_node(runtime)

        types = event_types(update)
        assert types[0] == "agent_started"
        assert "search_completed" in types
        assert types.count("source_found") == 2
        assert types[-1] == "agent_finished"
        assert [e["type"] for e in store_of(runtime).events] == types
        found = [e for e in update["events"] if e["type"] == "source_found"]
        assert {e["payload"]["url"] for e in found} == {url(0), url(1)}
        assert all(e["payload"]["passages"] >= 1 for e in found)

    async def test_source_records_carry_metadata(self) -> None:
        runtime = build_fake_runtime(web=make_web(1))
        update = await run_node(runtime)

        source = update["sources"][0]
        assert source["title"] == "rag page 0"
        assert source["domain"] == "site0.example"
        assert source["sub_question_id"] == "sq-1"
        assert len(source["content_hash"]) == 64

    async def test_long_pages_are_chunked(self) -> None:
        web = FakeWeb(pages={url(0): ("rag", page_text("long") * 10)})
        runtime = build_fake_runtime(
            config(chunk_size_tokens=100, chunk_overlap_tokens=10, max_chunks_per_source=3), web=web
        )

        update = await run_node(runtime)

        assert len(update["passages"]) == 3
        assert [p["chunk_index"] for p in update["passages"]] == [0, 1, 2]

    async def test_embeds_in_one_batched_call(self) -> None:
        runtime = build_fake_runtime(web=make_web(3))
        await run_node(runtime)
        calls = runtime.embeddings._client.models.calls
        assert len(calls) == 1
        assert calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"


# ===========================================================================
# Choosing what to read
# ===========================================================================


class TestSelection:
    async def test_no_llm_when_candidates_fit(self) -> None:
        runtime = build_fake_runtime(config(pages_per_sub_question=4), web=make_web(3))
        update = await run_node(runtime)

        assert llm_of(runtime).calls_for("SourceSelection") == []
        assert update["llm_calls"] == []

    async def test_llm_picks_are_fetched_first(self) -> None:
        web = make_web(5)
        runtime = build_fake_runtime(
            config(pages_per_sub_question=2),
            web=web,
            responses={"SourceSelection": '{"selected": ["R4", "[r2]"]}'},
        )

        update = await run_node(runtime)

        # R1..R5 follow search order: site0..site4.
        assert web.fetched == [url(3), url(1)]
        assert result_of(update)["sources"] == 2
        assert len(update["llm_calls"]) == 1
        assert update["llm_calls"][0]["prompt_version"] == PROMPT_VERSION
        prompt = llm_of(runtime).calls_for("SourceSelection")[0]["prompt"]
        assert "[R5]" in prompt
        assert "https://" not in prompt

    async def test_picks_beyond_the_page_count_are_ignored(self) -> None:
        web = make_web(5)
        runtime = build_fake_runtime(
            config(pages_per_sub_question=1),
            web=web,
            responses={"SourceSelection": '{"selected": ["R3", "R5", "R1"]}'},
        )
        await run_node(runtime)
        assert web.fetched == [url(2)]

    async def test_invalid_labels_fall_back_to_search_order(self) -> None:
        web = make_web(4)
        runtime = build_fake_runtime(
            config(pages_per_sub_question=2),
            web=web,
            responses={"SourceSelection": '{"selected": ["R99", "bogus"]}'},
        )

        update = await run_node(runtime)

        assert web.fetched == [url(0), url(1)]
        assert result_of(update)["status"] == "completed"

    async def test_selection_failure_falls_back_to_search_order(self) -> None:
        web = make_web(4)
        runtime = build_fake_runtime(
            config(pages_per_sub_question=2),
            web=web,
            responses={"SourceSelection": RuntimeError("model down")},
        )

        update = await run_node(runtime)

        assert web.fetched == [url(0), url(1)]
        assert result_of(update)["status"] == "completed"
        errors = [e for e in update["events"] if e["type"] == "error"]
        assert errors and "model down" in errors[0]["payload"]["reason"]
        assert update["llm_calls"] == []

    async def test_no_llm_when_research_budget_spent_mid_branch(self) -> None:
        """_choose re-checks the budget; with it spent, search order is used."""
        web = make_web(4)
        runtime = build_fake_runtime(config(pages_per_sub_question=2), web=web)

        original = runtime.search.search_multi

        async def spend_then_search(queries: list[str], **kwargs: Any) -> Any:
            runtime.budget.tokens_used = runtime.budget.max_tokens
            return await original(queries, **kwargs)

        runtime.search.search_multi = spend_then_search  # type: ignore[method-assign]
        await run_node(runtime)

        assert llm_of(runtime).calls_for("SourceSelection") == []
        assert web.fetched == [url(0), url(1)]


# ===========================================================================
# Fetching
# ===========================================================================


class TestFetching:
    async def test_broken_pages_are_replaced_up_to_the_attempt_cap(self) -> None:
        web = make_web(4, broken={url(0), url(1)})
        runtime = build_fake_runtime(
            config(pages_per_sub_question=2, max_fetch_attempts=3), web=web
        )

        update = await run_node(runtime)

        result = result_of(update)
        assert web.fetched == [url(0), url(1), url(2)]
        assert result["fetched"] == 3
        assert result["fetch_failures"] == 2
        assert result["sources"] == 1
        assert result["status"] == "completed"

    async def test_backups_fill_the_page_count(self) -> None:
        web = make_web(5, broken={url(0)})
        runtime = build_fake_runtime(
            config(pages_per_sub_question=2, max_fetch_attempts=8), web=web
        )

        update = await run_node(runtime)

        assert web.fetched == [url(0), url(1), url(2)]
        assert result_of(update)["sources"] == 2

    async def test_short_pages_are_skipped(self) -> None:
        web = FakeWeb(pages={url(0): ("rag", "Too short to be useful."), url(1): ("rag", page_text("x"))})
        runtime = build_fake_runtime(web=web)

        update = await run_node(runtime)

        result = result_of(update)
        assert result["fetch_failures"] == 1
        assert [s["url"] for s in update["sources"]] == [url(1)]

    async def test_only_short_pages_fails_the_sub_question(self) -> None:
        web = FakeWeb(pages={url(0): ("rag", "Short.")})
        runtime = build_fake_runtime(web=web)

        update = await run_node(runtime)

        result = result_of(update)
        assert result["status"] == "failed"
        assert result["error"] == "no usable sources"
        assert store_of(runtime).sub_questions["sq-1"]["status"] == "failed"


# ===========================================================================
# De-duplication
# ===========================================================================


class TestDeduplication:
    async def test_claimed_url_is_not_fetched(self) -> None:
        web = make_web(2)
        runtime = build_fake_runtime(web=web)
        runtime.claimed_urls.add(normalize_url(url(0)))

        update = await run_node(runtime)

        assert web.fetched == [url(1)]
        result = result_of(update)
        assert result["duplicates"] == 1
        assert result["sources"] == 1

    async def test_fetched_urls_are_claimed(self) -> None:
        runtime = build_fake_runtime(web=make_web(2))
        await run_node(runtime)
        assert {normalize_url(url(0)), normalize_url(url(1))} <= runtime.claimed_urls

    async def test_only_duplicates_still_completes(self) -> None:
        web = make_web(1)
        runtime = build_fake_runtime(web=web)
        runtime.claimed_urls.add(normalize_url(url(0)))

        update = await run_node(runtime)

        result = result_of(update)
        assert web.fetched == []
        assert result["status"] == "completed"
        assert result["error"] is None
        assert result["sources"] == 0
        assert result["duplicates"] == 1

    async def test_same_content_under_another_url_is_stored_once(self) -> None:
        text = page_text("mirror")
        web = FakeWeb(pages={url(0): ("rag a", text), url(1): ("rag b", text)})
        runtime = build_fake_runtime(web=web)

        update = await run_node(runtime)

        store = store_of(runtime)
        result = result_of(update)
        assert len(store.sources) == 1
        assert result["duplicates"] == 1
        assert result["sources"] == 1
        source_ids = {row["source_id"] for row in store.passages.values()}
        assert len(source_ids) == 1
        assert len(store.passages) == len(update["passages"])


    async def test_a_branch_re_run_after_a_resume_reuses_its_own_sources(self) -> None:
        """M8: the branch's sources are stored but its state was lost; keep them."""
        store = InMemoryResearchStore()
        first = await run_node(build_fake_runtime(web=make_web(2), store=store))

        again = await run_node(build_fake_runtime(web=make_web(2), store=store))

        result = result_of(again)
        assert result["sources"] == result_of(first)["sources"] == 2
        assert result["duplicates"] == 0
        assert [p["id"] for p in again["passages"]] == [p["id"] for p in first["passages"]]
        assert len(store.sources) == 2

    async def test_a_source_another_branch_stored_is_still_a_duplicate(self) -> None:
        store = InMemoryResearchStore()
        await run_node(build_fake_runtime(web=make_web(2), store=store))
        other = SubQuestionItem(id="sq-2", text="What does RAG cost?", queries=["rag"])

        update = await run_node(build_fake_runtime(web=make_web(2), store=store), other)

        result = result_of(update)
        assert (result["sources"], result["duplicates"]) == (0, 2)
        assert update["passages"] == []


# ===========================================================================
# Degradation
# ===========================================================================


class _RaisingProvider:
    @property
    def name(self) -> str:
        return "raising"

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        raise RuntimeError("search endpoint down")


class _FailingPassagesStore(InMemoryResearchStore):
    def __init__(self, bad_url: str) -> None:
        super().__init__()
        self._bad_url = bad_url

    async def save_passages(
        self,
        *,
        run_id: str,
        passages: list[PassageItem],
        embeddings: list[list[float] | None],
        embedding_model: str,
    ) -> None:
        if passages and self.sources[passages[0].source_id]["url"] == self._bad_url:
            raise RuntimeError("disk full")
        await super().save_passages(
            run_id=run_id, passages=passages, embeddings=embeddings, embedding_model=embedding_model
        )


class _SlowPageExtract(FakePageExtract):
    async def extract(self, url: str) -> PageContent:
        await asyncio.sleep(1)
        return await super().extract(url)


class TestDegradation:
    async def test_no_search_results(self) -> None:
        web = FakeWeb()
        runtime = build_fake_runtime(web=web)

        update = await run_node(runtime)

        result = result_of(update)
        assert result["status"] == "failed"
        assert result["error"] == "no usable sources"
        assert result["candidates"] == 0
        search = next(e for e in update["events"] if e["type"] == "search_completed")
        assert search["payload"]["candidates"] == 0
        assert web.fetched == []

    async def test_research_budget_exhausted_skips(self) -> None:
        web = make_web(2)
        runtime = build_fake_runtime(web=web)
        runtime.budget.tokens_used = runtime.budget.max_tokens

        update = await run_node(runtime)

        result = result_of(update)
        assert result["status"] == "skipped"
        assert result["error"] == "research budget exhausted"
        assert web.queries == []
        assert web.fetched == []
        assert store_of(runtime).sub_questions["sq-1"]["status"] == "skipped"
        assert event_types(update) == ["agent_started", "agent_finished"]

    async def test_search_provider_error_degrades(self) -> None:
        runtime = build_fake_runtime(web=make_web(2))
        runtime.search = SearchTool(runtime.config.search, provider=_RaisingProvider())

        update = await run_node(runtime)

        result = result_of(update)
        assert result["status"] == "failed"
        assert result["candidates"] == 0
        assert update["sources"] == []

    async def test_failed_passage_write_skips_only_that_source(self) -> None:
        web = make_web(2)
        runtime = build_fake_runtime(web=web, store=_FailingPassagesStore(bad_url=url(0)))

        update = await run_node(runtime)

        assert [s["url"] for s in update["sources"]] == [url(1)]
        assert result_of(update)["status"] == "completed"
        errors = [e for e in update["events"] if e["type"] == "error"]
        assert any("Failed to store passages" in e["payload"]["message"] for e in errors)
        stored_passage_sources = {p["source_id"] for p in update["passages"]}
        assert stored_passage_sources == {update["sources"][0]["id"]}

    async def test_timeout_marks_the_sub_question_failed(self) -> None:
        web = make_web(1)
        runtime = build_fake_runtime(
            AgentConfig(agent_timeout=0.05),  # type: ignore[arg-type]
            web=web,
        )
        runtime.page_extract = _SlowPageExtract(web)

        update = await run_node(runtime)

        result = result_of(update)
        assert result["status"] == "failed"
        assert "timed out" in result["error"]
        assert "error" in event_types(update)
        assert store_of(runtime).sub_questions["sq-1"]["status"] == "failed"

    async def test_unexpected_error_is_contained(self) -> None:
        runtime = build_fake_runtime(web=make_web(2))

        async def explode(texts: list[str]) -> Any:
            raise RuntimeError("embedding service exploded")

        runtime.embeddings.embed_documents = explode  # type: ignore[method-assign]

        update = await run_node(runtime)

        result = result_of(update)
        assert result["status"] == "failed"
        assert "exploded" in result["error"]
        assert event_types(update)[-1] == "agent_finished"


# ===========================================================================
# merge_candidates
# ===========================================================================


def sr(link: str, title: str = "t", snippet: str = "s") -> SearchResult:
    return SearchResult(title=title, url=link, snippet=snippet)


class TestMergeCandidates:
    def test_round_robin_dedupe_and_labels(self) -> None:
        results = {
            "a": [sr("https://a.example/1"), sr("https://a.example/2"), sr("https://a.example/3")],
            "b": [
                sr("https://A.example/1/#frag"),  # same page as a.example/1
                sr("https://b.example/4"),
                sr("ftp://b.example/file"),
            ],
        }

        merged = merge_candidates(["a", "b"], results)

        assert [c.url for c in merged] == [
            "https://a.example/1",
            "https://a.example/2",
            "https://b.example/4",
            "https://a.example/3",
        ]
        assert [c.label for c in merged] == ["R1", "R2", "R3", "R4"]

    def test_cleans_fields(self) -> None:
        merged = merge_candidates(
            ["q"],
            {"q": [sr(" https://www.x.example/a ", title="  Big \n Title ", snippet="x " * 400)]},
        )
        candidate = merged[0]
        assert candidate.url == "https://www.x.example/a"
        assert candidate.title == "Big Title"
        assert candidate.domain == "x.example"
        assert len(candidate.snippet) <= 300

    def test_missing_query_and_empty_input(self) -> None:
        assert merge_candidates(["missing"], {}) == []
        assert merge_candidates([], {}) == []


# ===========================================================================
# Run-wide source cap (the API's per-run max_sources)
# ===========================================================================


class TestSourceCap:
    async def test_cap_limits_stored_sources(self) -> None:
        runtime = build_fake_runtime(config(max_sources_per_run=2), web=make_web(5))

        update = await run_node(runtime)

        assert len(store_of(runtime).sources) == 2
        assert runtime.source_slots == 2
        assert result_of(update)["status"] == "completed"
        # Only as many pages as the cap allows are read.
        assert result_of(update)["fetched"] == 2

    async def test_branch_after_the_cap_is_skipped(self) -> None:
        runtime = build_fake_runtime(config(max_sources_per_run=1), web=make_web(3))
        runtime.source_slots = 1  # another branch already took the only slot

        update = await run_node(runtime)

        result = result_of(update)
        assert result["status"] == "skipped"
        assert result["error"] == "run source limit reached"
        assert result["fetched"] == 0
        assert store_of(runtime).sources == {}
        assert "source_limit_reached" in event_types(update)
        assert store_of(runtime).sub_questions["sq-1"]["status"] == "skipped"

    async def test_parallel_branches_share_the_cap(self) -> None:
        runtime = build_fake_runtime(config(max_sources_per_run=3), web=make_web(8))
        sqs = [SubQuestionItem(id=f"sq-{i}", text=f"Q{i}", queries=["rag"]) for i in range(3)]
        await runtime.store.save_plan(run_id=RUN_ID, plan={}, sub_questions=sqs)
        node = make_web_researcher_node(runtime)

        await asyncio.gather(
            *(
                node(WebResearchTask(run_id=RUN_ID, question="Q", sub_question=sq.model_dump()))
                for sq in sqs
            )
        )

        assert len(store_of(runtime).sources) == 3
        assert runtime.source_slots == 3

    async def test_duplicates_give_their_slot_back(self) -> None:
        runtime = build_fake_runtime(config(max_sources_per_run=5), web=make_web(2))
        # Both pages already stored by an earlier branch (different source IDs).
        await run_node(runtime)
        assert runtime.source_slots == 2
        runtime.claimed_urls.clear()

        await run_node(runtime, SubQuestionItem(id="sq-2", text="again", queries=["rag"]))

        assert len(store_of(runtime).sources) == 2
        assert runtime.source_slots == 2

    async def test_a_timeout_while_storing_gives_the_slot_back(self) -> None:
        class SlowStore(InMemoryResearchStore):
            async def save_source(self, *, run_id: str, source: Any) -> tuple[str, bool]:
                await asyncio.sleep(10)
                raise AssertionError("unreachable")

        cfg = AgentConfig(research=ResearchConfig(max_sources_per_run=3), agent_timeout=0.2)  # type: ignore[arg-type]
        runtime = build_fake_runtime(cfg, web=make_web(2), store=SlowStore())

        update = await run_node(runtime)

        assert result_of(update)["status"] == "failed"
        assert runtime.source_slots == 0

    async def test_no_cap_by_default(self) -> None:
        runtime = build_fake_runtime(web=make_web(4))

        await run_node(runtime)

        assert len(store_of(runtime).sources) == 4


# ===========================================================================
# M5: researcher profiles and source scoring
# ===========================================================================

NOW = datetime(2026, 10, 27, tzinfo=UTC)
DOCS_URL = "https://docs.rag.example/guide"


def web_with_docs(count: int = 3) -> FakeWeb:
    """Site pages match "rag retrieval" twice, the docs page once, so it ranks last."""
    pages = {url(i): (f"rag page {i}", page_text(f"unique{i}")) for i in range(count)}
    pages[DOCS_URL] = ("Guide", " ".join(f"Docs sentence {k} about rag setup." for k in range(30)))
    return FakeWeb(pages=pages)


def docs_sub_question() -> SubQuestionItem:
    return SubQuestionItem(id="sq-1", text="How is RAG set up?", queries=["rag retrieval"])


async def run_profile(
    runtime: AgentRuntime, factory: Any, sq: SubQuestionItem | None = None
) -> dict[str, Any]:
    sq = sq or docs_sub_question()
    await runtime.store.save_plan(run_id=RUN_ID, plan={}, sub_questions=[sq])
    node = factory(runtime)
    return await node(WebResearchTask(run_id=RUN_ID, question="Q", sub_question=sq.model_dump()))


class _DatedPageExtract(FakePageExtract):
    async def extract(self, url: str) -> PageContent:
        page = await super().extract(url)
        page.published_at = "2025-06-01T00:00:00+00:00"
        return page


class TestProfiles:
    async def test_technical_researcher_attributes_events_and_usage(self) -> None:
        runtime = build_fake_runtime(config(pages_per_sub_question=1), web=web_with_docs())

        update = await run_profile(runtime, make_technical_researcher_node)

        assert {e["agent"] for e in update["events"]} == {"technical_researcher"}
        assert {e["agent"] for e in store_of(runtime).events} == {"technical_researcher"}
        assert result_of(update)["agent"] == "technical_researcher"
        assert [c["agent"] for c in update["llm_calls"]] == ["technical_researcher"]
        assert set(runtime.budget.per_agent) == {"technical_researcher"}

    async def test_market_researcher_attributes_events(self) -> None:
        runtime = build_fake_runtime(web=make_web(2))

        update = await run_profile(runtime, make_market_researcher_node, sub_question())

        assert {e["agent"] for e in update["events"]} == {"market_researcher"}
        assert result_of(update)["agent"] == "market_researcher"

    async def test_web_researcher_keeps_its_name(self) -> None:
        runtime = build_fake_runtime(web=make_web(2))

        update = await run_node(runtime)

        assert {e["agent"] for e in update["events"]} == {"web_researcher"}
        assert result_of(update)["agent"] == "web_researcher"

    async def test_selection_prompt_carries_guidance_and_source_types(self) -> None:
        cfg = config(pages_per_sub_question=1)

        technical = build_fake_runtime(cfg, web=web_with_docs())
        await run_profile(technical, make_technical_researcher_node)
        prompt = llm_of(technical).calls_for("SourceSelection")[0]["prompt"]
        assert "You are the technical researcher" in prompt
        assert "site: docs.rag.example (official docs)" in prompt
        assert "(unknown)" not in prompt, "unknown types are not labelled"

        market = build_fake_runtime(cfg, web=web_with_docs())
        await run_profile(market, make_market_researcher_node)
        market_prompt = llm_of(market).calls_for("SourceSelection")[0]["prompt"]
        assert "You are the market and cost researcher" in market_prompt

        web = build_fake_runtime(cfg, web=web_with_docs())
        await run_profile(web, make_web_researcher_node)
        web_prompt = llm_of(web).calls_for("SourceSelection")[0]["prompt"]
        assert "Prefer:" in web_prompt
        assert "You are the" not in web_prompt

    async def test_specialists_read_preferred_source_types_first_without_a_model_pick(self) -> None:
        cfg = config(pages_per_sub_question=1)
        responses: dict[str, Any] = {"SourceSelection": RuntimeError("down")}

        technical = build_fake_runtime(cfg, web=web_with_docs(), responses=responses)
        update = await run_profile(technical, make_technical_researcher_node)
        assert [s["url"] for s in update["sources"]] == [DOCS_URL]

        web = build_fake_runtime(cfg, web=web_with_docs(), responses=responses)
        update = await run_profile(web, make_web_researcher_node)
        assert [s["url"] for s in update["sources"]] == [url(0)], "web keeps search order"

    def test_prefer_types_is_stable(self) -> None:
        candidates = [
            Candidate("R1", "u1", "t", "s", "d", kind="forum"),
            Candidate("R2", "u2", "t", "s", "d", kind="news"),
            Candidate("R3", "u3", "t", "s", "d", kind="official_docs"),
            Candidate("R4", "u4", "t", "s", "d", kind="news"),
        ]
        ordered = prefer_types(candidates, ("official_docs", "news"))
        assert [c.label for c in ordered] == ["R3", "R2", "R4", "R1"]
        assert prefer_types(candidates, ()) == candidates
        assert TECHNICAL_PROFILE.preferred_types[0] == "official_docs"
        assert MARKET_PROFILE.agent == "market_researcher"

    def test_candidates_carry_their_source_type(self) -> None:
        merged = merge_candidates(
            ["q"],
            {"q": [sr(DOCS_URL), sr("https://www.reddit.com/r/x"), sr("https://site.example/a")]},
        )
        assert [c.kind for c in merged] == ["official_docs", "forum", "unknown"]


class TestSourceScoring:
    async def test_stored_sources_are_typed_and_scored(self) -> None:
        runtime = build_fake_runtime(web=web_with_docs(1), now=lambda: NOW)

        update = await run_profile(runtime, make_web_researcher_node)

        by_url = {s["url"]: s for s in update["sources"]}
        docs = by_url[DOCS_URL]
        assert docs["domain_type"] == "official_docs"
        assert docs["published_at"] is None
        assert docs["reliability_score"] == source_reliability(
            SourceSignals(domain_type="official_docs"), now=NOW, config=runtime.config.scoring
        )
        site = by_url[url(0)]
        assert site["domain_type"] == "unknown"
        assert site["reliability_score"] < docs["reliability_score"]
        stored = store_of(runtime).sources[docs["id"]]
        assert stored["domain_type"] == "official_docs"
        assert stored["reliability_score"] == docs["reliability_score"]
        found = {
            e["payload"]["url"]: e["payload"] for e in update["events"] if e["type"] == "source_found"
        }
        assert found[DOCS_URL]["domain_type"] == "official_docs"
        assert found[DOCS_URL]["reliability_score"] == docs["reliability_score"]

    async def test_publication_dates_are_stored_and_scored(self) -> None:
        site = make_web(1)
        runtime = build_fake_runtime(web=site, now=lambda: NOW)
        runtime.page_extract = _DatedPageExtract(site)

        update = await run_node(runtime)

        [source] = update["sources"]
        assert source["published_at"] == "2025-06-01T00:00:00+00:00"
        assert source["reliability_score"] == source_reliability(
            SourceSignals(domain_type="unknown", published_at=datetime(2025, 6, 1, tzinfo=UTC)),
            now=NOW,
            config=runtime.config.scoring,
        )
        assert store_of(runtime).sources[source["id"]]["published_at"] == source["published_at"]
