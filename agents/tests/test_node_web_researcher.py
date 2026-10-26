"""
M3 tests — web researcher node.

Driven against the fakes in ``researchmind_agents.testing``: a FakeWeb corpus
for search and fetches, the real EmbeddingService over a fake client, a
scripted LLM, and the in-memory store.
"""

from __future__ import annotations

import asyncio
from typing import Any

from researchmind_agents.config import AgentConfig, ResearchConfig
from researchmind_agents.nodes.web_researcher import make_web_researcher_node, merge_candidates
from researchmind_agents.prompts.web_researcher import PROMPT_VERSION
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
