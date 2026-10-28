"""
M5 tests — academic researcher node (fake paper index, fake web, in-memory store).

Papers become scored sources (peer-reviewed or preprint, dated, with citation
counts); when the indexes return nothing usable the branch researches the web
instead, still as the academic researcher.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from researchmind_agents.config import AgentConfig, ResearchConfig
from researchmind_agents.nodes.academic_researcher import (
    AGENT,
    make_academic_researcher_node,
    merge_paper_lists,
    paper_type,
)
from researchmind_agents.reliability import SourceSignals, describes_methodology, source_reliability
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import SubQuestionItem, WebResearchTask
from researchmind_agents.storage import InMemoryResearchStore
from researchmind_agents.testing import FakeScholar, FakeWeb, ScriptedLLM, build_fake_runtime
from researchmind_agents.tools.scholarly import Paper
from researchmind_agents.tools.urls import normalize_url

RUN = "run-academic"
NOW = datetime(2026, 10, 27, tzinfo=UTC)
PUBLISHED = "2024-03-01T00:00:00+00:00"
ABSTRACT = (
    "We evaluate cold climate heat pumps using a dataset of 120 homes across three winters. "
    "Our method compares measured coefficients of performance with a gas furnace baseline. "
    "Heat pumps kept a coefficient above two at minus fifteen degrees in most homes."
)
assert len(ABSTRACT) >= 200, "above the default min_abstract_chars"


def paper(
    i: int,
    *,
    venue: str = "",
    citations: int | None = None,
    abstract: str = ABSTRACT,
    url: str | None = None,
    arxiv_id: str | None = None,
    published_at: str = PUBLISHED,
) -> Paper:
    return Paper(
        title=f"Cold climate heat pumps study {i}",
        url=url or f"https://arxiv.org/abs/2401.0000{i}",
        abstract=abstract,
        authors=["A. Author"],
        year=2024,
        published_at=published_at,
        venue=venue,
        citation_count=citations,
        arxiv_id=arxiv_id if arxiv_id is not None else f"2401.0000{i}",
        provider="fake",
    )


def sub_question(sq_id: str = "sq-1") -> SubQuestionItem:
    return SubQuestionItem(
        id=sq_id,
        text="How do heat pumps perform in cold climates?",
        queries=["heat pumps cold climates"],
        domain="academic",
    )


def page_text(seed: str) -> str:
    return " ".join(f"The {seed} heat pumps cold climates article sentence {k}." for k in range(15))


async def run_node(runtime: AgentRuntime, sq: SubQuestionItem | None = None) -> dict[str, Any]:
    sq = sq or sub_question()
    await runtime.store.save_plan(run_id=RUN, plan={}, sub_questions=[sq])
    node = make_academic_researcher_node(runtime)
    return await node(WebResearchTask(run_id=RUN, question="Q", sub_question=sq.model_dump()))


def runtime_with(
    papers: list[Paper] | None = None,
    *,
    config: AgentConfig | None = None,
    web: FakeWeb | None = None,
    scholar: FakeScholar | None = None,
    responses: dict[str, Any] | None = None,
) -> AgentRuntime:
    return build_fake_runtime(
        config,
        web=web,
        scholar=scholar or FakeScholar(papers=list(papers or [])),
        responses=responses,
        now=lambda: NOW,
    )


def store_of(runtime: AgentRuntime) -> InMemoryResearchStore:
    assert isinstance(runtime.store, InMemoryResearchStore)
    return runtime.store


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    assert isinstance(runtime.llm, ScriptedLLM)
    return runtime.llm


def result_of(update: dict[str, Any]) -> dict[str, Any]:
    assert len(update["research_results"]) == 1
    return update["research_results"][0]


def events(update: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    return [e for e in update["events"] if e["type"] == event_type]


def titles(update: dict[str, Any]) -> list[str]:
    return [s["title"] for s in update["sources"]]


# ===========================================================================
# Papers become scored sources
# ===========================================================================


class TestStoresPapers:
    async def test_papers_are_stored_as_typed_dated_scored_sources(self) -> None:
        reviewed = paper(1, venue="Energy and Buildings", citations=40)
        preprint = paper(2)
        runtime = runtime_with([reviewed, preprint])

        update = await run_node(runtime)

        assert result_of(update)["status"] == "completed"
        by_title = {s["title"]: s for s in update["sources"]}
        assert set(by_title) == {reviewed.title, preprint.title}
        for item, kind in ((reviewed, "peer_reviewed"), (preprint, "preprint")):
            source = by_title[item.title]
            assert source["domain_type"] == kind
            assert source["published_at"] == PUBLISHED
            assert source["url"] == normalize_url(item.url)
            text = f"{item.title}\n\n{item.abstract.strip()}"
            score = source_reliability(
                SourceSignals(
                    domain_type=kind,
                    published_at=datetime.fromisoformat(PUBLISHED),
                    citation_count=item.citation_count,
                    has_methodology=describes_methodology(text),
                ),
                now=NOW,
                config=runtime.config.scoring,
            )
            assert source["reliability_score"] == score
            stored = store_of(runtime).sources[source["id"]]
            assert stored["domain_type"] == kind
        assert by_title[reviewed.title]["reliability_score"] > by_title[preprint.title]["reliability_score"]

    async def test_text_is_title_and_abstract(self) -> None:
        runtime = runtime_with([paper(1)])

        update = await run_node(runtime)

        [passage] = update["passages"]
        assert passage["text"].startswith("Cold climate heat pumps study 1")
        assert "dataset of 120 homes" in passage["text"]

    async def test_no_page_is_fetched(self) -> None:
        web = FakeWeb(pages={"https://arxiv.org/abs/2401.00001": ("t", page_text("x"))})
        runtime = runtime_with([paper(1)], web=web)

        await run_node(runtime)

        assert web.fetched == []
        assert web.queries == []

    async def test_events_and_results_are_attributed_to_the_academic_researcher(self) -> None:
        runtime = runtime_with([paper(i) for i in range(1, 4)])

        update = await run_node(runtime)

        assert {e["agent"] for e in update["events"]} == {AGENT}
        assert {e["agent"] for e in store_of(runtime).events} == {AGENT}
        assert result_of(update)["agent"] == AGENT
        search = events(update, "search_completed")[0]["payload"]
        assert search["index"] == "scholarly"
        assert search["candidates"] == 3
        found = events(update, "source_found")
        assert {e["payload"]["domain_type"] for e in found} == {"preprint"}

    async def test_short_abstracts_are_skipped(self) -> None:
        runtime = runtime_with([paper(1, abstract="Too short to use."), paper(2)])

        update = await run_node(runtime)

        assert titles(update) == ["Cold climate heat pumps study 2"]
        search = events(update, "search_completed")[0]["payload"]
        assert search["papers_found"] == 2
        assert search["candidates"] == 1

    async def test_run_source_cap_applies(self) -> None:
        config = AgentConfig(research=ResearchConfig(max_sources_per_run=1))
        runtime = runtime_with([paper(1), paper(2)], config=config)

        update = await run_node(runtime)

        assert len(update["sources"]) == 1


# ===========================================================================
# Choosing papers
# ===========================================================================


class TestSelection:
    async def test_model_picks_when_there_are_more_papers_than_pages(self) -> None:
        config = AgentConfig(research=ResearchConfig(pages_per_sub_question=1))
        runtime = runtime_with(
            [paper(1, venue="Energy and Buildings", citations=40), paper(2), paper(3)],
            config=config,
            responses={"SourceSelection": json.dumps({"selected": ["R3"]})},
        )

        update = await run_node(runtime)

        assert titles(update) == ["Cold climate heat pumps study 3"]
        [call] = llm_of(runtime).calls_for("SourceSelection")
        assert "Energy and Buildings, 2024, 40 citations" in call["prompt"]
        assert "preprint, 2024" in call["prompt"]
        assert "arxiv.org" not in call["prompt"], "the model sees labels, not URLs"
        assert "You are the academic researcher" in call["prompt"]
        assert [c["agent"] for c in update["llm_calls"]] == [AGENT]

    async def test_no_model_call_when_papers_fit(self) -> None:
        runtime = runtime_with([paper(1), paper(2)])

        await run_node(runtime)

        assert llm_of(runtime).calls_for("SourceSelection") == []

    async def test_selection_failure_falls_back_to_index_order(self) -> None:
        config = AgentConfig(research=ResearchConfig(pages_per_sub_question=1))
        runtime = runtime_with(
            [paper(1), paper(2), paper(3)],
            config=config,
            responses={"SourceSelection": RuntimeError("model broke")},
        )

        update = await run_node(runtime)

        assert titles(update) == ["Cold climate heat pumps study 1"]
        assert any(
            "Paper selection failed" in e["payload"]["message"] for e in events(update, "error")
        )

    async def test_invalid_labels_fall_back_to_index_order(self) -> None:
        config = AgentConfig(research=ResearchConfig(pages_per_sub_question=1))
        runtime = runtime_with(
            [paper(1), paper(2)],
            config=config,
            responses={"SourceSelection": json.dumps({"selected": ["R9", "nope"]})},
        )

        update = await run_node(runtime)

        assert titles(update) == ["Cold climate heat pumps study 1"]


# ===========================================================================
# Falling back to the web
# ===========================================================================


def matching_web() -> FakeWeb:
    return FakeWeb(
        pages={
            "https://energy.example.edu/heat-pumps": ("Heat pumps cold climates", page_text("edu")),
            "https://blog.example/heat-pumps": ("Heat pumps cold climates blog", page_text("blog")),
        }
    )


class TestWebFallback:
    async def test_empty_index_falls_back_to_web_research(self) -> None:
        web = matching_web()
        runtime = runtime_with([], web=web)

        update = await run_node(runtime)

        [fallback] = events(update, "scholarly_fallback")
        assert fallback["payload"]["reason"] == "no papers found"
        assert web.queries == ["heat pumps cold climates"]
        assert update["sources"], "web pages were stored instead"
        assert result_of(update)["status"] == "completed"
        assert {e["agent"] for e in update["events"]} == {AGENT}
        domain_types = {s["domain_type"] for s in update["sources"]}
        assert "academic" in domain_types, ".edu pages are typed as academic"

    async def test_failing_index_falls_back_to_web_research(self) -> None:
        web = matching_web()
        runtime = runtime_with(web=web, scholar=FakeScholar(papers=[paper(1)], fail=True))

        update = await run_node(runtime)

        assert events(update, "scholarly_fallback")[0]["payload"]["reason"] == "no papers found"
        assert update["sources"]
        assert result_of(update)["agent"] == AGENT

    async def test_only_short_abstracts_falls_back_to_web_research(self) -> None:
        runtime = runtime_with([paper(1, abstract="Short.")], web=matching_web())

        update = await run_node(runtime)

        assert events(update, "scholarly_fallback")[0]["payload"]["reason"] == (
            "no papers with an abstract"
        )
        assert update["sources"]

    async def test_fallback_prefers_academic_sources(self) -> None:
        """With no model selection, .edu pages are read before others."""
        web = matching_web()
        config = AgentConfig(research=ResearchConfig(pages_per_sub_question=1))
        runtime = runtime_with(
            [], web=web, config=config, responses={"SourceSelection": RuntimeError("down")}
        )

        update = await run_node(runtime)

        assert [s["url"] for s in update["sources"]] == ["https://energy.example.edu/heat-pumps"]


# ===========================================================================
# Shared run state across branches
# ===========================================================================


class TestDeduplication:
    async def test_a_claimed_paper_is_not_stored_again(self) -> None:
        first, second = paper(1), paper(2)
        runtime = runtime_with([first, second])
        runtime.claimed_urls.add(normalize_url(first.url))

        update = await run_node(runtime)

        assert titles(update) == [second.title]
        assert result_of(update)["duplicates"] == 1

    async def test_parallel_branches_store_each_paper_once(self) -> None:
        papers = [paper(1), paper(2), paper(3)]
        runtime = runtime_with(papers)
        sq_a, sq_b = sub_question("sq-a"), sub_question("sq-b")
        await runtime.store.save_plan(run_id=RUN, plan={}, sub_questions=[sq_a, sq_b])
        node = make_academic_researcher_node(runtime)

        updates = await asyncio.gather(
            *(
                node(WebResearchTask(run_id=RUN, question="Q", sub_question=sq.model_dump()))
                for sq in (sq_a, sq_b)
            )
        )

        stored = [row["url"] for row in store_of(runtime).sources.values()]
        assert sorted(stored) == sorted(normalize_url(p.url) for p in papers)
        assert sum(len(u["sources"]) for u in updates) == 3

    async def test_budget_exhausted_skips_the_branch(self) -> None:
        scholar = FakeScholar(papers=[paper(1)])
        runtime = runtime_with(scholar=scholar)
        runtime.budget.tokens_used = runtime.budget.max_tokens

        update = await run_node(runtime)

        assert result_of(update)["status"] == "skipped"
        assert scholar.queries == []


# ===========================================================================
# Helpers
# ===========================================================================


class TestPaperType:
    def test_a_named_venue_is_peer_reviewed(self) -> None:
        assert paper_type(paper(1, venue="Nature Energy")) == "peer_reviewed"
        assert paper_type(paper(1, venue="  Energy   and Buildings ")) == "peer_reviewed"

    def test_no_venue_or_an_archive_is_a_preprint(self) -> None:
        for venue in ("", "arXiv", "arXiv.org", "CoRR", "bioRxiv", "SSRN", "arXiv preprint 2401.1"):
            assert paper_type(paper(1, venue=venue)) == "preprint", venue


class TestMergePaperLists:
    def test_round_robin_across_queries(self) -> None:
        a = [paper(1), paper(2)]
        b = [paper(3), paper(4)]

        merged = merge_paper_lists(["a", "b"], {"a": a, "b": b})

        assert [p.title[-1] for p in merged] == ["1", "3", "2", "4"]

    def test_drops_repeats_by_key_and_by_url(self) -> None:
        original = paper(1)
        same_key = paper(1, url="https://www.semanticscholar.org/paper/xyz")
        same_url = paper(9, url=original.url, arxiv_id="9999.99999")
        no_url = Paper(title="No link", abstract=ABSTRACT, url="")

        merged = merge_paper_lists(["q"], {"q": [original, same_key, same_url, no_url, paper(2)]})

        assert [p.title for p in merged] == [original.title, paper(2).title]

    def test_missing_queries(self) -> None:
        assert merge_paper_lists(["x"], {}) == []
        assert merge_paper_lists([], {}) == []
