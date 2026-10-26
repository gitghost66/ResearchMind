"""
M3 tests — the compiled graph end to end, with fakes for LLM, search, pages,
and embeddings.

Covers the M3 "Done when" items that can run without live APIs: one graph test
end to end, citations pointing only at stored sources, parallel fan-out, and a
run that stops at its budget cap and still returns a report.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from researchmind_agents.config import AgentConfig, ResearchConfig
from researchmind_agents.evals.check_grounding import check_grounding
from researchmind_agents.graph import build_research_graph, run_research
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import new_research_state
from researchmind_agents.storage import InMemoryResearchStore
from researchmind_agents.testing import FakeWeb, ScriptedLLM, build_fake_runtime

QUESTION = "heat pumps cold climates"
FILLER = " Heat pumps move heat rather than burning fuel, which is why their efficiency is quoted as a coefficient of performance." * 6


def web(pages: int = 6) -> FakeWeb:
    topics = ["benefits", "costs risks", "benefits efficiency", "costs installation", "benefits emissions", "risks noise"]
    return FakeWeb(
        pages={
            f"https://site{i}.example/article": (
                f"Heat pumps cold climates {topics[i % len(topics)]}",
                f"Heat pumps cold climates {topics[i % len(topics)]} finding {i}." + FILLER,
            )
            for i in range(pages)
        }
    )


def plan_json(*texts: str) -> str:
    return json.dumps(
        {
            "objective": "Decide",
            "sub_questions": [{"text": t, "queries": [t]} for t in texts],
        }
    )


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    assert isinstance(runtime.llm, ScriptedLLM)
    return runtime.llm


def store_of(runtime: AgentRuntime) -> InMemoryResearchStore:
    assert isinstance(runtime.store, InMemoryResearchStore)
    return runtime.store


class TestEndToEnd:
    async def test_question_in_grounded_report_out(self) -> None:
        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-e2e", question=QUESTION)

        report = state["report"]
        assert report["generation"] == "llm"
        stored = await runtime.store.list_source_ids(run_id="run-e2e")
        assert stored
        assert all(c["source_id"] in stored for c in report["sources"])

        grounding = check_grounding(report, state["report_markdown"], stored)
        assert grounding.citation_resolution_rate == 1.0
        assert grounding.cited_statement_rate >= 0.8

        agents = {e["agent"] for e in state["events"]}
        assert agents == {"planner", "web_researcher", "synthesizer"}
        assert len(state["research_results"]) == len(state["sub_questions"]) == 2
        assert len(store_of(runtime).reports) == 1
        # Events were persisted as they happened, not only kept in state.
        assert len(store_of(runtime).events) == len(state["events"])

    async def test_final_state_is_plain_json(self) -> None:
        """Checkpointers must be able to serialise the state without custom types."""
        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-json", question=QUESTION)
        json.dumps(state)

    async def test_state_is_checkpointed_under_the_run_id(self) -> None:
        runtime = build_fake_runtime(web=web())
        saver = InMemorySaver()
        graph = build_research_graph(runtime, checkpointer=saver)
        config: Any = {"configurable": {"thread_id": "run-ckpt"}}
        await graph.ainvoke(new_research_state(run_id="run-ckpt", question=QUESTION), config=config)

        snapshot = await graph.aget_state(config)
        assert snapshot.values["report"]["question"] == QUESTION
        assert snapshot.next == ()

    async def test_planner_failure_still_produces_a_report(self) -> None:
        runtime = build_fake_runtime(web=web(), responses={"PlanOutput": RuntimeError("down")})
        state = await run_research(runtime, run_id="run-fallback", question=QUESTION)

        assert state["plan"]["fallback_reason"]
        assert len(state["sub_questions"]) == 1
        assert state["report"]["sources"]

    async def test_rejects_unsupported_mode_and_empty_question(self) -> None:
        runtime = build_fake_runtime()
        with pytest.raises(ValueError, match="not available"):
            await run_research(runtime, run_id="r", question=QUESTION, mode="deep")
        with pytest.raises(ValueError, match="empty"):
            await run_research(runtime, run_id="r", question="   ")


class TestFanOut:
    async def _max_parallel_searches(self, config: AgentConfig, sub_questions: int) -> tuple[int, int]:
        runtime = build_fake_runtime(
            config,
            web=web(),
            responses={"PlanOutput": plan_json(*(f"heat pumps topic {i}" for i in range(sub_questions)))},
        )
        original = runtime.search.search_multi
        in_flight = peak = 0

        async def slow_search_multi(queries: list[str], **kwargs: Any) -> Any:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return await original(queries, **kwargs)

        runtime.search.search_multi = slow_search_multi  # type: ignore[method-assign]
        state = await run_research(runtime, run_id="run-fan", question=QUESTION)
        return peak, len(state["research_results"])

    async def test_one_branch_per_sub_question_running_in_parallel(self) -> None:
        peak, branches = await self._max_parallel_searches(AgentConfig(concurrency_cap=4), 3)
        assert branches == 3
        assert peak == 3

    async def test_concurrency_cap_limits_parallel_branches(self) -> None:
        peak, branches = await self._max_parallel_searches(AgentConfig(concurrency_cap=1), 3)
        assert branches == 3
        assert peak == 1

    async def test_branches_do_not_fetch_the_same_page_twice(self) -> None:
        site = web()
        runtime = build_fake_runtime(
            web=site,
            responses={"PlanOutput": plan_json("heat pumps cold climates benefits", "heat pumps cold climates costs")},
        )
        state = await run_research(runtime, run_id="run-dedupe", question=QUESTION)

        assert len(site.fetched) == len(set(site.fetched))
        urls = [s["url"] for s in store_of(runtime).sources.values()]
        assert len(urls) == len(set(urls))
        assert all(r["status"] == "completed" for r in state["research_results"])


class TestBudget:
    async def test_budget_spent_by_planner_skips_research_and_still_reports(self) -> None:
        site = web()
        config = AgentConfig(max_tokens_per_run=1000, budget_synthesis_reserve=0.2)
        runtime = build_fake_runtime(config, web=site, tokens=(900, 100))
        state = await run_research(runtime, run_id="run-broke", question=QUESTION)

        assert site.queries == [], "research must be skipped"
        assert state["research_results"] == []
        report = state["report"]
        assert report["generation"] == "empty"
        assert any("budget" in item for item in report["limitations"])
        assert len(store_of(runtime).reports) == 1
        assert llm_of(runtime).calls_for("ReportOutput") == []

    async def test_budget_hit_during_research_gives_extractive_report(self) -> None:
        """Planner + selection calls spend the cap; the stored passages are still reported."""
        config = AgentConfig(
            max_tokens_per_run=2500,
            budget_synthesis_reserve=0.0,
            research=ResearchConfig(pages_per_sub_question=1),
        )
        runtime = build_fake_runtime(
            config,
            web=web(),
            tokens=(900, 100),
            responses={"PlanOutput": plan_json("heat pumps cold climates benefits", "heat pumps cold climates costs")},
        )
        state = await run_research(runtime, run_id="run-cap", question=QUESTION)

        assert runtime.budget.exceeded
        assert llm_of(runtime).calls_for("SourceSelection"), "selection spent the budget"
        assert llm_of(runtime).calls_for("ReportOutput") == []
        report = state["report"]
        assert report["generation"] == "extractive"
        assert report["sources"]
        stored = await runtime.store.list_source_ids(run_id="run-cap")
        assert all(c["source_id"] in stored for c in report["sources"])
        assert any("budget" in item for item in report["limitations"])
