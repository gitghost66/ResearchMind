"""
The compiled graph end to end, with fakes for LLM, search, pages, papers, and
embeddings.

M3: one graph test end to end, citations pointing only at stored sources,
parallel fan-out, and a run that stops at its budget cap and still returns a
report.

M5 (acceptance): in the verified modes every claim links to stored passages,
the report carries verdicts and confidence, and the confidence is the
documented formula applied to the report's own claims.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

from researchmind_agents.config import AgentConfig, ResearchConfig
from researchmind_agents.evals.check_grounding import check_grounding
from researchmind_agents.graph import (
    build_research_graph,
    build_run_config,
    pipeline_steps,
    run_research,
)
from researchmind_agents.reliability import overall_confidence
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import USABLE_VERDICTS, new_research_state
from researchmind_agents.storage import InMemoryResearchStore
from researchmind_agents.testing import FakeScholar, FakeWeb, ScriptedLLM, build_fake_runtime
from researchmind_agents.tools.scholarly import Paper

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
            await run_research(runtime, run_id="r", question=QUESTION, mode="nonsense")
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


class _ChainRecorder(AsyncCallbackHandler):
    """Records the name and metadata of every chain (graph and node) a run starts."""

    def __init__(self) -> None:
        self.starts: list[tuple[str, dict[str, Any], list[str]]] = []

    async def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        self.starts.append(
            (kwargs.get("name") or "", dict(kwargs.get("metadata") or {}), list(kwargs.get("tags") or []))
        )


class TestTracing:
    def test_run_config_names_the_trace_and_tags_the_run(self) -> None:
        runtime = build_fake_runtime(AgentConfig(concurrency_cap=3))
        config = build_run_config(runtime, run_id="run-t", mode="quick")

        assert config["configurable"] == {"thread_id": "run-t"}
        assert config["max_concurrency"] == 3
        assert config["run_name"] == "research:quick"
        assert config["metadata"] == {"run_id": "run-t", "mode": "quick"}
        assert "researchmind" in config["tags"]

    async def test_every_node_span_carries_the_run_id(self) -> None:
        runtime = build_fake_runtime(web=web())
        recorder = _ChainRecorder()
        graph = build_research_graph(runtime)
        config = build_run_config(runtime, run_id="run-trace", mode="quick")
        config["callbacks"] = [recorder]

        await graph.ainvoke(new_research_state(run_id="run-trace", question=QUESTION), config=config)

        names = [name for name, _, _ in recorder.starts]
        assert names[0] == "research:quick"
        assert {"planner", "web_researcher", "synthesizer"} <= set(names)
        assert all(meta.get("run_id") == "run-trace" for _, meta, _ in recorder.starts)


class TestPipelineSteps:
    def test_quick_mode_steps_match_the_event_agents(self) -> None:
        assert pipeline_steps("quick") == ("planner", "web_researcher", "synthesizer")

    def test_verified_modes_add_researchers_and_verification(self) -> None:
        assert pipeline_steps("deep") == (
            "planner",
            "web_researcher",
            "technical_researcher",
            "market_researcher",
            "claim_extractor",
            "fact_checker",
            "contradiction_detector",
            "sufficiency_checker",
            "synthesizer",
        )
        assert pipeline_steps("academic") == (
            "planner",
            "web_researcher",
            "technical_researcher",
            "market_researcher",
            "academic_researcher",
            "claim_extractor",
            "fact_checker",
            "contradiction_detector",
            "sufficiency_checker",
            "synthesizer",
        )

    def test_unknown_mode_has_no_steps(self) -> None:
        assert pipeline_steps("nope") == ()

    async def test_every_step_emits_events_under_its_own_name(self) -> None:
        """Run progress is read from events, so step names must match event agents."""
        runtime = build_fake_runtime(web=web(), scholar=scholar())
        state = await run_research(runtime, run_id="run-steps", question=QUESTION, mode="academic")
        agents = {e["agent"] for e in state["events"]}
        assert agents <= set(pipeline_steps("academic"))


# ===========================================================================
# M5 — verified modes
# ===========================================================================


def scholar() -> FakeScholar:
    return FakeScholar(
        papers=[
            Paper(
                title=f"Heat pumps in cold climates: field study {i}",
                url=f"https://arxiv.org/abs/2401.0000{i}",
                abstract=(
                    f"We evaluate heat pumps cold climates benefits across {40 + i} homes. "
                    "Our method compares measured coefficient of performance with a dataset "
                    "of gas furnace installations over two winters. " * 2
                ),
                venue="Energy and Buildings" if i % 2 else "arXiv",
                year=2024,
                published_at="2024-03-01T00:00:00+00:00",
                citation_count=30 * i,
                arxiv_id=f"2401.0000{i}",
                provider="fake",
            )
            for i in range(3)
        ]
    )


def domain_plan(*pairs: tuple[str, str]) -> str:
    return json.dumps(
        {
            "objective": "Decide",
            "sub_questions": [{"text": t, "queries": [t], "domain": d} for t, d in pairs],
        }
    )


def all_statements(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [*report["executive_summary"], *(s for sec in report["sections"] for s in sec["statements"])]


class TestVerifiedModes:
    async def test_claims_always_link_to_stored_passages(self) -> None:
        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-deep", question=QUESTION, mode="deep")
        store = store_of(runtime)

        assert state["claims"], "the fake pipeline extracts claims"
        for claim in state["claims"]:
            assert claim["passage_ids"]
            assert claim["evidence"]
            assert set(claim["passage_ids"]) <= {e["passage_id"] for e in claim["evidence"]}
            for link in claim["evidence"]:
                assert store.passages[link["passage_id"]]["run_id"] == "run-deep"
                assert store.passages[link["passage_id"]]["source_id"] == link["source_id"]
        # The fact checker stored every claim, with its links.
        assert set(store.claims) == {c["id"] for c in state["claims"]}
        assert all(row["evidence"] for row in store.claims.values())

    async def test_report_includes_verdicts_and_confidence(self) -> None:
        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-deep", question=QUESTION, mode="deep")
        report = state["report"]

        assert report["generation"] == "llm"
        assert 0.0 < report["overall_confidence"] <= 1.0
        assert report["key_findings"]
        labels = {s["label"] for s in report["sources"]}
        for finding in report["key_findings"]:
            assert finding["verdict"] in USABLE_VERDICTS
            assert 0.0 <= finding["confidence"] <= 1.0
            assert finding["citations"] and set(finding["citations"]) <= labels
            assert all(e["source_label"] in labels for e in finding["evidence"])
        assert report["verification"]["claims_extracted"] == len(state["claims"])

        finding_labels = {f["label"] for f in report["key_findings"]}
        for statement in all_statements(report):
            assert statement["claims"] and set(statement["claims"]) <= finding_labels
            assert statement["citations"]

        markdown = state["report_markdown"]
        assert "**Overall confidence:**" in markdown
        assert "## Key findings" in markdown

    async def test_citations_resolve_to_stored_sources(self) -> None:
        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-deep", question=QUESTION, mode="deep")
        stored = await runtime.store.list_source_ids(run_id="run-deep")

        grounding = check_grounding(state["report"], state["report_markdown"], stored)
        assert grounding.citation_resolution_rate == 1.0
        assert grounding.cited_statement_rate == 1.0

    async def test_overall_confidence_is_the_documented_formula(self) -> None:
        """Recomputed from the report alone: no hidden or model-made number."""
        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-deep", question=QUESTION, mode="deep")
        report = state["report"]

        confidence = {f["label"]: f["confidence"] for f in report["key_findings"]}
        counts = Counter(label for s in all_statements(report) for label in s["claims"])
        covered = {f["sub_question_id"] for f in report["key_findings"]}
        expected = overall_confidence(
            [(confidence[label], n) for label, n in counts.items()],
            sub_questions=len(state["sub_questions"]),
            covered_sub_questions=len(covered & {sq["id"] for sq in state["sub_questions"]}),
            config=runtime.config.scoring,
        )
        assert report["overall_confidence"] == expected

    async def test_sub_questions_go_to_their_domain_researcher(self) -> None:
        runtime = build_fake_runtime(
            web=web(),
            responses={
                "PlanOutput": domain_plan(
                    ("heat pumps cold climates benefits", "web"),
                    ("heat pumps cold climates efficiency benchmark", "technical"),
                    ("heat pumps cold climates costs installation", "market"),
                )
            },
        )
        state = await run_research(runtime, run_id="run-route", question=QUESTION, mode="deep")

        by_agent = {r["agent"]: r for r in state["research_results"]}
        assert set(by_agent) == {"web_researcher", "technical_researcher", "market_researcher"}
        domains = {sq["domain"] for sq in state["sub_questions"]}
        assert domains == {"web", "technical", "market"}
        # Usage is attributed to the specialist that spent it.
        agents = {e["agent"] for e in state["events"]}
        assert {"claim_extractor", "fact_checker"} <= agents

    async def test_deep_mode_never_routes_to_the_academic_researcher(self) -> None:
        runtime = build_fake_runtime(
            web=web(),
            responses={"PlanOutput": domain_plan(("heat pumps cold climates benefits", "academic"))},
        )
        state = await run_research(runtime, run_id="run-noacad", question=QUESTION, mode="deep")
        assert [r["agent"] for r in state["research_results"]] == ["web_researcher"]

    async def test_academic_mode_reads_papers(self) -> None:
        runtime = build_fake_runtime(web=web(), scholar=scholar())
        state = await run_research(runtime, run_id="run-acad", question=QUESTION, mode="academic")

        assert "academic_researcher" in {r["agent"] for r in state["research_results"]}
        papers = [s for s in state["sources"] if s["url"].startswith("https://arxiv.org/abs/")]
        assert papers
        assert {s["domain_type"] for s in papers} <= {"peer_reviewed", "preprint"}
        assert all(s["published_at"] for s in papers)
        paper_ids = {s["id"] for s in papers}
        assert any(
            e["source_id"] in paper_ids for c in state["claims"] for e in c["evidence"]
        ), "claims are extracted from paper abstracts too"
        assert state["report"]["overall_confidence"] is not None

    async def test_quick_mode_has_no_claims_or_confidence(self) -> None:
        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-quick", question=QUESTION)
        report = state["report"]

        assert state["claims"] == []
        assert report["key_findings"] == [] and report["verification"] == {}
        assert report["overall_confidence"] is None
        assert llm_of(runtime).calls_for("ClaimExtractionOutput") == []
        assert store_of(runtime).claims == {}

    async def test_sources_are_scored_in_every_mode(self) -> None:
        site = web()
        site.pages["https://docs.example.org/docs/heat-pumps"] = (
            "Heat pumps cold climates benefits costs risks",
            "Heat pumps cold climates benefits costs risks documentation." + FILLER,
        )
        runtime = build_fake_runtime(web=site, config=AgentConfig(research=ResearchConfig(pages_per_sub_question=8)))
        state = await run_research(runtime, run_id="run-score", question=QUESTION)

        kinds = {s["url"]: (s["domain_type"], s["reliability_score"]) for s in state["sources"]}
        docs_type, docs_score = kinds["https://docs.example.org/docs/heat-pumps"]
        assert docs_type == "official_docs"
        unknown_scores = [score for kind, score in kinds.values() if kind == "unknown"]
        assert unknown_scores and all(docs_score > s for s in unknown_scores)

    async def test_failed_fact_check_falls_back_to_a_passage_report(self) -> None:
        runtime = build_fake_runtime(
            web=web(), responses={"FactCheckOutput": RuntimeError("model down")}
        )
        state = await run_research(runtime, run_id="run-nofc", question=QUESTION, mode="deep")
        report = state["report"]

        assert all(c["verdict"] == "unverified" for c in state["claims"])
        # Unverified claims are stored too, still linked to their passages.
        assert all(row["evidence"] for row in store_of(runtime).claims.values())
        assert report["overall_confidence"] is None
        assert report["key_findings"] == []
        assert report["sources"], "the passage path still writes a cited report"
        assert any("No claim could be verified" in item for item in report["limitations"])

    async def test_budget_spent_by_planner_skips_verification(self) -> None:
        config = AgentConfig(max_tokens_per_run=1000, budget_synthesis_reserve=0.2)
        runtime = build_fake_runtime(config, web=web(), tokens=(900, 100))
        state = await run_research(runtime, run_id="run-broke-deep", question=QUESTION, mode="deep")

        llm = llm_of(runtime)
        assert llm.calls_for("ClaimExtractionOutput") == []
        assert llm.calls_for("FactCheckOutput") == []
        assert state["report"]["generation"] == "empty"
        assert len(store_of(runtime).reports) == 1

    async def test_verified_state_is_plain_json(self) -> None:
        runtime = build_fake_runtime(web=web(), scholar=scholar())
        state = await run_research(runtime, run_id="run-json-acad", question=QUESTION, mode="academic")
        json.dumps(state)


# ===========================================================================
# M6 — contradictions, the gap-filling loop, decision mode
# ===========================================================================

LAB_CLAIM = "The Thermo X200 heat pump reaches a heating efficiency of 310 percent at minus fifteen degrees."
FIELD_CLAIM = "The Thermo X200 heat pump reaches a heating efficiency of 220 percent at minus fifteen degrees."
X200 = "Thermo X200 heat pump heating efficiency at minus fifteen degrees"


def conflicting_web() -> FakeWeb:
    """Two sources that disagree on one figure, four years apart, plus an unrelated page."""
    return FakeWeb(
        pages={
            "https://thermo.example/x200/datasheet": (
                "Thermo X200 datasheet",
                LAB_CLAIM + " The figure comes from the 2021 laboratory rating." + FILLER,
            ),
            "https://energy-agency.example/field-trial-2025": (
                "Thermo X200 field trial",
                FIELD_CLAIM + " The agency measured it in occupied homes in the winter of 2025." + FILLER,
            ),
            "https://furnace.example/service": (
                "Furnace servicing",
                "Gas furnaces need yearly servicing by a certified technician." + FILLER,
            ),
        },
        published={
            "https://thermo.example/x200/datasheet": "2021-02-01T00:00:00+00:00",
            "https://energy-agency.example/field-trial-2025": "2025-06-01T00:00:00+00:00",
        },
    )


def x200_plan() -> str:
    return domain_plan((f"What is the {X200}", "web"))


def contradiction_responder(cause: str) -> Any:
    """Name the two X200 claims, by the labels the prompt gives them, as conflicting."""

    def respond(prompt: str) -> str:
        labels = {m.group(2): m.group(1) for m in re.finditer(r"^\[(C\d+)\] \(\w+\) (.*)$", prompt, re.M)}
        if LAB_CLAIM not in labels or FIELD_CLAIM not in labels:
            return json.dumps({"contradictions": []})
        return json.dumps(
            {
                "contradictions": [
                    {
                        "claims": [labels[LAB_CLAIM], labels[FIELD_CLAIM]],
                        "likely_cause": cause,
                        "explanation": "The datasheet gives a 2021 laboratory rating; the 2025 field trial measured less.",
                    }
                ]
            }
        )

    return respond


def recompute(runtime: AgentRuntime, report: dict[str, Any], sub_questions: int) -> float | None:
    confidence = {f["label"]: f["confidence"] for f in report["key_findings"]}
    statements = [*all_statements(report), *report["recommendation"]]
    counts = Counter(label for s in statements for label in s["claims"])
    covered = {f["sub_question_id"] for f in report["key_findings"]}
    return overall_confidence(
        [(confidence[label], n) for label, n in counts.items()],
        sub_questions=sub_questions,
        covered_sub_questions=len(covered - {None}),
        unresolved_contradictions=sum(1 for x in report["contradictory_evidence"] if not x["resolved"]),
        config=runtime.config.scoring,
    )


class TestContradictions:
    async def test_seeded_conflicting_sources_produce_a_contradiction_with_its_cause(self) -> None:
        """M6 acceptance: a contradiction between the seeded sources, with a plausible cause."""
        runtime = build_fake_runtime(
            web=conflicting_web(),
            responses={"PlanOutput": x200_plan(), "ContradictionOutput": contradiction_responder("date")},
        )
        state = await run_research(runtime, run_id="run-x200", question=QUESTION, mode="deep")

        [item] = state["contradictions"]
        texts = {c["id"]: c["text"] for c in state["claims"]}
        assert {texts[cid] for cid in item["claim_ids"]} == {LAB_CLAIM, FIELD_CLAIM}
        assert item["likely_cause"] == "date" and item["resolved"] is True
        # The cause is plausible because the evidence shows it: the model saw both dates.
        [call] = llm_of(runtime).calls_for("ContradictionOutput")
        assert "2021-02-01" in call["prompt"] and "2025-06-01" in call["prompt"]
        assert store_of(runtime).contradictions[item["id"]]["claim_ids"] == item["claim_ids"]
        assert any(e["type"] == "contradiction_found" for e in state["events"])

        report = state["report"]
        [listed] = report["contradictory_evidence"]
        findings = {f["label"]: f for f in report["key_findings"]}
        assert {findings[label]["text"] for label in listed["claims"]} == {LAB_CLAIM, FIELD_CLAIM}
        assert listed["likely_cause"] == "date" and listed["resolved"] is True
        assert set(listed["citations"]) <= {s["label"] for s in report["sources"]}
        assert len(listed["citations"]) == 2
        assert report["verification"]["contradictions"] == 1
        assert "## Contradictory evidence" in state["report_markdown"]
        # The synthesizer was told about it.
        assert "Contradictions between claims:" in llm_of(runtime).calls_for("VerifiedReportOutput")[0]["prompt"]

    async def test_an_unresolved_contradiction_lowers_the_confidence(self) -> None:
        results = {}
        for cause in ("date", "unexplained"):
            runtime = build_fake_runtime(
                web=conflicting_web(),
                responses={"PlanOutput": x200_plan(), "ContradictionOutput": contradiction_responder(cause)},
            )
            state = await run_research(runtime, run_id=f"run-{cause}", question=QUESTION, mode="deep")
            report = state["report"]
            assert report["overall_confidence"] == recompute(runtime, report, len(state["sub_questions"]))
            results[cause] = report
        assert results["unexplained"]["verification"]["unresolved_contradictions"] == 1
        assert results["unexplained"]["overall_confidence"] < results["date"]["overall_confidence"]
        assert any("does not explain why" in item for item in results["unexplained"]["limitations"])

    async def test_no_conflict_no_record(self) -> None:
        runtime = build_fake_runtime(web=conflicting_web(), responses={"PlanOutput": x200_plan()})
        state = await run_research(runtime, run_id="run-none", question=QUESTION, mode="deep")
        assert state["contradictions"] == []
        assert state["report"]["contradictory_evidence"] == []


def always_gaps_config(**overrides: Any) -> AgentConfig:
    """The rules never find a sub-question covered, so every round has gaps."""
    from researchmind_agents.config import SufficiencyConfig

    return AgentConfig(
        research=ResearchConfig(pages_per_sub_question=2),
        sufficiency=SufficiencyConfig(min_sites_per_sub_question=99, loop_deadlines=()),
        **overrides,
    )


def gap_responder() -> Any:
    """Each gap round adds one sub-question for its first gap."""
    count = 0

    def respond(prompt: str) -> str:
        nonlocal count
        count += 1
        label = re.search(r"^\[(G\d+)\]", prompt, re.M)
        return json.dumps(
            {
                "sub_questions": [
                    {
                        "text": f"Heat pumps cold climates follow-up {count}",
                        "queries": [f"heat pumps cold climates follow-up round {count}"],
                        "gap": label.group(1) if label else "",
                    }
                ]
            }
        )

    return respond


class TestLoop:
    async def test_the_loop_terminates_at_max_iterations(self) -> None:
        """M6 acceptance: gaps every round, yet exactly max_iterations rounds run."""
        runtime = build_fake_runtime(
            always_gaps_config(max_iterations=3), web=web(12), responses={"GapPlanOutput": gap_responder()}
        )
        state = await run_research(runtime, run_id="run-loop", question=QUESTION, mode="deep")

        assert state["iteration"] == 2
        assert state["sufficiency"]["status"] == "needs_more"
        assert state["sufficiency"]["stopped_by"] == "max_iterations"
        llm = llm_of(runtime)
        assert len(llm.calls_for("PlanOutput")) == 1
        assert len(llm.calls_for("GapPlanOutput")) == 2
        checks = [e for e in state["events"] if e["type"] == "sufficiency_checked"]
        assert [e["payload"]["decision"] for e in checks] == ["loop", "loop", "synthesize"]
        assert [sq["iteration"] for sq in state["sub_questions"]] == [0, 0, 1, 2]
        assert store_of(runtime).iterations["run-loop"] == 2
        assert state["report"]["coverage"]["iterations"] == 3
        assert len(store_of(runtime).reports) == 1

    async def test_each_round_researches_and_verifies_only_its_new_material(self) -> None:
        runtime = build_fake_runtime(
            always_gaps_config(max_iterations=3), web=web(12), responses={"GapPlanOutput": gap_responder()}
        )
        state = await run_research(runtime, run_id="run-rounds", question=QUESTION, mode="deep")

        # Researchers ran once per sub-question: 2 in round 0, one per gap round.
        assert len(state["research_results"]) == 4
        # A follow-up fills its root's gap, so its claims count toward the root.
        roots = {sq["id"] for sq in state["sub_questions"] if sq["iteration"] == 0}
        assert all(sq["parent_id"] in roots for sq in state["sub_questions"] if sq["iteration"] > 0)

        claims = state["claims"]
        rounds = {c["iteration"] for c in claims}
        assert rounds == {0, 1, 2}, "every round found new passages and claims"
        # No passage is read in two rounds.
        by_round: dict[int, set[str]] = {}
        for c in claims:
            by_round.setdefault(c["iteration"], set()).update(c["passage_ids"])
        assert not (by_round[0] & by_round[1]) and not (by_round[1] & by_round[2])
        # Each claim was fact-checked once, in its own round, and stored once.
        prompts = [call["prompt"] for call in llm_of(runtime).calls_for("FactCheckOutput")]
        for c in claims:
            assert sum(f"] {c['text']}" in p for p in prompts) == 1, c["text"]
        assert set(store_of(runtime).claims) == {c["id"] for c in claims}
        assert all(c["verdict"] != "unverified" for c in claims)

    async def test_a_gap_round_that_adds_nothing_goes_to_the_report(self) -> None:
        runtime = build_fake_runtime(always_gaps_config(max_iterations=3), web=web(12))
        state = await run_research(runtime, run_id="run-empty-gap", question=QUESTION, mode="deep")

        assert len(llm_of(runtime).calls_for("GapPlanOutput")) == 1
        assert state["iteration"] == 1
        assert len(state["research_results"]) == 2, "nothing new was researched"
        assert llm_of(runtime).calls_for("VerifiedReportOutput")

    async def test_the_loop_stops_at_the_modes_deadline(self) -> None:
        from researchmind_agents.config import SufficiencyConfig

        clock = [0.0]
        config = AgentConfig(
            research=ResearchConfig(pages_per_sub_question=2),
            sufficiency=SufficiencyConfig(min_sites_per_sub_question=99, loop_deadlines=(("deep", 60.0),)),
        )
        runtime = build_fake_runtime(config, web=web(12), clock=lambda: clock[0])
        clock[0] = 61.0
        state = await run_research(runtime, run_id="run-late", question=QUESTION, mode="deep")

        assert state["iteration"] == 0
        assert state["sufficiency"]["stopped_by"] == "deadline"
        assert llm_of(runtime).calls_for("GapPlanOutput") == []

    async def test_quick_mode_never_loops(self) -> None:
        runtime = build_fake_runtime(always_gaps_config(), web=web())
        state = await run_research(runtime, run_id="run-q", question=QUESTION)
        assert state["max_iterations"] == 1
        assert state["sufficiency"] == {}
        assert {e["agent"] for e in state["events"]} == {"planner", "web_researcher", "synthesizer"}

    async def test_many_rounds_fit_the_recursion_limit(self) -> None:
        runtime = build_fake_runtime(
            always_gaps_config(max_iterations=5), web=web(20), responses={"GapPlanOutput": gap_responder()}
        )
        assert build_run_config(runtime, run_id="r", mode="deep")["recursion_limit"] == 40
        state = await run_research(runtime, run_id="run-five", question=QUESTION, mode="deep")
        assert state["iteration"] == 4

    async def test_looped_state_is_plain_json(self) -> None:
        runtime = build_fake_runtime(
            always_gaps_config(max_iterations=2), web=web(12), responses={"GapPlanOutput": gap_responder()}
        )
        state = await run_research(runtime, run_id="run-json-loop", question=QUESTION, mode="deep")
        json.dumps(state)


class TestDecisionMode:
    async def test_report_ends_in_a_recommendation(self) -> None:
        from researchmind_agents.prompts.synthesizer import DECISION_PROMPT_VERSION

        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-decide", question=QUESTION, mode="decision")
        report = state["report"]

        assert report["generation"] == "llm"
        assert report["prompt_version"] == DECISION_PROMPT_VERSION
        assert report["recommendation"]
        labels = {f["label"] for f in report["key_findings"]}
        for statement in report["recommendation"]:
            assert statement["claims"] and set(statement["claims"]) <= labels
            assert statement["citations"]
        assert "## Recommendation" in state["report_markdown"]
        assert report["overall_confidence"] == recompute(runtime, report, len(state["sub_questions"]))
        # Market research is favoured.
        assert "market" in {sq["domain"] for sq in state["sub_questions"]}
        assert "market_researcher" in {r["agent"] for r in state["research_results"]}

    async def test_citations_resolve_to_stored_sources(self) -> None:
        runtime = build_fake_runtime(web=web())
        state = await run_research(runtime, run_id="run-decide2", question=QUESTION, mode="decision")
        stored = await runtime.store.list_source_ids(run_id="run-decide2")
        grounding = check_grounding(state["report"], state["report_markdown"], stored)
        assert grounding.citation_resolution_rate == 1.0
        assert grounding.cited_statement_rate == 1.0


# ---------------------------------------------------------------------------
# M8: resuming from a checkpoint
# ---------------------------------------------------------------------------


class _HeldLLM(ScriptedLLM):
    """Calls for the schema named ``held`` wait for ``gate``."""

    def __init__(self, *args: Any, held: str, gate: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.held = held
        self.gate = gate
        self.reached = asyncio.Event()

    async def _complete(self, **kwargs: Any) -> Any:
        schema = kwargs.get("json_schema")
        if schema is not None and schema.__name__ == self.held:
            self.reached.set()
            await self.gate.wait()
        return await super()._complete(**kwargs)


def held_runtime(store: InMemoryResearchStore, *, held: str) -> tuple[AgentRuntime, _HeldLLM]:
    runtime = build_fake_runtime(web=web(), store=store)
    llm = _HeldLLM(
        replace(runtime.config.llm, base_delay=0.0, overload_delay=0.0),
        runtime.budget.wrap(None),
        held=held,
        gate=asyncio.Event(),
    )
    runtime.llm = llm
    return runtime, llm


class TestResume:
    async def test_an_interrupted_run_resumes_without_redoing_finished_steps(self) -> None:
        saver, store = InMemorySaver(), InMemoryResearchStore()
        first, held = held_runtime(store, held="ReportOutput")
        task = asyncio.create_task(
            run_research(first, run_id="run-resume", question=QUESTION, checkpointer=saver)
        )
        await asyncio.wait_for(held.reached.wait(), 10)
        task.cancel()  # the worker dies at the synthesizer
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert held.calls_for("PlanOutput") and "report" not in store.reports

        second = build_fake_runtime(web=web(), store=store)
        state = await run_research(second, run_id="run-resume", question=QUESTION, checkpointer=saver)

        assert state["report"]["question"] == QUESTION
        # Only the step that was cut short ran again.
        assert llm_of(second).calls_for("PlanOutput") == []
        assert llm_of(second).calls_for("SourceSelection") == []
        assert len(llm_of(second).calls_for("ReportOutput")) == 1
        # Every citation still resolves to a source the first attempt stored.
        grounding = check_grounding(state["report"], state["report_markdown"], set(store.sources))
        assert grounding.citation_resolution_rate == 1.0

    async def test_a_finished_run_is_not_run_again(self) -> None:
        saver, store = InMemorySaver(), InMemoryResearchStore()
        done = await run_research(
            build_fake_runtime(web=web(), store=store),
            run_id="run-done",
            question=QUESTION,
            checkpointer=saver,
        )

        again = build_fake_runtime(web=web(), store=store)
        state = await run_research(again, run_id="run-done", question=QUESTION, checkpointer=saver)

        assert llm_of(again).calls == []
        assert state["report"] == done["report"]

    async def test_a_replanned_round_reuses_its_sub_question_ids(self) -> None:
        """A planner re-run (after a resume) stores the same IDs, so no second set of rows."""
        store = InMemoryResearchStore()
        first = await run_research(
            build_fake_runtime(web=web(), store=store), run_id="run-ids", question=QUESTION
        )
        again = await run_research(
            build_fake_runtime(web=web(), store=store), run_id="run-ids", question=QUESTION
        )

        ids = [sq["id"] for sq in first["sub_questions"]]
        assert ids == [sq["id"] for sq in again["sub_questions"]]
        assert len(store.sub_questions) == len(ids)
