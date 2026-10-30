"""
M7: the three systems under comparison, on the fakes.

What matters is that each citation is resolved to the evidence that system
actually had — that is what the judge checks — and that the baselines are
what they claim to be (no retrieval; one query and one call).
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from researchmind_agents.config import AgentConfig, EvalConfig
from researchmind_agents.evals.benchmark import BenchmarkQuestion
from researchmind_agents.evals.offline import (
    OFFLINE_QUESTIONS,
    OfflineWeb,
    offline_config,
    offline_runtime_factory,
)
from researchmind_agents.evals.systems import (
    RAG,
    RESEARCHMIND,
    classify_page,
    rank_sources,
    report_statements,
    run_closed_book,
    run_rag,
    run_researchmind,
    system_config,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    CitationItem,
    ReportClaim,
    ReportEvidence,
    ReportSection,
    ReportStatement,
    ResearchReport,
)
from researchmind_agents.testing import ScriptedLLM, build_fake_runtime
from researchmind_agents.tools.page_extract import PageContent

QUESTION = BenchmarkQuestion(
    id="t1",
    question="How do heat pumps perform in cold climates?",
    mode="quick",
    user_context={"home": "1990s house"},
)
LONG = " Heat pumps move heat rather than burn fuel, so efficiency is quoted as a coefficient of performance." * 6


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    llm = runtime.llm
    assert isinstance(llm, ScriptedLLM)
    return llm


class TestClassifyPage:
    @pytest.mark.parametrize(
        "page, expected",
        [
            (PageContent(url="u", text="Readable text."), "resolved"),
            (PageContent(url="u", text="   "), "unreachable"),
            (PageContent(url="u", error="Client error '404 Not Found' for url 'https://a.example/x'"), "missing"),
            (PageContent(url="u", error="Client error '410 Gone' for url 'https://a.example/x'"), "missing"),
            # A 404 in the URL is not a 404 status.
            (PageContent(url="u", error="Client error '403 Forbidden' for url 'https://a.example/404'"), "unreachable"),
            (PageContent(url="u", error="DNS resolution failed for 'nope.example': [Errno 11001]"), "missing"),
            (PageContent(url="u", error="robots disallowed: robots.txt disallows https://a.example"), "unreachable"),
            (PageContent(url="u", error="ReadTimeout"), "unreachable"),
        ],
    )
    def test_status(self, page: PageContent, expected: str) -> None:
        assert classify_page(page)[0] == expected


class TestClosedBook:
    def runtime(self, statements: list[dict], config: AgentConfig | None = None) -> AgentRuntime:
        web = OfflineWeb(pages={"https://real.example/heat-pumps": ("Heat pumps in the cold", "Heat pumps work." + LONG)})
        runtime = build_fake_runtime(
            config, web=web, responses={"ClosedBookAnswer": json.dumps({"statements": statements})}
        )
        return runtime

    async def test_cited_urls_are_fetched_once_and_classified(self) -> None:
        runtime = self.runtime(
            [
                {"text": "Heat pumps work.", "sources": ["https://real.example/heat-pumps", "https://made-up.example/x"]},
                {"text": "Second  sentence.", "sources": ["https://real.example/heat-pumps/", "Wikipedia"]},
                {"text": "No source.", "sources": []},
                {"text": "  ", "sources": ["https://real.example/heat-pumps"]},
            ]
        )
        answer = await run_closed_book(runtime, QUESTION, "run-1")

        assert answer.retrieves is False and answer.sources == []
        assert [s.text for s in answer.statements] == ["Heat pumps work.", "Second sentence.", "No source."]
        assert [[c.status for c in s.citations] for s in answer.statements] == [
            ["resolved", "missing"],
            ["resolved", "missing"],
            [],
        ]
        resolved = answer.statements[0].citations[0]
        assert resolved.title == "Heat pumps in the cold" and "Heat pumps work." in resolved.evidence[0]
        assert answer.statements[1].citations[1].reason == "not a URL"
        # One fetch per distinct page, however it is spelled.
        web = runtime.page_extract._web  # type: ignore[attr-defined]
        assert web.fetched.count("https://real.example/heat-pumps") == 1
        prompt = llm_of(runtime).calls[0]["prompt"]
        assert QUESTION.question in prompt and "1990s house" in prompt

    async def test_urls_over_the_cap_are_not_fetched(self) -> None:
        config = AgentConfig(eval=EvalConfig(baseline_max_urls=1))
        runtime = self.runtime(
            [{"text": "A.", "sources": ["https://real.example/heat-pumps", "https://other.example/y"]}],
            config,
        )
        answer = await run_closed_book(runtime, QUESTION, "run-1")
        first, second = answer.statements[0].citations
        assert first.status == "resolved"
        assert second.status == "unreachable" and "EVAL_BASELINE_MAX_URLS" in second.reason


def heat_web(pages: int = 8) -> OfflineWeb:
    return OfflineWeb(
        pages={
            f"https://site{i}.example/heat-pumps": (
                f"Heat pumps cold climates report {i}",
                f"Heat pumps in cold climates finding {i}." + LONG,
            )
            for i in range(pages)
        }
    )


def rag_reply(*labels: str) -> str:
    return json.dumps(
        {"statements": [{"text": f"Claim from {label}.", "citations": [label]} for label in labels]}
    )


class TestRag:
    async def test_one_query_search_order_and_the_passages_it_was_shown(self) -> None:
        web = heat_web()
        config = system_config(AgentConfig(eval=EvalConfig(rag_pages=3)), RAG)
        runtime = build_fake_runtime(config, web=web, responses={"RagAnswer": rag_reply("S1", "[S2]", "S99")})

        answer = await run_rag(runtime, QUESTION, "run-1")

        assert web.queries == [QUESTION.question]
        assert len(answer.sources) == 3 and answer.retrieves
        llm = llm_of(runtime)
        assert llm.calls_for("SourceSelection") == [], "the RAG baseline reads pages in search order"
        assert len(llm.calls_for("RagAnswer")) == 1
        assert "Sources (labels: S1, S2, S3)" in llm.calls[0]["prompt"]
        first, second, unknown = (s.citations[0] for s in answer.statements)
        assert first.status == second.status == "resolved"
        assert first.url in web.pages and first.evidence and "Heat pumps" in first.evidence[0]
        assert (unknown.status, unknown.reason) == ("missing", "unknown label")

    async def test_no_pages_means_no_answer_and_no_model_call(self) -> None:
        runtime = build_fake_runtime(web=OfflineWeb())
        answer = await run_rag(runtime, QUESTION, "run-1")
        assert answer.statements == [] and answer.notes
        assert llm_of(runtime).calls == []

    async def test_sources_rank_by_similarity_to_the_question(self) -> None:
        web = heat_web(4)
        web.pages["https://exact.example/match"] = (
            "Cold climates",
            "How do heat pumps perform in cold climates? Heat pumps perform in cold climates well." + LONG,
        )
        runtime = build_fake_runtime(
            system_config(AgentConfig(eval=EvalConfig(rag_pages=5)), RAG),
            web=web,
            responses={"RagAnswer": rag_reply("S1")},
        )
        answer = await run_rag(runtime, QUESTION, "run-1")
        ranked = await rank_sources(runtime, run_id="run-1", question=QUESTION.question, sources=answer.sources)
        assert ranked[0] == "https://exact.example/match"
        assert sorted(ranked) == sorted(s.url for s in answer.sources)


class TestSystemConfig:
    def test_only_the_rag_baseline_reads_rag_pages(self) -> None:
        config = AgentConfig(eval=EvalConfig(rag_pages=7))
        assert system_config(config, RAG).research.pages_per_sub_question == 7
        assert system_config(config, RAG).research.max_fetch_attempts >= 14
        assert system_config(config, RESEARCHMIND) is config


class TestResearchMind:
    @pytest.mark.parametrize("question", OFFLINE_QUESTIONS, ids=lambda q: q.mode)
    async def test_every_citation_resolves_to_passages_of_the_run(self, question: BenchmarkQuestion) -> None:
        runtime = offline_runtime_factory(offline_config())
        answer = await run_researchmind(runtime, question, "run-1")

        assert answer.generation == "llm" and answer.statements and answer.sources
        stored = {p["text"] for p in runtime.store.passages.values()}  # type: ignore[attr-defined]
        for statement in answer.statements:
            assert statement.citations, statement.text
            for citation in statement.citations:
                assert citation.status == "resolved"
                assert citation.evidence and set(citation.evidence) <= stored
        assert answer.iterations >= 1

    def report(self) -> ResearchReport:
        return ResearchReport(
            title="T",
            question="Q",
            mode="deep",
            executive_summary=[ReportStatement(text="From claims.", citations=["S1"], claims=["C1"])],
            sections=[
                ReportSection(
                    heading="H",
                    statements=[
                        ReportStatement(text="From the source.", citations=["S1"]),
                        ReportStatement(text="Unknown label.", citations=["S7"]),
                    ],
                )
            ],
            recommendation=[ReportStatement(text="Choose A.", citations=["S1"], claims=["C1"])],
            sources=[CitationItem(label="S1", source_id="s1", url="https://a.example", passage_ids=["p1", "p2", "p3"])],
            key_findings=[
                ReportClaim(
                    label="C1",
                    claim_id="c1",
                    text="Claim.",
                    verdict="supported",
                    confidence=0.8,
                    citations=["S1"],
                    evidence=[
                        ReportEvidence(passage_id="p1", source_label="S1", stance="supports"),
                        ReportEvidence(passage_id="p2", source_label="S1", stance="neutral"),
                    ],
                )
            ],
        )

    def test_a_sentence_resting_on_claims_is_checked_against_their_passages(self) -> None:
        passages = {"p1": "one", "p2": "two", "p3": "three"}
        summary, from_source, unknown, recommendation = report_statements(self.report(), passages)
        assert summary.citations[0].evidence == ["one"]
        assert recommendation.citations[0].evidence == ["one"]
        assert from_source.citations[0].evidence == ["one", "two", "three"]
        assert (unknown.citations[0].status, unknown.citations[0].reason) == ("missing", "unknown label")

    def test_cited_passages_missing_from_state_cannot_be_read(self) -> None:
        [first, *_] = report_statements(self.report(), {})
        assert first.citations[0].status == "unreachable"

    async def test_the_question_mode_and_context_reach_the_graph(self) -> None:
        question = replace(OFFLINE_QUESTIONS[3], id="x")
        runtime = offline_runtime_factory(offline_config())
        answer = await run_researchmind(runtime, question, "run-ctx")
        llm = llm_of(runtime)
        assert answer.statements
        assert llm.calls_for("DecisionReportOutput"), "decision mode writes a recommendation"
        assert "two developers" in llm.calls_for("DecisionReportOutput")[0]["prompt"]
