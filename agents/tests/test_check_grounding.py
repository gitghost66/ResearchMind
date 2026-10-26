"""
M3 tests — seed eval set and grounding checker.
"""

from __future__ import annotations

from researchmind_agents.evals.check_grounding import (
    GroundingResult,
    check_grounding,
    load_seed,
    summarize,
)
from researchmind_agents.report import render_markdown
from researchmind_agents.state import (
    SUPPORTED_MODES,
    CitationItem,
    ReportSection,
    ReportStatement,
    ResearchReport,
)


def report(*statements: tuple[str, list[str]], sources: dict[str, str] | None = None) -> ResearchReport:
    sources = sources if sources is not None else {"S1": "id-1", "S2": "id-2"}
    return ResearchReport(
        title="T",
        question="Q",
        mode="quick",
        executive_summary=[],
        sections=[
            ReportSection(
                heading="H",
                statements=[ReportStatement(text=t, citations=c) for t, c in statements],
            )
        ],
        sources=[
            CitationItem(label=label, source_id=sid, url=f"https://{sid}.example")
            for label, sid in sources.items()
        ],
    )


class TestSeedSet:
    def test_has_five_to_ten_unique_supported_questions(self) -> None:
        seed = load_seed()
        assert 5 <= len(seed) <= 10
        assert len({q.id for q in seed}) == len(seed)
        assert all(q.mode in SUPPORTED_MODES for q in seed)
        assert all(q.question.strip() and q.expected_subtopics for q in seed)


class TestCheckGrounding:
    def test_fully_grounded_report_passes(self) -> None:
        r = report(("a", ["S1"]), ("b", ["S1", "S2"]))
        result = check_grounding(r, render_markdown(r), {"id-1", "id-2"})

        # 3 markers in the markdown body + 3 labels on statements.
        assert result.citations_total == 6
        assert result.citation_resolution_rate == 1.0
        assert result.cited_statement_rate == 1.0
        assert result.passes()

    def test_citation_to_unstored_source_does_not_resolve(self) -> None:
        r = report(("a", ["S1"]), ("b", ["S2"]))
        result = check_grounding(r, render_markdown(r), {"id-1"})

        assert result.unresolved == ["S2"]
        assert result.citation_resolution_rate == 0.5
        assert result.statements_cited == 1
        assert not result.passes()

    def test_marker_without_a_listed_source_does_not_resolve(self) -> None:
        r = report(("a", ["S1"]))
        markdown = render_markdown(r).replace("a [S1]", "a [S1][S7]")
        result = check_grounding(r, markdown, {"id-1", "id-2"})
        assert "S7" in result.unresolved

    def test_uncited_statements_lower_the_cited_rate(self) -> None:
        r = report(("a", ["S1"]), ("b", []), ("c", []), ("d", ["S2"]), ("e", ["S1"]))
        result = check_grounding(r, render_markdown(r), {"id-1", "id-2"})
        assert result.cited_statement_rate == 0.6
        assert result.passes(min_cited=0.6)
        assert not result.passes(min_cited=0.8)

    def test_source_list_is_not_counted_as_citations(self) -> None:
        r = report(("a", ["S1"]))
        markdown = render_markdown(r) + "\n- [S2] appears only below the Sources heading\n"
        result = check_grounding(r, markdown, {"id-1"})
        assert result.unresolved == []

    def test_report_with_no_statements_fails(self) -> None:
        r = report(sources={})
        result = check_grounding(r, render_markdown(r), set())
        assert result.cited_statement_rate == 0.0
        assert not result.passes()

    def test_accepts_report_json(self) -> None:
        r = report(("a", ["S1"]))
        result = check_grounding(r.model_dump(mode="json"), render_markdown(r), {"id-1"})
        assert result.passes()


class TestSummarize:
    def test_renders_success_and_error_rows(self) -> None:
        grounding = GroundingResult(citations_total=4, citations_resolved=4, statements_total=5, statements_cited=4)
        table = summarize(
            [
                {"id": "seed-01", "grounding": grounding, "generation": "llm", "sources": 3, "seconds": 12.34, "tokens": 9000},
                {"id": "seed-02", "error": "TimeoutError: slow"},
            ]
        )
        lines = table.splitlines()
        assert len(lines) == 4
        assert "| seed-01 | llm | 3 | 4/4 (100%) | 4/5 (80%) | 0 | 9000 | 12.3 |" in table
        assert "error: TimeoutError: slow" in lines[3]

    def test_as_dict_includes_rates(self) -> None:
        data = GroundingResult(citations_total=2, citations_resolved=1, statements_total=4, statements_cited=2).as_dict()
        assert data["citation_resolution_rate"] == 0.5
        assert data["cited_statement_rate"] == 0.5
