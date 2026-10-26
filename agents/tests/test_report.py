"""
M3 tests — citation checking and markdown rendering (researchmind_agents.report).
"""

from __future__ import annotations

import pytest

from researchmind_agents.report import (
    check_citations,
    cited_labels,
    clean_statement,
    enforce_citations,
    normalize_label,
    normalize_output,
    render_markdown,
)
from researchmind_agents.state import (
    CitationItem,
    ReportOutput,
    ReportSection,
    ReportStatement,
    ResearchReport,
)


def stmt(text: str, *citations: str) -> ReportStatement:
    return ReportStatement(text=text, citations=list(citations))


def output(summary: list[ReportStatement], *sections: ReportSection) -> ReportOutput:
    return ReportOutput(
        title="T", executive_summary=summary, sections=list(sections), limitations=[]
    )


class TestNormalizeLabel:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("S1", "S1"),
            ("s2", "S2"),
            ("[S3]", "S3"),
            (" S04 ", "S4"),
            ("(S5)", "S5"),
            ("S", None),
            ("X1", None),
            ("source 1", None),
            ("S1a", None),
        ],
    )
    def test_variants(self, raw: str, expected: str | None) -> None:
        assert normalize_label(raw) == expected


class TestCleanStatement:
    def test_moves_inline_markers_into_citations(self) -> None:
        cleaned = clean_statement(stmt("RAG is cheaper [S1].", "S2"))
        assert cleaned.text == "RAG is cheaper."
        assert cleaned.citations == ["S2", "S1"]

    def test_handles_grouped_and_repeated_markers(self) -> None:
        cleaned = clean_statement(stmt("A [S1, S2] and b [s3][S1]"))
        assert cleaned.text == "A and b"
        assert cleaned.citations == ["S1", "S2", "S3"]

    def test_keeps_unparseable_labels_verbatim_for_the_check(self) -> None:
        cleaned = clean_statement(stmt("Claim", "source one", "[S2]"))
        assert cleaned.citations == ["source one", "S2"]

    def test_normalize_output_cleans_every_statement(self) -> None:
        result = normalize_output(
            ReportOutput(
                title="  A   title ",
                executive_summary=[stmt("x [S1]")],
                sections=[ReportSection(heading=" H ", statements=[stmt("y", "s2")])],
                limitations=["  ", " one  limit "],
            )
        )
        assert result.title == "A title"
        assert result.executive_summary[0].citations == ["S1"]
        assert result.sections[0].heading == "H"
        assert result.sections[0].statements[0].citations == ["S2"]
        assert result.limitations == ["one limit"]


class TestCheckCitations:
    def test_clean_report_passes(self) -> None:
        check = check_citations(output([stmt("a", "S1")]), {"S1"})
        assert check.ok

    def test_reports_unknown_labels_once(self) -> None:
        report = output(
            [stmt("a", "S1", "S9")],
            ReportSection(heading="h", statements=[stmt("b", "S9")]),
        )
        check = check_citations(report, {"S1"})
        assert check.unknown_labels == ["S9"]
        assert not check.ok

    def test_reports_uncited_statements(self) -> None:
        check = check_citations(output([stmt("no source")]), {"S1"})
        assert check.uncited_statements == ["no source"]
        assert not check.ok


class TestEnforceCitations:
    def test_strips_unknown_labels_but_keeps_supported_statement(self) -> None:
        cleaned, dropped = enforce_citations(output([stmt("a", "S1", "S9")]), {"S1"})
        assert cleaned.executive_summary == [stmt("a", "S1")]
        assert dropped == 0

    def test_drops_statements_left_uncited_and_empty_sections(self) -> None:
        report = output(
            [stmt("kept", "S1"), stmt("invented", "S7")],
            ReportSection(heading="empty after", statements=[stmt("x")]),
            ReportSection(heading="stays", statements=[stmt("y", "S2")]),
        )
        cleaned, dropped = enforce_citations(report, {"S1", "S2"})
        assert [s.text for s in cleaned.executive_summary] == ["kept"]
        assert [s.heading for s in cleaned.sections] == ["stays"]
        assert dropped == 2

    def test_cited_labels_sorted_numerically(self) -> None:
        report = output([stmt("a", "S10"), stmt("b", "S2")], ReportSection(heading="h", statements=[stmt("c", "S2", "S1")]))
        assert cited_labels(report) == ["S1", "S2", "S10"]


def make_report(**overrides: object) -> ResearchReport:
    values: dict[str, object] = {
        "title": "Title",
        "question": "Q?",
        "mode": "quick",
        "executive_summary": [stmt("Summary.", "S1")],
        "sections": [ReportSection(heading="Costs", statements=[stmt("Cheap.", "S1", "S2")])],
        "limitations": ["Limited."],
        "methodology": "Method.",
        "sources": [
            CitationItem(label="S1", source_id="id-1", url="https://a.example", title="A [draft]"),
            CitationItem(label="S2", source_id="id-2", url="https://b.example", title=""),
        ],
    }
    values.update(overrides)
    return ResearchReport.model_validate(values)


class TestRenderMarkdown:
    def test_body_has_markers_and_sources_use_bare_labels(self) -> None:
        markdown = render_markdown(make_report())
        body, sources = markdown.split("## Sources")
        assert "Summary. [S1]" in body
        assert "Cheap. [S1][S2]" in body
        assert "[S1]" not in sources
        assert "- **S1**: [A (draft)](https://a.example)" in sources
        # A source without a title falls back to its URL.
        assert "- **S2**: [https://b.example](https://b.example)" in sources

    def test_sections_appear_in_order(self) -> None:
        markdown = render_markdown(make_report())
        order = [markdown.index(h) for h in ("# Title", "## Executive summary", "## Costs", "## Limitations", "## Methodology", "## Sources")]
        assert order == sorted(order)

    def test_empty_report_says_so(self) -> None:
        markdown = render_markdown(make_report(executive_summary=[], sections=[], sources=[]))
        assert "No findings could be grounded" in markdown
        assert "_None._" in markdown
