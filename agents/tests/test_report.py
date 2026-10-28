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
    label_sort_key,
    normalize_label,
    normalize_output,
    render_markdown,
)
from researchmind_agents.state import (
    CitationItem,
    ReportClaim,
    ReportContradiction,
    ReportEvidence,
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


# ===========================================================================
# M5: claim labels and verified reports
# ===========================================================================


class TestClaimLabels:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("C3", "C3"), ("[c3]", "C3"), (" (C07) ", "C7"), ("S3", None), ("C", None), ("claim 3", None)],
    )
    def test_normalize_with_claim_prefix(self, raw: str, expected: str | None) -> None:
        assert normalize_label(raw, prefix="C") == expected

    def test_source_prefix_is_the_default_and_rejects_claims(self) -> None:
        assert normalize_label("s2") == "S2"
        assert normalize_label("C2") is None

    @pytest.mark.parametrize(("label", "key"), [("S1", 1), ("C12", 12), ("c3", 3), ("bogus", 10**9)])
    def test_label_sort_key_reads_any_prefix(self, label: str, key: int) -> None:
        assert label_sort_key(label) == key

    def test_clean_statement_moves_inline_claim_markers(self) -> None:
        cleaned = clean_statement(stmt("Output drops in the cold [C1, c2][C3].", "C1", "[c4]"), prefix="C")
        assert cleaned.text == "Output drops in the cold."
        assert cleaned.citations == ["C1", "C4", "C2", "C3"]

    def test_clean_statement_with_claim_prefix_leaves_source_markers_alone(self) -> None:
        cleaned = clean_statement(stmt("Cheap [S1].", "C1"), prefix="C")
        assert cleaned.text == "Cheap [S1]."
        assert cleaned.citations == ["C1"]

    def test_normalize_output_with_claim_prefix(self) -> None:
        normalized = normalize_output(
            output(
                [stmt("Summary [C2].")],
                ReportSection(heading="  Costs  ", statements=[stmt("Cheap.", "[c1]", "S1")]),
            ),
            prefix="C",
        )
        assert normalized.executive_summary[0].citations == ["C2"]
        assert normalized.sections[0].heading == "Costs"
        # An S label is not a claim label: kept verbatim, so the check reports it.
        assert normalized.sections[0].statements[0].citations == ["C1", "S1"]
        check = check_citations(normalized, {"C1", "C2"})
        assert check.unknown_labels == ["S1"]

    def test_enforce_citations_keeps_claim_labels(self) -> None:
        statement = ReportStatement(text="Cheap.", citations=["S1", "S9"], claims=["C1", "C2"])
        cleaned, dropped = enforce_citations(output([statement]), {"S1"})
        assert dropped == 0
        kept = cleaned.executive_summary[0]
        assert kept.citations == ["S1"]
        assert kept.claims == ["C1", "C2"]


def claim(label: str, verdict: str, confidence: float, *citations: str) -> ReportClaim:
    return ReportClaim(
        label=label,
        claim_id=f"id-{label}",
        text=f"Claim {label} text.",
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        citations=list(citations),
        evidence=[ReportEvidence(passage_id=f"p-{label}", source_label=c, stance="supports") for c in citations],
    )


def verified_report(**overrides: object) -> ResearchReport:
    values: dict[str, object] = {
        "mode": "deep",
        "key_findings": [
            claim("C1", "supported", 0.8234, "S1", "S2"),
            claim("C2", "weak", 0.3, "S2"),
            claim("C3", "mixed", 0.12, "S1", "S2"),
        ],
        "verification": {"claims_checked": 5},
        "overall_confidence": 0.6789,
    }
    values.update(overrides)
    return make_report(**values)


class TestVerifiedMarkdown:
    def test_overall_confidence_line(self) -> None:
        markdown = render_markdown(verified_report())
        assert "> **Overall confidence:** 0.68 (from 3 verified claims; 5 checked)" in markdown
        assert markdown.index("**Question:**") < markdown.index("**Overall confidence:**")

    def test_checked_count_defaults_to_the_findings(self) -> None:
        markdown = render_markdown(verified_report(verification={}))
        assert "(from 3 verified claims; 3 checked)" in markdown

    def test_key_findings_show_verdict_confidence_and_sources(self) -> None:
        markdown = render_markdown(verified_report())
        assert "## Key findings" in markdown
        assert "- **C1** Claim C1 text. [S1][S2] _(supported; confidence 0.82)_" in markdown
        assert "- **C2** Claim C2 text. [S2] _(weakly supported; confidence 0.30)_" in markdown
        assert "- **C3** Claim C3 text. [S1][S2] _(sources disagree; confidence 0.12)_" in markdown

    def test_key_findings_sit_between_the_body_and_the_limitations(self) -> None:
        markdown = render_markdown(verified_report())
        order = [markdown.index(h) for h in ("## Costs", "## Key findings", "## Limitations", "## Sources")]
        assert order == sorted(order)

    def test_key_finding_markers_are_body_citations(self) -> None:
        body, sources = render_markdown(verified_report()).split("## Sources")
        assert "[S1][S2] _(supported" in body
        assert "[S1]" not in sources

    def test_a_finding_without_citations_has_no_double_space(self) -> None:
        markdown = render_markdown(verified_report(key_findings=[claim("C1", "supported", 0.5)]))
        assert "- **C1** Claim C1 text. _(supported; confidence 0.50)_" in markdown

    def test_zero_confidence_is_still_shown(self) -> None:
        markdown = render_markdown(verified_report(overall_confidence=0.0))
        assert "**Overall confidence:** 0.00" in markdown

    def test_quick_report_has_neither(self) -> None:
        markdown = render_markdown(make_report())
        assert "Overall confidence" not in markdown
        assert "## Key findings" not in markdown

    def test_verified_report_json_round_trips(self) -> None:
        report = verified_report()
        restored = ResearchReport.model_validate(report.model_dump(mode="json"))
        assert restored == report
        assert restored.key_findings[0].evidence[0].source_label == "S1"


# ===========================================================================
# M6: recommendation and contradictory evidence
# ===========================================================================


class TestRecommendation:
    def test_recommendation_statements_are_checked_and_enforced(self) -> None:
        out = output([stmt("Summary.", "S1")])
        out.recommendation = [stmt("Pick A.", "S1"), stmt("Pick B.", "S9"), stmt("Unsure.")]

        check = check_citations(out, {"S1"})
        assert check.unknown_labels == ["S9"]
        assert check.uncited_statements == ["Unsure."]

        cleaned, dropped = enforce_citations(out, {"S1"})
        assert [s.text for s in cleaned.recommendation] == ["Pick A."]
        assert dropped == 2
        assert cited_labels(cleaned) == ["S1"]

    def test_inline_markers_are_moved_out_of_recommendations(self) -> None:
        out = output([])
        out.recommendation = [stmt("Pick A [S2].")]
        assert normalize_output(out).recommendation[0].citations == ["S2"]

    def test_rendered_after_the_body(self) -> None:
        markdown = render_markdown(
            make_report(mode="decision", recommendation=[stmt("Choose PostgreSQL.", "S1")])
        )
        assert "## Recommendation\n\nChoose PostgreSQL. [S1]" in markdown
        assert markdown.index("## Costs") < markdown.index("## Recommendation") < markdown.index("## Sources")

    def test_decision_report_without_one_says_so(self) -> None:
        markdown = render_markdown(make_report(mode="decision"))
        assert "_No recommendation could be grounded in the verified claims._" in markdown
        assert "## Recommendation" not in render_markdown(make_report(mode="deep"))


def contradiction(label: str, resolved: bool, cause: str = "date") -> ReportContradiction:
    return ReportContradiction(
        label=label,
        contradiction_id=f"id-{label}",
        claims=["C1", "C3"],
        citations=["S1", "S2"],
        likely_cause=cause,
        explanation="The 2021 rating and the 2025 trial differ.",
        resolved=resolved,
    )


class TestContradictoryEvidence:
    def test_listed_with_claims_cause_and_status(self) -> None:
        markdown = render_markdown(
            verified_report(contradictory_evidence=[contradiction("X1", True), contradiction("X2", False, "unexplained")])
        )
        assert "## Contradictory evidence" in markdown
        assert (
            "- **X1** C1 vs C3: The 2021 rating and the 2025 trial differ. [S1][S2] "
            "_(likely cause: date; resolved)_"
        ) in markdown
        assert "_(likely cause: unexplained; unresolved)_" in markdown
        order = [markdown.index(h) for h in ("## Key findings", "## Contradictory evidence", "## Limitations")]
        assert order == sorted(order)

    def test_absent_without_contradictions(self) -> None:
        assert "Contradictory evidence" not in render_markdown(verified_report())

    def test_round_trips(self) -> None:
        report = verified_report(contradictory_evidence=[contradiction("X1", False)])
        assert ResearchReport.model_validate(report.model_dump(mode="json")) == report
