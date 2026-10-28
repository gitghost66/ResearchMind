"""
Report helpers: citation checking and markdown rendering.

Grounding is enforced by construction:

* the model cites short labels (``S1``, ``S2``, ...), never IDs or URLs;
* :func:`check_citations` finds unknown labels and uncited statements;
* :func:`enforce_citations` strips unknown labels and drops any statement left
  without a valid citation;
* :func:`render_markdown` writes the markdown from the validated structure, so
  every ``[S#]`` marker in it maps to a listed, stored source.

In verified modes the model cites claim labels (``C1``, ``C2``, ...) instead;
the same helpers run with ``prefix="C"``, and the synthesizer then derives
each statement's source labels from its claims. A decision-mode
recommendation is checked like any other statements.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from researchmind_agents.state import (
    ReportOutput,
    ReportSection,
    ReportStatement,
    ResearchReport,
)

__all__ = [
    "normalize_label",
    "clean_statement",
    "normalize_output",
    "CitationCheck",
    "check_citations",
    "enforce_citations",
    "cited_labels",
    "label_sort_key",
    "render_markdown",
]

_ANY_LABEL = re.compile(r"^([A-Z])(\d+)$", re.IGNORECASE)


def _inline_markers(prefix: str) -> re.Pattern[str]:
    # "[S1]", "[S1, S2]", "[S1][S3]" written into the sentence itself.
    p = re.escape(prefix)
    return re.compile(rf"\s*\[\s*({p}\d+(?:\s*[,;]\s*{p}\d+)*)\s*\]", re.IGNORECASE)


def normalize_label(raw: str, prefix: str = "S") -> str | None:
    """``"[s3]"`` / ``" S3 "`` -> ``"S3"``; anything not shaped like a label -> ``None``."""
    match = _ANY_LABEL.match(raw.strip().strip("[]()").strip())
    if not match or match.group(1).upper() != prefix.upper():
        return None
    return f"{prefix.upper()}{int(match.group(2))}"


def label_sort_key(label: str) -> int:
    match = _ANY_LABEL.match(label)
    return int(match.group(2)) if match else 10**9


def clean_statement(statement: ReportStatement, prefix: str = "S") -> ReportStatement:
    """
    Move citation markers the model wrote inline into ``citations`` and
    normalise every label. Unparseable labels are kept verbatim so the check
    can report them as unknown.
    """
    inline: list[str] = []

    def collect(match: re.Match[str]) -> str:
        inline.extend(part.strip() for part in re.split(r"[,;]", match.group(1)))
        return ""

    text = _inline_markers(prefix).sub(collect, statement.text).strip()
    labels: list[str] = []
    for raw in [*statement.citations, *inline]:
        label = normalize_label(raw, prefix) or raw.strip()
        if label and label not in labels:
            labels.append(label)
    return ReportStatement(text=text, citations=labels)


def normalize_output(output: ReportOutput, prefix: str = "S") -> ReportOutput:
    return ReportOutput(
        title=" ".join(output.title.split()),
        executive_summary=[clean_statement(s, prefix) for s in output.executive_summary],
        sections=[
            ReportSection(
                heading=" ".join(section.heading.split()),
                statements=[clean_statement(s, prefix) for s in section.statements],
            )
            for section in output.sections
        ],
        limitations=[" ".join(item.split()) for item in output.limitations if item.strip()],
        recommendation=[clean_statement(s, prefix) for s in output.recommendation],
    )


@dataclass
class CitationCheck:
    unknown_labels: list[str] = field(default_factory=list)
    uncited_statements: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unknown_labels and not self.uncited_statements


def _statements(output: ReportOutput) -> Iterable[ReportStatement]:
    yield from output.executive_summary
    for section in output.sections:
        yield from section.statements
    yield from output.recommendation


def check_citations(output: ReportOutput, allowed: set[str]) -> CitationCheck:
    check = CitationCheck()
    for statement in _statements(output):
        if not statement.text:
            continue
        if not statement.citations:
            check.uncited_statements.append(statement.text)
        for label in statement.citations:
            if label not in allowed and label not in check.unknown_labels:
                check.unknown_labels.append(label)
    return check


def enforce_citations(output: ReportOutput, allowed: set[str]) -> tuple[ReportOutput, int]:
    """
    Keep only valid citations; drop statements left with none and sections
    left empty. Returns the cleaned output and the number of dropped statements.
    """
    dropped = 0

    def keep(statements: list[ReportStatement]) -> list[ReportStatement]:
        nonlocal dropped
        kept = []
        for statement in statements:
            labels = [label for label in statement.citations if label in allowed]
            if statement.text and labels:
                kept.append(
                    ReportStatement(text=statement.text, citations=labels, claims=statement.claims)
                )
            elif statement.text:
                dropped += 1
        return kept

    sections = []
    for section in output.sections:
        statements = keep(section.statements)
        if statements:
            sections.append(ReportSection(heading=section.heading, statements=statements))

    cleaned = ReportOutput(
        title=output.title,
        executive_summary=keep(output.executive_summary),
        sections=sections,
        limitations=output.limitations,
        recommendation=keep(output.recommendation),
    )
    return cleaned, dropped


def cited_labels(output: ReportOutput) -> list[str]:
    """Distinct labels cited anywhere, in label order."""
    labels = {label for statement in _statements(output) for label in statement.citations}
    return sorted(labels, key=label_sort_key)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _cite(statement: ReportStatement) -> str:
    markers = "".join(f"[{label}]" for label in statement.citations)
    return f"{statement.text} {markers}".strip()


_VERDICT_TEXT = {
    "supported": "supported",
    "weak": "weakly supported",
    "mixed": "sources disagree",
    "unsupported": "unsupported",
    "unverified": "not verified",
}


def render_markdown(report: ResearchReport) -> str:
    """
    Render the report. Citation markers appear only as ``[S#]`` in the body;
    the source list uses bare labels so it is never mistaken for a citation.
    """
    lines = [f"# {report.title}", "", f"> **Question:** {report.question}"]
    if report.overall_confidence is not None:
        checked = int(report.verification.get("claims_checked", len(report.key_findings)))
        lines += [
            ">",
            f"> **Overall confidence:** {report.overall_confidence:.2f} "
            f"(from {len(report.key_findings)} verified claims; {checked} checked)",
        ]
    lines.append("")

    if report.executive_summary:
        lines += ["## Executive summary", ""]
        lines += [" ".join(_cite(s) for s in report.executive_summary), ""]

    for section in report.sections:
        lines += [f"## {section.heading}", ""]
        lines += [" ".join(_cite(s) for s in section.statements), ""]

    if not report.executive_summary and not report.sections:
        lines += ["_No findings could be grounded in the sources that were found._", ""]

    if report.mode == "decision" or report.recommendation:
        lines += ["## Recommendation", ""]
        if report.recommendation:
            lines += [" ".join(_cite(s) for s in report.recommendation), ""]
        else:
            lines += ["_No recommendation could be grounded in the verified claims._", ""]

    if report.key_findings:
        lines += ["## Key findings", ""]
        for claim in report.key_findings:
            markers = "".join(f"[{label}]" for label in claim.citations)
            verdict = _VERDICT_TEXT.get(claim.verdict, claim.verdict)
            lines.append(
                f"- **{claim.label}** {claim.text} {markers} "
                f"_({verdict}; confidence {claim.confidence:.2f})_".replace("  ", " ")
            )
        lines.append("")

    if report.contradictory_evidence:
        lines += ["## Contradictory evidence", ""]
        for item in report.contradictory_evidence:
            markers = "".join(f"[{label}]" for label in item.citations)
            status = "resolved" if item.resolved else "unresolved"
            cause = item.likely_cause.replace("_", " ")
            lines.append(
                f"- **{item.label}** {' vs '.join(item.claims)}: {item.explanation} {markers} "
                f"_(likely cause: {cause}; {status})_".replace("  ", " ")
            )
        lines.append("")

    if report.limitations:
        lines += ["## Limitations", ""]
        lines += [f"- {item}" for item in report.limitations]
        lines.append("")

    if report.methodology:
        lines += ["## Methodology", "", report.methodology, ""]

    lines += ["## Sources", ""]
    if report.sources:
        for citation in report.sources:
            title = citation.title or citation.url
            lines.append(f"- **{citation.label}**: [{_escape_link_text(title)}]({citation.url})")
    else:
        lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


def _escape_link_text(text: str) -> str:
    return text.replace("[", "(").replace("]", ")")
