"""
Seed eval: is every citation grounded in a stored source?

Two measures from the M3 plan:

* **citation resolution** — share of citations (``[S#]`` markers in the
  markdown body, plus labels on every statement in the JSON) that map to a
  listed source whose ID is in the store for that run. Target: 100%.
* **cited statements** — share of report statements (executive summary,
  sections, and a decision-mode recommendation) with at least one resolving
  citation. Target: >= 80%.

The report is the pipeline's real output and the source IDs come from the
store, so this checks the whole path end to end, not just the validator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from researchmind_agents.state import ResearchReport

__all__ = [
    "SEED_PATH",
    "SeedQuestion",
    "GroundingResult",
    "load_seed",
    "check_grounding",
    "summarize",
]

SEED_PATH = Path(__file__).with_name("seed.jsonl")

_MARKER = re.compile(r"\[(S\d+)\]")
_SOURCES_HEADING = re.compile(r"^## Sources\s*$", re.MULTILINE)


@dataclass(frozen=True)
class SeedQuestion:
    id: str
    question: str
    mode: str = "quick"
    expected_subtopics: tuple[str, ...] = ()


@dataclass
class GroundingResult:
    citations_total: int = 0
    citations_resolved: int = 0
    statements_total: int = 0
    statements_cited: int = 0
    dropped_statements: int = 0
    unresolved: list[str] = field(default_factory=list)

    @property
    def citation_resolution_rate(self) -> float:
        # No citations at all resolves vacuously; the cited-statement rate catches it.
        return self.citations_resolved / self.citations_total if self.citations_total else 1.0

    @property
    def cited_statement_rate(self) -> float:
        return self.statements_cited / self.statements_total if self.statements_total else 0.0

    def passes(self, *, min_resolution: float = 1.0, min_cited: float = 0.8) -> bool:
        return (
            self.citation_resolution_rate >= min_resolution
            and self.cited_statement_rate >= min_cited
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citation_resolution_rate"] = round(self.citation_resolution_rate, 4)
        data["cited_statement_rate"] = round(self.cited_statement_rate, 4)
        return data


def load_seed(path: Path | str | None = None) -> list[SeedQuestion]:
    questions = []
    for line in Path(path or SEED_PATH).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        questions.append(
            SeedQuestion(
                id=item["id"],
                question=item["question"],
                mode=item.get("mode", "quick"),
                expected_subtopics=tuple(item.get("expected_subtopics", ())),
            )
        )
    return questions


def check_grounding(
    report: ResearchReport | dict[str, Any],
    markdown: str,
    stored_source_ids: Iterable[str],
) -> GroundingResult:
    parsed = report if isinstance(report, ResearchReport) else ResearchReport.model_validate(report)
    stored = set(stored_source_ids)
    resolves = {c.label for c in parsed.sources if c.source_id in stored}

    result = GroundingResult(dropped_statements=parsed.dropped_statements)

    # The source list itself uses bare labels, but skip it to be safe.
    body = _SOURCES_HEADING.split(markdown, maxsplit=1)[0]
    citations = _MARKER.findall(body)
    statements = [
        *parsed.executive_summary,
        *(s for sec in parsed.sections for s in sec.statements),
        *parsed.recommendation,
    ]
    citations += [label for statement in statements for label in statement.citations]

    result.citations_total = len(citations)
    for label in citations:
        if label in resolves:
            result.citations_resolved += 1
        elif label not in result.unresolved:
            result.unresolved.append(label)

    result.statements_total = len(statements)
    result.statements_cited = sum(
        1 for statement in statements if any(label in resolves for label in statement.citations)
    )
    return result


def summarize(rows: Sequence[dict[str, Any]]) -> str:
    """
    Markdown table for seed-eval rows. Each row needs ``id``, ``grounding``
    (a :class:`GroundingResult`), ``generation``, ``sources``, ``seconds``,
    and ``tokens``; an ``error`` key marks a failed run.
    """
    lines = [
        "| id | generation | sources | citations resolved | statements cited | dropped | tokens | seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("error"):
            lines.append(f"| {row['id']} | error: {row['error'][:60]} | | | | | | |")
            continue
        g: GroundingResult = row["grounding"]
        lines.append(
            f"| {row['id']} | {row['generation']} | {row['sources']} "
            f"| {g.citations_resolved}/{g.citations_total} ({g.citation_resolution_rate:.0%}) "
            f"| {g.statements_cited}/{g.statements_total} ({g.cited_statement_rate:.0%}) "
            f"| {g.dropped_statements} | {row['tokens']} | {row['seconds']:.1f} |"
        )
    return "\n".join(lines)
