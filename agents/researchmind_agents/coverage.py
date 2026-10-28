"""
Sub-question coverage: how much verified evidence each part of the plan has.

Deterministic, like ``reliability.py``. Used by the sufficiency check (is the
run done?), the gap planner (what is missing?), and the synthesizer (the
report's coverage and confidence).

* A *usable* claim has a usable verdict (supported, weak, mixed), clears the
  minimum confidence, and has at least one supporting passage.
* Gap-filling sub-questions (``parent_id`` set) count toward the top-level
  sub-question they fill, its *root*. Coverage is reported per root, in plan
  order.
* A root is *covered* with at least ``min_claims`` usable claims, supported by
  at least ``min_sites`` independent sites (hosts without ``www.``) in total.
* Roots from the first plan are *required*. A later round's root (filling a
  gap the model named, not a sub-question's) is optional: it counts once it
  has evidence, so a gap round that finds nothing never makes a run look
  less complete than it was (:func:`report_roots`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from researchmind_agents.state import USABLE_VERDICTS, ClaimItem, SubQuestionItem
from researchmind_agents.tools.urls import domain_of

__all__ = [
    "SubQuestionCoverage",
    "root_ids",
    "roots",
    "report_roots",
    "rounds_run",
    "is_usable",
    "assess_coverage",
]


@dataclass(frozen=True)
class SubQuestionCoverage:
    """Verified evidence for one top-level sub-question and its follow-ups."""

    sub_question: SubQuestionItem
    # 1-based plan position among the roots.
    position: int
    claims: int
    sites: int
    covered: bool
    # Usable claims, best first.
    claim_ids: tuple[str, ...]
    # Queries searched for the root and its follow-ups.
    queries: tuple[str, ...]
    # From the first plan; an optional root only counts once it has evidence.
    required: bool = True


def root_ids(sub_questions: Iterable[SubQuestionItem]) -> dict[str, str]:
    """Each sub-question's ID -> the ID of the top-level sub-question it belongs to."""
    by_id = {sq.id: sq for sq in sub_questions}
    result: dict[str, str] = {}
    for sq in by_id.values():
        current, seen = sq, {sq.id}
        while current.parent_id and current.parent_id in by_id and current.parent_id not in seen:
            seen.add(current.parent_id)
            current = by_id[current.parent_id]
        result[sq.id] = current.id
    return result


def roots(sub_questions: Sequence[SubQuestionItem]) -> list[SubQuestionItem]:
    """Top-level sub-questions, in plan order."""
    mapping = root_ids(sub_questions)
    return [sq for sq in sub_questions if mapping[sq.id] == sq.id]


def report_roots(
    sub_questions: Sequence[SubQuestionItem], with_evidence: set[str]
) -> list[SubQuestionItem]:
    """
    The roots a report covers, in plan order: every required root, plus the
    optional ones in *with_evidence* (root IDs with at least one usable claim).
    """
    return [sq for sq in roots(sub_questions) if sq.iteration == 0 or sq.id in with_evidence]


def rounds_run(sub_questions: Iterable[SubQuestionItem]) -> int:
    """Research rounds that researched something: a gap round that added nothing does not count."""
    return 1 + max((sq.iteration for sq in sub_questions), default=0)


def is_usable(claim: ClaimItem, *, min_confidence: float = 0.0) -> bool:
    return (
        claim.verdict in USABLE_VERDICTS
        and claim.confidence >= min_confidence
        and any(e.stance == "supports" for e in claim.evidence)
    )


def assess_coverage(
    sub_questions: Sequence[SubQuestionItem],
    claims: Iterable[ClaimItem],
    source_urls: Mapping[str, str],
    *,
    min_claims: int,
    min_sites: int,
    min_confidence: float = 0.0,
) -> list[SubQuestionCoverage]:
    """
    Coverage per root, in plan order. *source_urls* maps source IDs to URLs;
    evidence from a source not in it does not count.
    """
    mapping = root_ids(sub_questions)
    by_root: dict[str, list[ClaimItem]] = {}
    for claim in claims:
        root_id = mapping.get(claim.sub_question_id or "")
        if root_id is not None and is_usable(claim, min_confidence=min_confidence):
            by_root.setdefault(root_id, []).append(claim)

    queries: dict[str, list[str]] = {}
    for sq in sub_questions:
        bucket = queries.setdefault(mapping[sq.id], [])
        for query in sq.queries:
            if query not in bucket:
                bucket.append(query)

    result = []
    for position, root in enumerate(roots(sub_questions), start=1):
        members = sorted(by_root.get(root.id, []), key=lambda c: -c.confidence)
        sites = {
            domain_of(source_urls[e.source_id])
            for claim in members
            for e in claim.evidence
            if e.stance == "supports" and e.source_id in source_urls
        }
        sites.discard("")
        result.append(
            SubQuestionCoverage(
                sub_question=root,
                position=position,
                claims=len(members),
                sites=len(sites),
                covered=len(members) >= min_claims and len(sites) >= min_sites,
                claim_ids=tuple(c.id for c in members),
                queries=tuple(queries.get(root.id, [])),
                required=root.iteration == 0,
            )
        )
    return result
