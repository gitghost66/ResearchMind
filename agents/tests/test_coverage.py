"""M6 tests — sub-question coverage (researchmind_agents.coverage)."""

from __future__ import annotations

from researchmind_agents.coverage import assess_coverage, is_usable, root_ids, roots
from researchmind_agents.state import ClaimItem, EvidenceItem, SubQuestionItem

URLS = {
    "s1": "https://www.a.example/1",
    "s2": "https://a.example/2",
    "s3": "https://b.example/3",
    "s4": "https://c.example/4",
}


def sq(sid: str, parent: str | None = None, queries: tuple[str, ...] = ()) -> SubQuestionItem:
    return SubQuestionItem(id=sid, text=f"question {sid}", queries=list(queries), parent_id=parent)


def claim(
    cid: str, sub_question: str | None, *sources: str, verdict: str = "supported", confidence: float = 0.5
) -> ClaimItem:
    return ClaimItem(
        id=cid,
        text=f"claim {cid}",
        sub_question_id=sub_question,
        passage_ids=[f"p-{cid}"],
        evidence=[EvidenceItem(passage_id=f"p-{cid}-{s}", source_id=s) for s in sources],
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
    )


class TestRoots:
    def test_follow_ups_belong_to_their_top_level_sub_question(self) -> None:
        items = [sq("a"), sq("b"), sq("a1", "a"), sq("a2", "a1")]
        assert root_ids(items) == {"a": "a", "b": "b", "a1": "a", "a2": "a"}
        assert [item.id for item in roots(items)] == ["a", "b"]

    def test_an_unknown_or_cyclic_parent_makes_a_root(self) -> None:
        items = [sq("x", "missing"), sq("y", "z"), sq("z", "y")]
        mapping = root_ids(items)
        assert mapping["x"] == "x"
        assert mapping["y"] in {"y", "z"} and mapping["z"] in {"y", "z"}


class TestUsable:
    def test_usable_verdicts_with_support_only(self) -> None:
        assert is_usable(claim("c", "a", "s1"))
        assert is_usable(claim("c", "a", "s1", verdict="weak"))
        assert not is_usable(claim("c", "a", "s1", verdict="unsupported"))
        assert not is_usable(claim("c", "a", "s1", verdict="unverified"))
        assert not is_usable(claim("c", "a"))  # no evidence
        assert not is_usable(claim("c", "a", "s1", confidence=0.1), min_confidence=0.2)


class TestAssessCoverage:
    def test_claims_and_independent_sites_per_sub_question(self) -> None:
        items = [sq("a", queries=("q1",)), sq("b")]
        claims = [
            claim("c1", "a", "s1"),
            claim("c2", "a", "s2"),  # same site as s1 (www. ignored)
            claim("c3", "b", "s3"),
            claim("c4", "b", "s4"),
        ]
        a, b = assess_coverage(items, claims, URLS, min_claims=2, min_sites=2)
        assert (a.claims, a.sites, a.covered) == (2, 1, False)
        assert (b.claims, b.sites, b.covered) == (2, 2, True)
        assert (a.position, b.position) == (1, 2)
        assert a.queries == ("q1",)

    def test_follow_up_claims_and_queries_count_toward_the_root(self) -> None:
        items = [sq("a", queries=("q1",)), sq("a1", "a", queries=("q2", "q1"))]
        claims = [claim("c1", "a", "s1", confidence=0.4), claim("c2", "a1", "s3", confidence=0.9)]
        [a] = assess_coverage(items, claims, URLS, min_claims=2, min_sites=2)
        assert a.covered
        assert a.claim_ids == ("c2", "c1"), "best first"
        assert a.queries == ("q1", "q2")

    def test_unusable_claims_and_unknown_sources_do_not_count(self) -> None:
        items = [sq("a")]
        claims = [
            claim("c1", "a", "s1", verdict="unsupported"),
            claim("c2", "a", "nope"),
            claim("c3", None, "s3"),
        ]
        [a] = assess_coverage(items, claims, URLS, min_claims=1, min_sites=1)
        assert a.claims == 1 and a.sites == 0 and not a.covered
