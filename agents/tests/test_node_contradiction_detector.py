"""
M6 tests — contradiction detector node.

Candidate pairs and topics are deterministic (similarity, fact-checker links,
caps); the model only says which claims of a topic conflict and why. The
properties under test: only usable claims of this round are compared, the
model's labels must name two or more claims of one topic, ``resolved`` comes
from the cause and never from the model, and records are stored, streamed,
and added to earlier rounds' records.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from researchmind_agents.config import AgentConfig, ContradictionConfig
from researchmind_agents.nodes.contradiction_detector import (
    CandidatePair,
    find_candidates,
    group_pairs,
    make_contradiction_detector_node,
)
from researchmind_agents.prompts.contradiction_detector import PROMPT_VERSION
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    ClaimItem,
    ContradictionItem,
    EvidenceItem,
    PassageItem,
    ResearchState,
    SourceItem,
    SubQuestionItem,
    new_research_state,
    stable_id,
)
from researchmind_agents.storage import InMemoryResearchStore
from researchmind_agents.testing import ScriptedLLM, build_fake_runtime

RUN = "run-contra"

LAB = "The Thermo X200 heat pump reaches a coefficient of performance of 3.1 at minus fifteen degrees."
FIELD = "The Thermo X200 heat pump reaches a coefficient of performance of 2.2 at minus fifteen degrees."
OTHER = "Gas furnaces need yearly servicing by a certified technician."

# (url, published_at, domain_type)
LAB_PAGE = ("https://thermo.example/datasheet", "2021-02-01T00:00:00+00:00", "official_docs")
FIELD_PAGE = ("https://energy-agency.example/field-trial", "2025-06-01T00:00:00+00:00", "government")
OTHER_PAGE = ("https://furnace.example/service", None, "unknown")


async def seed(
    runtime: AgentRuntime,
    claims: list[tuple[str, tuple[str, str | None, str]]],
    *,
    verdict: str = "supported",
    iteration: int = 0,
) -> ResearchState:
    """One source and passage per claim; claims are stored as the fact checker leaves them."""
    sub_question = SubQuestionItem(id=str(uuid.uuid4()), text="How do heat pumps cope with cold?")
    await runtime.store.save_plan(run_id=RUN, plan={}, sub_questions=[sub_question])
    sources, passages, items = [], [], []
    for text, (url, published, kind) in claims:
        source = SourceItem(
            id=str(uuid.uuid4()),
            url=url,
            title=f"Page at {url}",
            domain_type=kind,
            reliability_score=0.7,
            published_at=published,
        )
        await runtime.store.save_source(run_id=RUN, source=source)
        passage = PassageItem(
            id=str(uuid.uuid4()),
            source_id=source.id,
            sub_question_id=sub_question.id,
            text=f"{text} More context about the measurement follows here.",
        )
        await runtime.store.save_passages(
            run_id=RUN, passages=[passage], embeddings=[None], embedding_model="test"
        )
        items.append(
            ClaimItem(
                id=str(uuid.uuid4()),
                text=text,
                sub_question_id=sub_question.id,
                passage_ids=[passage.id],
                evidence=[
                    EvidenceItem(
                        passage_id=passage.id,
                        source_id=source.id,
                        stance="supports",
                        excerpt=passage.text,
                    )
                ],
                verdict=verdict,  # type: ignore[arg-type]
                confidence=0.6,
                iteration=iteration,
            )
        )
        sources.append(source.model_dump())
        passages.append(passage.model_dump())
    await runtime.store.save_claims(run_id=RUN, claims=items)

    state = new_research_state(run_id=RUN, question="Do heat pumps work in the cold?", mode="deep")
    state["sub_questions"] = [sub_question.model_dump()]
    state["sources"] = sources
    state["passages"] = passages
    state["claims"] = [c.model_dump() for c in items]
    state["iteration"] = iteration
    return state


def responder(*conflicts: tuple[list[str], str]) -> Callable[[str], str]:
    """Report each (claim texts, cause) as a contradiction, by the labels the prompt gives them."""

    def respond(prompt: str) -> str:
        labels = {m.group(2): m.group(1) for m in re.finditer(r"^\[(C\d+)\] \(\w+\) (.*)$", prompt, re.M)}
        found = []
        for texts, cause in conflicts:
            if all(text in labels for text in texts):
                found.append(
                    {
                        "claims": [labels[text] for text in texts],
                        "likely_cause": cause,
                        "explanation": "The datasheet figure is from 2021; the 2025 field trial measured less.",
                    }
                )
        return json.dumps({"contradictions": found})

    return respond


def store_of(runtime: AgentRuntime) -> InMemoryResearchStore:
    assert isinstance(runtime.store, InMemoryResearchStore)
    return runtime.store


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    assert isinstance(runtime.llm, ScriptedLLM)
    return runtime.llm


def events(result: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in result["events"] if e["type"] == kind]


# ---------------------------------------------------------------------------
# Candidates and topics
# ---------------------------------------------------------------------------


def claim(cid: str, *passages: str, contradicts: tuple[str, ...] = (), iteration: int = 0) -> ClaimItem:
    return ClaimItem(
        id=cid,
        text=f"claim {cid}",
        passage_ids=list(passages),
        evidence=[
            *(EvidenceItem(passage_id=p, source_id=f"s-{p}") for p in passages),
            *(EvidenceItem(passage_id=p, source_id=f"s-{p}", stance="contradicts") for p in contradicts),
        ],
        verdict="supported",
        confidence=0.5,
        iteration=iteration,
    )


SAME = [1.0, 0.0]
CLOSE = [0.9, 0.1]
FAR = [0.0, 1.0]


class TestFindCandidates:
    def test_similar_claims_are_paired_most_similar_first(self) -> None:
        claims = [claim("a", "p1"), claim("b", "p2"), claim("c", "p3"), claim("d", "p4")]
        vectors = {"a": SAME, "b": SAME, "c": CLOSE, "d": FAR}
        pairs = find_candidates(
            claims, vectors, new_ids={"a", "b", "c", "d"}, min_similarity=0.9, max_pairs=10
        )
        assert [(p.a, p.b) for p in pairs] == [("a", "b"), ("a", "c"), ("b", "c")]
        assert pairs[0].similarity > pairs[1].similarity
        assert not any(p.linked for p in pairs)

    def test_fact_checker_links_are_compared_whatever_their_similarity(self) -> None:
        # a's evidence includes p2, judged as contradicting a; b was stated by p2.
        claims = [claim("a", "p1", contradicts=("p2",)), claim("b", "p2")]
        pairs = find_candidates(
            claims, {"a": SAME, "b": FAR}, new_ids={"a", "b"}, min_similarity=0.9, max_pairs=10
        )
        assert pairs == [CandidatePair(a="a", b="b", similarity=0.0, linked=True)]

    def test_links_come_before_similar_pairs_under_the_cap(self) -> None:
        claims = [claim("a", "p1"), claim("b", "p2"), claim("c", "p3", contradicts=("p4",)), claim("d", "p4")]
        vectors = {"a": SAME, "b": SAME, "c": FAR, "d": SAME}
        pairs = find_candidates(claims, vectors, new_ids={"a", "b", "c", "d"}, min_similarity=0.9, max_pairs=1)
        assert [(p.a, p.b, p.linked) for p in pairs] == [("c", "d", True)]

    def test_claims_from_one_passage_are_not_paired(self) -> None:
        claims = [claim("a", "p1"), claim("b", "p1", "p2")]
        assert find_candidates(claims, {"a": SAME, "b": SAME}, new_ids={"a", "b"}, min_similarity=0.5, max_pairs=10) == []

    def test_a_pair_needs_a_claim_from_this_round(self) -> None:
        claims = [claim("old1", "p1"), claim("old2", "p2"), claim("new", "p3", iteration=1)]
        vectors = {"old1": SAME, "old2": SAME, "new": SAME}
        pairs = find_candidates(claims, vectors, new_ids={"new"}, min_similarity=0.5, max_pairs=10)
        assert {(p.a, p.b) for p in pairs} == {("old1", "new"), ("old2", "new")}

    def test_recorded_contradictions_are_not_compared_again(self) -> None:
        claims = [claim("a", "p1"), claim("b", "p2"), claim("c", "p3")]
        vectors = {"a": SAME, "b": SAME, "c": SAME}
        pairs = find_candidates(
            claims, vectors, new_ids={"a", "b", "c"}, min_similarity=0.5, max_pairs=10, known=[{"a", "b"}]
        )
        assert {(p.a, p.b) for p in pairs} == {("a", "c"), ("b", "c")}

    def test_claims_without_vectors_are_only_paired_by_links(self) -> None:
        claims = [claim("a", "p1"), claim("b", "p2")]
        assert find_candidates(claims, {}, new_ids={"a", "b"}, min_similarity=0.0, max_pairs=10) == [
            CandidatePair(a="a", b="b", similarity=0.0)
        ]


class TestGroupPairs:
    def test_connected_pairs_form_one_topic(self) -> None:
        pairs = [CandidatePair("a", "b", 1.0), CandidatePair("b", "c", 0.9), CandidatePair("x", "y", 0.8)]
        assert group_pairs(pairs, max_size=6) == [["a", "b", "c"], ["x", "y"]]

    def test_a_pair_that_would_overflow_a_topic_becomes_its_own(self) -> None:
        pairs = [CandidatePair("a", "b", 1.0), CandidatePair("b", "c", 0.9), CandidatePair("c", "d", 0.8)]
        groups = group_pairs(pairs, max_size=3)
        assert groups == [["a", "b", "c"], ["c", "d"]]
        assert all(len(g) <= 3 for g in groups)

    def test_no_pairs_no_topics(self) -> None:
        assert group_pairs([], max_size=4) == []


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


CONFLICT = [(LAB, LAB_PAGE), (FIELD, FIELD_PAGE), (OTHER, OTHER_PAGE)]


class TestDetection:
    async def test_conflicting_claims_become_a_stored_contradiction(self) -> None:
        runtime = build_fake_runtime(responses={"ContradictionOutput": responder(([LAB, FIELD], "date"))})
        state = await seed(runtime, CONFLICT)

        result = await make_contradiction_detector_node(runtime)(state)

        [item] = [ContradictionItem.model_validate(c) for c in result["contradictions"]]
        ids = {c["text"]: c["id"] for c in state["claims"]}
        assert item.claim_ids == [ids[LAB], ids[FIELD]]
        assert item.likely_cause == "date"
        assert item.resolved is True, "a date difference is a difference in scope"
        assert item.explanation.startswith("The datasheet figure")
        assert store_of(runtime).contradictions[item.id]["claim_ids"] == item.claim_ids
        [found] = events(result, "contradiction_found")
        assert found["payload"]["contradiction_id"] == item.id
        assert found["payload"]["resolved"] is True
        assert [c["prompt_version"] for c in result["llm_calls"]] == [PROMPT_VERSION]

    async def test_a_re_run_finds_the_same_contradiction_under_the_same_id(self) -> None:
        """M8: after a resume the detector may run again; it must replace, not duplicate."""
        runtime = build_fake_runtime(responses={"ContradictionOutput": responder(([LAB, FIELD], "date"))})
        state = await seed(runtime, CONFLICT)
        node = make_contradiction_detector_node(runtime)

        [first] = (await node(state))["contradictions"]
        [second] = (await node(state))["contradictions"]

        assert first["id"] == second["id"]
        assert first["id"] == stable_id(RUN, "contradiction", *sorted(first["claim_ids"]))
        assert len(store_of(runtime).contradictions) == 1

    async def test_the_prompt_shows_what_explains_the_difference(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, CONFLICT)
        await make_contradiction_detector_node(runtime)(state)

        [call] = llm_of(runtime).calls_for("ContradictionOutput")
        prompt = call["prompt"]
        # Both claims, each with its source's type and date: enough to name the cause.
        assert LAB in prompt and FIELD in prompt
        assert "official docs, 2021-02-01" in prompt
        assert "government, 2025-06-01" in prompt
        assert OTHER not in prompt, "an unrelated claim is not a candidate"

    async def test_unexplained_and_methodology_conflicts_are_unresolved(self) -> None:
        for cause in ("unexplained", "methodology"):
            runtime = build_fake_runtime(responses={"ContradictionOutput": responder(([LAB, FIELD], cause))})
            state = await seed(runtime, CONFLICT)
            result = await make_contradiction_detector_node(runtime)(state)
            [item] = result["contradictions"]
            assert item["resolved"] is False, cause
            assert events(result, "agent_finished")[0]["payload"]["unresolved"] == 1

    async def test_resolved_causes_are_a_setting(self) -> None:
        from researchmind_agents.config import ScoringConfig

        config = AgentConfig(scoring=ScoringConfig(resolved_causes=()))
        runtime = build_fake_runtime(config, responses={"ContradictionOutput": responder(([LAB, FIELD], "date"))})
        state = await seed(runtime, CONFLICT)
        [item] = (await make_contradiction_detector_node(runtime)(state))["contradictions"]
        assert item["resolved"] is False

    async def test_model_labels_must_name_two_claims_of_one_topic(self) -> None:
        bad = json.dumps(
            {
                "contradictions": [
                    {"claims": ["C1"], "likely_cause": "date", "explanation": "One claim only."},
                    {"claims": ["C1", "C99"], "likely_cause": "date", "explanation": "Unknown label."},
                    {"claims": ["C1", "C1"], "likely_cause": "date", "explanation": "Same claim twice."},
                ]
            }
        )
        runtime = build_fake_runtime(responses={"ContradictionOutput": bad})
        state = await seed(runtime, CONFLICT)
        result = await make_contradiction_detector_node(runtime)(state)
        assert result["contradictions"] == []
        assert store_of(runtime).contradictions == {}

    async def test_claims_from_different_topics_cannot_conflict(self) -> None:
        # Two topics in one call: X200 claims, and two near-identical furnace claims.
        furnace = "Gas furnaces need yearly servicing by a certified heating technician."
        pages = [*CONFLICT, (furnace, ("https://other-furnace.example/x", None, "unknown"))]
        cross = json.dumps(
            {"contradictions": [{"claims": ["C1", "C3"], "likely_cause": "date", "explanation": "Cross-topic."}]}
        )
        runtime = build_fake_runtime(responses={"ContradictionOutput": cross})
        state = await seed(runtime, pages)
        result = await make_contradiction_detector_node(runtime)(state)

        [call] = llm_of(runtime).calls_for("ContradictionOutput")
        assert "Topic T1:" in call["prompt"] and "Topic T2:" in call["prompt"]
        assert result["contradictions"] == []

    async def test_earlier_contradictions_are_kept_and_not_reported_twice(self) -> None:
        runtime = build_fake_runtime(responses={"ContradictionOutput": responder(([LAB, FIELD], "date"))})
        state = await seed(runtime, CONFLICT)
        first = await make_contradiction_detector_node(runtime)(state)
        # A later round with one new claim on the same topic.
        state["contradictions"] = first["contradictions"]
        newer = (await seed(runtime, [(LAB.replace("3.1", "2.9"), ("https://review.example/x200", None, "news"))], iteration=1))
        state["claims"] = [*state["claims"], *newer["claims"]]
        state["sources"] = [*state["sources"], *newer["sources"]]
        state["passages"] = [*state["passages"], *newer["passages"]]
        state["iteration"] = 1

        second = await make_contradiction_detector_node(runtime)(state)

        assert [c["id"] for c in second["contradictions"]] == [c["id"] for c in first["contradictions"]]
        assert events(second, "contradiction_found") == []

    async def test_only_usable_claims_are_compared(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, CONFLICT, verdict="unsupported")
        result = await make_contradiction_detector_node(runtime)(state)
        assert llm_of(runtime).calls_for("ContradictionOutput") == []
        assert "contradictions" not in result

    async def test_a_round_without_new_claims_compares_nothing(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, CONFLICT)
        state["iteration"] = 1  # every claim is from round 0
        result = await make_contradiction_detector_node(runtime)(state)
        assert llm_of(runtime).calls_for("ContradictionOutput") == []
        assert "contradictions" not in result

    async def test_a_failed_call_loses_only_its_topics(self) -> None:
        runtime = build_fake_runtime(responses={"ContradictionOutput": RuntimeError("model down")})
        state = await seed(runtime, CONFLICT)
        result = await make_contradiction_detector_node(runtime)(state)

        assert result["contradictions"] == []
        assert events(result, "error")
        assert events(result, "agent_finished")[0]["payload"]["failed_topics"] == 1

    async def test_spent_budget_skips_the_comparison(self) -> None:
        runtime = build_fake_runtime(AgentConfig(max_tokens_per_run=10, budget_synthesis_reserve=0.0))
        runtime.budget.tokens_used = 10
        state = await seed(runtime, CONFLICT)
        result = await make_contradiction_detector_node(runtime)(state)
        assert llm_of(runtime).calls_for("ContradictionOutput") == []
        assert events(result, "agent_finished")[0]["payload"]["skipped"]

    async def test_a_store_failure_still_returns_the_contradictions(self) -> None:
        runtime = build_fake_runtime(responses={"ContradictionOutput": responder(([LAB, FIELD], "date"))})
        state = await seed(runtime, CONFLICT)

        async def broken(**_: Any) -> None:
            raise RuntimeError("db down")

        runtime.store.save_contradictions = broken  # type: ignore[method-assign]
        result = await make_contradiction_detector_node(runtime)(state)
        assert len(result["contradictions"]) == 1
        assert any("Failed to store" in e["payload"]["message"] for e in events(result, "error"))

    async def test_topics_are_batched_per_call(self) -> None:
        config = AgentConfig(contradictions=ContradictionConfig(groups_per_call=1))
        furnace = "Gas furnaces need yearly servicing by a certified heating technician."
        pages = [*CONFLICT, (furnace, ("https://other-furnace.example/x", None, "unknown"))]
        runtime = build_fake_runtime(config)
        state = await seed(runtime, pages)
        await make_contradiction_detector_node(runtime)(state)
        assert len(llm_of(runtime).calls_for("ContradictionOutput")) == 2
