"""
M5 tests — fact checker node.

The store is seeded with sources (each with a known reliability), embedded
passages, and claims as the extractor would leave them. The properties under
test: the model only gives labels, stances, and verdicts; verdicts are made to
agree with the stances; confidence comes from ``reliability.claim_confidence``;
and every claim is stored linked to passages, whatever happens to the model.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from researchmind_agents.config import AgentConfig, FactCheckConfig
from researchmind_agents.nodes.common import EventLog
from researchmind_agents.nodes.fact_checker import (
    CandidatePassage,
    SourceInfo,
    check_claims,
    make_fact_checker_node,
)
from researchmind_agents.prompts.fact_checker import PROMPT_VERSION
from researchmind_agents.reliability import EvidenceSignal, claim_confidence
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    ClaimItem,
    EvidenceItem,
    PassageItem,
    ResearchState,
    SourceItem,
    SubQuestionItem,
    new_research_state,
)
from researchmind_agents.storage import InMemoryResearchStore
from researchmind_agents.testing import ScriptedLLM, build_fake_runtime

RUN = "run-check"
CLAIM = "Heat pumps keep a coefficient of performance above two at minus fifteen."

# (url, text, reliability)
PAGE_A = (
    "https://a.example/cold",
    "Heat pumps keep a coefficient of performance above two at minus fifteen degrees.",
    0.5,
)
PAGE_B = (
    "https://b.example/field",
    "Field tests show heat pumps keep a coefficient of performance above two at minus fifteen.",
    0.7,
)
PAGE_A2 = (
    "https://a.example/other",
    "Another page on the same site says heat pumps keep a coefficient of performance above two.",
    0.5,
)


async def seed(
    runtime: AgentRuntime,
    pages: list[tuple[str, str, float]],
    claims: list[tuple[str, int]],
) -> ResearchState:
    """Store pages (one passage each) and claims as ``(text, page index)``."""
    sub_question = SubQuestionItem(id=str(uuid.uuid4()), text="Do heat pumps work in the cold?")
    await runtime.store.save_plan(run_id=RUN, plan={}, sub_questions=[sub_question])
    sources, passages = [], []
    for number, (url, text, reliability) in enumerate(pages, start=1):
        source = SourceItem(
            id=str(uuid.uuid4()),
            url=url,
            title=f"Page {number}",
            domain_type="news",
            reliability_score=reliability,
            sub_question_id=sub_question.id,
        )
        await runtime.store.save_source(run_id=RUN, source=source)
        passage = PassageItem(
            id=str(uuid.uuid4()), source_id=source.id, sub_question_id=sub_question.id, text=text
        )
        [vector] = await runtime.embeddings.embed_documents([text])
        await runtime.store.save_passages(
            run_id=RUN, passages=[passage], embeddings=[vector.vector], embedding_model="test"
        )
        sources.append(source.model_dump())
        passages.append(passage.model_dump())

    claim_items = [
        ClaimItem(
            id=str(uuid.uuid4()),
            text=text,
            sub_question_id=sub_question.id,
            passage_ids=[passages[index]["id"]],
            evidence=[
                EvidenceItem(
                    passage_id=passages[index]["id"],
                    source_id=passages[index]["source_id"],
                    stance="supports",
                )
            ],
        )
        for text, index in claims
    ]
    state = new_research_state(run_id=RUN, question="Do heat pumps work in the cold?", mode="deep")
    state["sub_questions"] = [sub_question.model_dump()]
    state["sources"] = sources
    state["passages"] = passages
    state["claims"] = [c.model_dump() for c in claim_items]
    return state


def blocks(prompt: str) -> list[tuple[str, str, list[str]]]:
    """``(claim label, claim text, evidence labels)`` per claim block in a fact-check prompt."""
    found: list[tuple[str, str, list[str]]] = []
    for line in prompt.splitlines():
        claim = re.match(r"^\[(C\d+)\] (.*)$", line)
        evidence = re.match(r"^\s+\[(E\d+)\] ", line)
        if claim:
            found.append((claim.group(1), claim.group(2), []))
        elif evidence and found:
            found[-1][2].append(evidence.group(1))
    return found


Decide = Callable[[str, str, list[str]], tuple[str, list[str]] | None]


def checks(decide: Decide, **extra: Any) -> Callable[[str], str]:
    """A responder: *decide(label, text, evidence labels)* -> (verdict, stances), or None to omit."""

    def respond(prompt: str) -> str:
        out = []
        for label, text, evidence in blocks(prompt):
            decision = decide(label, text, evidence)
            if decision is None:
                continue
            verdict, stances = decision
            out.append(
                {
                    "claim": label,
                    "verdict": verdict,
                    "evidence": [
                        {"evidence": e, "stance": s} for e, s in zip(evidence, stances, strict=False)
                    ],
                    "reason": "Checked.",
                    **extra,
                }
            )
        return json.dumps({"checks": out})

    return respond


def all_support(label: str, text: str, evidence: list[str]) -> tuple[str, list[str]]:
    return "supported", ["supports"] * len(evidence)


def config(**fact_check: Any) -> AgentConfig:
    return AgentConfig(fact_check=FactCheckConfig(**fact_check))


def store_of(runtime: AgentRuntime) -> InMemoryResearchStore:
    assert isinstance(runtime.store, InMemoryResearchStore)
    return runtime.store


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    assert isinstance(runtime.llm, ScriptedLLM)
    return runtime.llm


def claims_of(result: dict[str, Any]) -> list[ClaimItem]:
    return [ClaimItem.model_validate(c) for c in result["claims"]]


def events(result: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    return [e for e in result["events"] if e["type"] == event_type]


def expected(runtime: AgentRuntime, verdict: str, evidence: list[tuple[str, float, str]]) -> float:
    return claim_confidence(
        verdict,
        [EvidenceSignal(site=site, reliability=r, stance=stance) for site, r, stance in evidence],
        config=runtime.config.scoring,
    )


async def run(runtime: AgentRuntime, state: ResearchState) -> dict[str, Any]:
    return await make_fact_checker_node(runtime)(state)


# ===========================================================================
# Verdicts, stances, and confidence
# ===========================================================================


class TestVerdicts:
    async def test_supported_by_its_own_passage(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, [PAGE_A], [(CLAIM, 0)])

        [claim] = claims_of(await run(runtime, state))

        assert claim.verdict == "supported"
        assert claim.reason == "Scripted."
        [link] = claim.evidence
        assert link.passage_id == state["passages"][0]["id"]
        assert link.stance == "supports"
        assert claim.confidence == expected(runtime, "supported", [("a.example", 0.5, "supports")])
        assert claim.confidence == round(runtime.config.scoring.source_weight * 0.5, 4)

    async def test_confidence_is_computed_never_taken_from_the_model(self) -> None:
        runtime = build_fake_runtime(
            responses={"FactCheckOutput": checks(all_support, confidence=0.99)}
        )
        state = await seed(runtime, [PAGE_A], [(CLAIM, 0)])

        [claim] = claims_of(await run(runtime, state))

        assert claim.confidence != 0.99
        assert claim.confidence == expected(runtime, "supported", [("a.example", 0.5, "supports")])

    async def test_an_independent_source_raises_confidence(self) -> None:
        alone = build_fake_runtime(
            config(evidence_per_claim=1), responses={"FactCheckOutput": checks(all_support)}
        )
        [own_only] = claims_of(await run(alone, await seed(alone, [PAGE_A, PAGE_B], [(CLAIM, 0)])))

        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(all_support)})
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0)])
        [corroborated] = claims_of(await run(runtime, state))

        assert len(own_only.evidence) == 1
        assert {e.passage_id for e in corroborated.evidence} == {p["id"] for p in state["passages"]}
        assert corroborated.evidence[0].passage_id == state["passages"][0]["id"], "own passage first"
        assert corroborated.confidence > own_only.confidence
        assert corroborated.confidence == expected(
            runtime,
            "supported",
            [("a.example", 0.5, "supports"), ("b.example", 0.7, "supports")],
        )

    async def test_a_second_page_from_the_same_site_is_not_independent(self) -> None:
        alone = build_fake_runtime(
            config(evidence_per_claim=1), responses={"FactCheckOutput": checks(all_support)}
        )
        [own_only] = claims_of(await run(alone, await seed(alone, [PAGE_A, PAGE_A2], [(CLAIM, 0)])))

        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(all_support)})
        [same_site] = claims_of(
            await run(runtime, await seed(runtime, [PAGE_A, PAGE_A2], [(CLAIM, 0)]))
        )

        assert len(same_site.evidence) == 2
        assert same_site.confidence == own_only.confidence

    async def test_a_contradicting_excerpt_makes_the_claim_mixed(self) -> None:
        def decide(label: str, text: str, evidence: list[str]) -> tuple[str, list[str]]:
            return "supported", ["supports", "contradicts"]

        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(decide)})
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0)])

        [claim] = claims_of(await run(runtime, state))

        assert claim.verdict == "mixed"
        assert [e.stance for e in claim.evidence] == ["supports", "contradicts"]
        assert claim.confidence == expected(
            runtime, "mixed", [("a.example", 0.5, "supports"), ("b.example", 0.7, "contradicts")]
        )

    async def test_supported_with_nothing_supporting_is_unsupported(self) -> None:
        def decide(label: str, text: str, evidence: list[str]) -> tuple[str, list[str]]:
            return "supported", ["neutral"] * len(evidence)

        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(decide)})
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0)])

        [claim] = claims_of(await run(runtime, state))

        assert claim.verdict == "unsupported"
        assert claim.confidence == 0.0
        assert claim.evidence, "an unsupported claim still links to the passages it was checked against"

    async def test_unjudged_excerpts_count_as_neutral(self) -> None:
        def decide(label: str, text: str, evidence: list[str]) -> tuple[str, list[str]]:
            return "weak", ["supports"]  # judges the first excerpt only

        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(decide)})
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0)])

        [claim] = claims_of(await run(runtime, state))

        assert claim.verdict == "weak"
        assert [e.stance for e in claim.evidence] == ["supports", "neutral"]


# ===========================================================================
# Claims the model did not judge
# ===========================================================================


class TestUnverified:
    async def test_a_claim_missing_from_the_output_stays_unverified(self) -> None:
        def decide(label: str, text: str, evidence: list[str]) -> tuple[str, list[str]] | None:
            return all_support(label, text, evidence) if label == "C1" else None

        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(decide)})
        state = await seed(
            runtime, [PAGE_A, PAGE_B], [(CLAIM, 0), ("Gas furnaces burn natural gas for heat.", 1)]
        )

        first, second = claims_of(await run(runtime, state))

        assert first.verdict == "supported"
        assert second.verdict == "unverified"
        assert second.confidence == 0.0
        assert [e.passage_id for e in second.evidence] == [state["passages"][1]["id"]]
        assert second.evidence[0].stance == "supports"

    async def test_unknown_claim_and_evidence_labels_are_ignored(self) -> None:
        def respond(prompt: str) -> str:
            [(_, _, first_evidence), (_, _, second_evidence)] = blocks(prompt)
            return json.dumps(
                {
                    "checks": [
                        {"claim": "C9", "verdict": "supported", "evidence": [], "reason": "?"},
                        # C1 names only evidence that belongs to C2, or to nobody.
                        {
                            "claim": "C1",
                            "verdict": "supported",
                            "evidence": [
                                {"evidence": second_evidence[0], "stance": "supports"},
                                {"evidence": "E99", "stance": "supports"},
                            ],
                            "reason": "?",
                        },
                        {
                            "claim": "[c2]",
                            "verdict": "supported",
                            "evidence": [{"evidence": second_evidence[0], "stance": "supports"}],
                            "reason": "ok",
                        },
                    ]
                }
            )

        runtime = build_fake_runtime(
            config(evidence_per_claim=1), responses={"FactCheckOutput": respond}
        )
        state = await seed(
            runtime, [PAGE_A, PAGE_B], [(CLAIM, 0), ("Gas furnaces burn natural gas for heat.", 1)]
        )

        first, second = claims_of(await run(runtime, state))

        assert first.verdict == "unverified"
        assert second.verdict == "supported"

    async def test_a_failed_batch_leaves_only_its_claims_unverified(self) -> None:
        def respond(prompt: str) -> str:
            if "Gas furnaces" in prompt:
                raise RuntimeError("model broke")
            return checks(all_support)(prompt)

        runtime = build_fake_runtime(config(batch_size=1), responses={"FactCheckOutput": respond})
        state = await seed(
            runtime, [PAGE_A, PAGE_B], [(CLAIM, 0), ("Gas furnaces burn natural gas for heat.", 1)]
        )

        result = await run(runtime, state)

        first, second = claims_of(result)
        assert first.verdict == "supported"
        assert second.verdict == "unverified"
        assert len(llm_of(runtime).calls_for("FactCheckOutput")) == 2
        [error] = events(result, "error")
        assert "model broke" in error["payload"]["reason"]
        assert error["payload"]["claims"] == 1
        assert len(result["llm_calls"]) == 1

    async def test_budget_exhausted_leaves_every_claim_unverified(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0), ("Gas furnaces burn gas.", 1)])
        runtime.budget.tokens_used = runtime.budget.max_tokens

        result = await run(runtime, state)

        assert llm_of(runtime).calls_for("FactCheckOutput") == []
        assert {c.verdict for c in claims_of(result)} == {"unverified"}
        assert all(c.confidence == 0.0 and c.evidence for c in claims_of(result))
        assert events(result, "fact_check_skipped")
        assert len(store_of(runtime).claims) == 2, "unverified claims are still stored"

    async def test_retrieval_failure_checks_against_own_passages(self) -> None:
        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(all_support)})
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0)])

        async def broken(texts: list[str]) -> Any:
            raise RuntimeError("embeddings down")

        runtime.embeddings.embed_queries = broken  # type: ignore[method-assign]
        result = await run(runtime, state)

        [claim] = claims_of(result)
        assert claim.verdict == "supported"
        assert [e.passage_id for e in claim.evidence] == [state["passages"][0]["id"]]
        assert any("Evidence retrieval failed" in e["payload"]["message"] for e in events(result, "error"))


# ===========================================================================
# Storage, events, and the prompt
# ===========================================================================


class _FailingClaimsStore(InMemoryResearchStore):
    async def save_claims(self, *, run_id: str, claims: list[ClaimItem]) -> None:
        raise RuntimeError("db down")


class TestStorageAndEvents:
    async def test_claims_are_stored_with_their_evidence_links(self) -> None:
        def decide(label: str, text: str, evidence: list[str]) -> tuple[str, list[str]]:
            return "supported", ["supports", "contradicts"]

        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(decide)})
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0)])

        result = await run(runtime, state)

        [claim] = claims_of(result)
        stored = store_of(runtime).claims[claim.id]
        assert stored["run_id"] == RUN
        assert stored["verdict"] == "mixed"
        assert stored["confidence"] == claim.confidence
        assert [(e["passage_id"], e["stance"]) for e in stored["evidence"]] == [
            (p["id"], stance) for p, stance in zip(state["passages"], ["supports", "contradicts"], strict=True)
        ]
        assert all(e["excerpt"] for e in stored["evidence"])

    async def test_one_claim_verified_event_per_claim(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0), ("Gas furnaces burn gas.", 1)])

        result = await run(runtime, state)

        verified = events(result, "claim_verified")
        assert [e["payload"]["claim_id"] for e in verified] == [c["id"] for c in state["claims"]]
        for event, claim in zip(verified, claims_of(result), strict=True):
            assert event["payload"]["verdict"] == claim.verdict
            assert event["payload"]["confidence"] == claim.confidence
            assert event["payload"]["text"] == claim.text
        finished = result["events"][-1]
        assert finished["type"] == "agent_finished"
        assert finished["payload"] == {"iteration": 0, "claims": 2, "by_verdict": {"supported": 2}}
        assert {e["agent"] for e in store_of(runtime).events} == {"fact_checker"}
        assert [c["prompt_version"] for c in result["llm_calls"]] == [PROMPT_VERSION]

    async def test_a_store_failure_is_reported_and_claims_still_returned(self) -> None:
        runtime = build_fake_runtime(store=_FailingClaimsStore())
        state = await seed(runtime, [PAGE_A], [(CLAIM, 0)])

        result = await run(runtime, state)

        assert claims_of(result)[0].verdict == "supported"
        assert any("Failed to store claims" in e["payload"]["message"] for e in events(result, "error"))

    async def test_no_claims_no_model_call(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, [PAGE_A], [])

        result = await run(runtime, state)

        assert "claims" not in result, "the state's claims are left as they are"
        assert llm_of(runtime).calls_for("FactCheckOutput") == []
        assert result["events"][-1]["payload"] == {"iteration": 0, "claims": 0}

    async def test_excerpts_are_bounded(self) -> None:
        long_text = " ".join(
            f"Sentence {i} talks about something unrelated to the topic at hand." for i in range(40)
        ) + " Heat pumps keep a coefficient of performance above two at minus fifteen."
        runtime = build_fake_runtime(config(excerpt_chars=120))
        state = await seed(runtime, [("https://a.example/long", long_text, 0.5)], [(CLAIM, 0)])

        [claim] = claims_of(await run(runtime, state))

        excerpt = claim.evidence[0].excerpt
        assert len(excerpt.strip("…")) <= 120
        assert "minus fifteen" in excerpt, "the excerpt is the part nearest the claim"
        prompt = llm_of(runtime).calls_for("FactCheckOutput")[0]["prompt"]
        assert long_text not in prompt

    async def test_model_sees_labels_not_ids_or_urls(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0)])

        await run(runtime, state)

        prompt = llm_of(runtime).calls_for("FactCheckOutput")[0]["prompt"]
        assert "[C1]" in prompt and "[E1]" in prompt and "[E2]" in prompt
        for item in [*state["sources"], *state["passages"], *state["claims"]]:
            assert item["id"] not in prompt
        for source in state["sources"]:
            assert source["url"] not in prompt
        assert "a.example (news)" in prompt, "each excerpt names its site and source type"


class TestCheckClaims:
    async def test_runs_without_graph_state(self) -> None:
        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(all_support)})
        claim = ClaimItem(
            id="c-1",
            text=CLAIM,
            passage_ids=["p-1"],
            evidence=[EvidenceItem(passage_id="p-1", source_id="s-1")],
        )
        candidates = {
            "c-1": [
                CandidatePassage(passage_id="p-1", source_id="s-1", text=PAGE_A[1]),
                CandidatePassage(passage_id="p-2", source_id="s-2", text=PAGE_B[1]),
            ]
        }
        sources = {
            "s-1": SourceInfo("s-1", PAGE_A[0], "A", "news", 0.5),
            "s-2": SourceInfo("s-2", PAGE_B[0], "B", "official_docs", 0.8),
        }
        calls: list[dict[str, Any]] = []

        [checked] = await check_claims(
            runtime,
            run_id="eval",
            question="Do heat pumps work in the cold?",
            claims=[claim],
            candidates=candidates,
            sources=sources,
            log=EventLog(runtime, run_id="eval", agent="fact_checker"),
            llm_calls=calls,
        )

        assert checked.verdict == "supported"
        assert checked.confidence == expected(
            runtime, "supported", [("a.example", 0.5, "supports"), ("b.example", 0.8, "supports")]
        )
        assert len(calls) == 1
        assert claim.verdict == "unverified", "the input claim is not mutated"

    async def test_unknown_sources_are_typed_from_their_url(self) -> None:
        runtime = build_fake_runtime(responses={"FactCheckOutput": checks(all_support)})
        claim = ClaimItem(
            id="c-1",
            text=CLAIM,
            passage_ids=["p-1"],
            evidence=[EvidenceItem(passage_id="p-1", source_id="s-9")],
        )
        candidates = {
            "c-1": [
                CandidatePassage(
                    passage_id="p-1",
                    source_id="s-9",
                    text=PAGE_A[1],
                    url="https://docs.python.org/3/library/x.html",
                )
            ]
        }

        [checked] = await check_claims(
            runtime,
            run_id="eval",
            question="Q",
            claims=[claim],
            candidates=candidates,
            sources={},
            log=EventLog(runtime, run_id="eval", agent="fact_checker"),
            llm_calls=[],
        )

        prompt = llm_of(runtime).calls_for("FactCheckOutput")[0]["prompt"]
        assert "official docs" in prompt
        assert checked.confidence > 0


# ===========================================================================
# M6 — gap-filling rounds
# ===========================================================================


class TestRounds:
    async def test_only_this_rounds_claims_are_checked_and_stored(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, [PAGE_A, PAGE_B], [(CLAIM, 0), ("Gas furnaces burn gas.", 1)])
        old, new = (ClaimItem.model_validate(c) for c in state["claims"])
        old = old.model_copy(update={"verdict": "weak", "confidence": 0.3})
        new = new.model_copy(update={"iteration": 1})
        state["claims"] = [old.model_dump(), new.model_dump()]
        state["iteration"] = 1

        result = await run(runtime, state)

        [call] = llm_of(runtime).calls_for("FactCheckOutput")
        assert "Gas furnaces burn gas." in call["prompt"] and CLAIM not in call["prompt"]
        out = {c["id"]: c for c in result["claims"]}
        assert list(out) == [old.id, new.id], "every round's claims are returned, in order"
        assert (out[old.id]["verdict"], out[old.id]["confidence"]) == ("weak", 0.3)
        assert out[new.id]["verdict"] == "supported"
        assert set(store_of(runtime).claims) == {new.id}
        assert [e["payload"]["claim_id"] for e in events(result, "claim_verified")] == [new.id]

    async def test_a_round_without_new_claims_changes_nothing(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime, [PAGE_A], [(CLAIM, 0)])
        state["iteration"] = 1
        result = await run(runtime, state)
        assert "claims" not in result
        assert llm_of(runtime).calls_for("FactCheckOutput") == []
