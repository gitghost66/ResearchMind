"""
M5 tests — claim extractor node.

The store is seeded directly (plan, sources, embedded passages) and the node
is called on its own. The property under test is the M5 acceptance item:
every claim links to at least one stored passage, so a claim the model cannot
tie to a passage label never exists.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from researchmind_agents.config import AgentConfig, ClaimsConfig
from researchmind_agents.nodes.claim_extractor import (
    claim_key,
    clean_claim_text,
    make_claim_extractor_node,
)
from researchmind_agents.prompts.claim_extractor import PROMPT_VERSION
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    ClaimItem,
    PassageItem,
    ResearchState,
    SourceItem,
    SubQuestionItem,
    new_research_state,
)
from researchmind_agents.storage import InMemoryResearchStore
from researchmind_agents.testing import ScriptedLLM, build_fake_runtime

RUN = "run-claims"

SUB_QUESTIONS = [
    "How efficient are heat pumps in cold weather",
    "What do gas furnaces cost to run each winter",
]

# (url, text, sub-question index). Vocabulary differs per sub-question, so
# retrieval assigns each passage to the sub-question it was written for.
PAGES = [
    (
        "https://a.example/1",
        "Heat pumps stay efficient in cold weather down to minus fifteen. "
        "Efficient heat pumps keep a high coefficient in cold weather.",
        0,
    ),
    (
        "https://b.example/2",
        "Gas furnaces cost more to run each winter than expected. "
        "Running gas furnaces each winter costs households money.",
        1,
    ),
]


async def seed(
    runtime: AgentRuntime,
    pages: list[tuple[str, str, int]] = PAGES,
    *,
    sub_questions: list[str] | None = None,
) -> ResearchState:
    texts = sub_questions or SUB_QUESTIONS
    items = [SubQuestionItem(id=str(uuid.uuid4()), text=t, queries=[t]) for t in texts]
    await runtime.store.save_plan(run_id=RUN, plan={"objective": "obj"}, sub_questions=items)

    sources, passages = [], []
    for url, text, sq_index in pages:
        source = SourceItem(
            id=str(uuid.uuid4()), url=url, title=f"Title {url[-1]}", sub_question_id=items[sq_index].id
        )
        await runtime.store.save_source(run_id=RUN, source=source)
        passage = PassageItem(
            id=str(uuid.uuid4()), source_id=source.id, sub_question_id=items[sq_index].id, text=text
        )
        [vector] = await runtime.embeddings.embed_documents([text])
        await runtime.store.save_passages(
            run_id=RUN, passages=[passage], embeddings=[vector.vector], embedding_model="test-model"
        )
        sources.append(source.model_dump())
        passages.append(passage.model_dump())

    state = new_research_state(run_id=RUN, question="Heat pumps or gas furnaces?", mode="deep")
    state["sub_questions"] = [item.model_dump() for item in items]
    state["sources"] = sources
    state["passages"] = passages
    return state


def claims_json(*claims: tuple[str, list[str]]) -> str:
    return json.dumps({"claims": [{"text": text, "passages": labels} for text, labels in claims]})


def store_of(runtime: AgentRuntime) -> InMemoryResearchStore:
    assert isinstance(runtime.store, InMemoryResearchStore)
    return runtime.store


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    assert isinstance(runtime.llm, ScriptedLLM)
    return runtime.llm


def events(result: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    return [e for e in result["events"] if e["type"] == event_type]


def claims_of(result: dict[str, Any]) -> list[ClaimItem]:
    return [ClaimItem.model_validate(c) for c in result["claims"]]


def by_sub_question(state: ResearchState, index: int) -> Callable[[str], bool]:
    text = state["sub_questions"][index]["text"]
    return lambda prompt: f"Sub-question:\n{text}\n" in prompt


async def run(runtime: AgentRuntime, state: ResearchState) -> dict[str, Any]:
    return await make_claim_extractor_node(runtime)(state)


# ===========================================================================
# Linking claims to passages
# ===========================================================================


class TestClaimsLinkToPassages:
    async def test_every_claim_links_to_a_stored_passage(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)

        result = await run(runtime, state)

        claims = claims_of(result)
        assert claims, "the default responder states one claim per passage"
        stored = store_of(runtime).passages
        for claim in claims:
            assert claim.passage_ids, claim
            assert set(claim.passage_ids) <= set(stored)
            assert [e.passage_id for e in claim.evidence] == claim.passage_ids
            assert all(e.stance == "supports" for e in claim.evidence)
            assert all(e.source_id == stored[e.passage_id]["source_id"] for e in claim.evidence)
            assert claim.verdict == "unverified"
            assert claim.confidence == 0.0

    async def test_claims_are_tied_to_the_sub_question_of_their_passages(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)

        claims = claims_of(await run(runtime, state))

        stored = store_of(runtime).passages
        for claim in claims:
            assert claim.sub_question_id == stored[claim.passage_ids[0]]["sub_question_id"]
        assert {c.sub_question_id for c in claims} == {sq["id"] for sq in state["sub_questions"]}

    async def test_claims_citing_only_unknown_labels_are_dropped(self) -> None:
        runtime = build_fake_runtime(
            responses={
                "ClaimExtractionOutput": claims_json(
                    ("Heat pumps work well at minus fifteen.", ["P1"]),
                    ("A claim with no real passage behind it.", ["P9", "X1"]),
                    ("Another claim citing nothing at all.", []),
                )
            }
        )
        state = await seed(runtime)

        result = await run(runtime, state)

        texts = [c.text for c in claims_of(result)]
        assert "Heat pumps work well at minus fifteen." in texts
        assert not any("no real passage" in t or "citing nothing" in t for t in texts)
        dropped = [e["payload"]["dropped_without_passage"] for e in events(result, "claims_extracted")]
        assert sum(dropped) == 2 * len(dropped)

    async def test_labels_are_normalised(self) -> None:
        runtime = build_fake_runtime(
            responses={"ClaimExtractionOutput": claims_json(("Heat pumps work in the cold.", ["[p1]"]))}
        )
        state = await seed(runtime, PAGES[:1], sub_questions=SUB_QUESTIONS[:1])

        [claim] = claims_of(await run(runtime, state))

        assert claim.passage_ids == [state["passages"][0]["id"]]

    async def test_inline_passage_markers_are_stripped(self) -> None:
        runtime = build_fake_runtime(
            responses={
                "ClaimExtractionOutput": claims_json(
                    ("Heat pumps reach a COP of 2 at minus fifteen [P1].", ["P1"])
                )
            }
        )
        state = await seed(runtime, PAGES[:1], sub_questions=SUB_QUESTIONS[:1])

        [claim] = claims_of(await run(runtime, state))

        assert claim.text == "Heat pumps reach a COP of 2 at minus fifteen."

    async def test_too_short_claims_are_dropped(self) -> None:
        runtime = build_fake_runtime(
            responses={"ClaimExtractionOutput": claims_json(("Yes.", ["P1"]))}
        )
        state = await seed(runtime, PAGES[:1], sub_questions=SUB_QUESTIONS[:1])

        assert claims_of(await run(runtime, state)) == []

    async def test_same_fact_twice_is_one_claim_with_both_passages(self) -> None:
        pages = [PAGES[0], ("https://c.example/3", "Heat pumps stay efficient in cold weather too.", 0)]
        runtime = build_fake_runtime(
            responses={
                "ClaimExtractionOutput": claims_json(
                    ("Heat pumps stay efficient in cold weather.", ["P1"]),
                    ("heat pumps stay efficient in cold weather", ["P2", "P1"]),
                )
            }
        )
        state = await seed(runtime, pages, sub_questions=SUB_QUESTIONS[:1])

        [claim] = claims_of(await run(runtime, state))

        assert claim.text == "Heat pumps stay efficient in cold weather."
        assert sorted(claim.passage_ids) == sorted(p["id"] for p in state["passages"])
        assert len(claim.evidence) == 2

    async def test_claims_are_capped_per_sub_question(self) -> None:
        config = AgentConfig(claims=ClaimsConfig(max_claims_per_sub_question=2))
        runtime = build_fake_runtime(
            config,
            responses={
                "ClaimExtractionOutput": claims_json(
                    *((f"Heat pump finding number {i} holds.", ["P1"]) for i in range(5))
                )
            },
        )
        state = await seed(runtime, PAGES[:1], sub_questions=SUB_QUESTIONS[:1])

        result = await run(runtime, state)

        assert len(result["claims"]) == 2
        assert "List up to 2 claims" in llm_of(runtime).calls_for("ClaimExtractionOutput")[0]["prompt"]

    async def test_model_sees_labels_not_ids_or_urls(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)

        await run(runtime, state)

        for call in llm_of(runtime).calls_for("ClaimExtractionOutput"):
            assert "[P1]" in call["prompt"]
            for passage in state["passages"]:
                assert passage["id"] not in call["prompt"]
            for source in state["sources"]:
                assert source["url"] not in call["prompt"]
                assert source["id"] not in call["prompt"]


# ===========================================================================
# Degradation
# ===========================================================================


class TestDegradation:
    async def test_one_sub_question_failing_does_not_cost_the_others(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)
        first = by_sub_question(state, 0)
        default = llm_of(runtime)._responses["ClaimExtractionOutput"][0]
        assert callable(default)

        def responder(prompt: str) -> str:
            if first(prompt):
                raise RuntimeError("model broke")
            return default(prompt)

        llm_of(runtime)._responses["ClaimExtractionOutput"] = [responder]

        result = await run(runtime, state)

        claims = claims_of(result)
        assert claims
        assert {c.sub_question_id for c in claims} == {state["sub_questions"][1]["id"]}
        [error] = events(result, "error")
        assert error["payload"]["sub_question_id"] == state["sub_questions"][0]["id"]
        assert "model broke" in error["payload"]["reason"]
        assert events(result, "agent_finished")[0]["payload"]["claims"] == len(claims)

    async def test_budget_exhausted_skips_extraction(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)
        runtime.budget.tokens_used = runtime.budget.max_tokens

        result = await run(runtime, state)

        assert result["claims"] == []
        assert result["llm_calls"] == []
        assert llm_of(runtime).calls_for("ClaimExtractionOutput") == []
        assert [e["type"] for e in result["events"]] == ["agent_started", "agent_finished"]
        assert result["events"][-1]["payload"]["skipped"] == "research budget exhausted"

    async def test_no_passages_means_no_model_call(self) -> None:
        runtime = build_fake_runtime()
        items = [SubQuestionItem(id="sq-1", text="Anything", queries=["anything"])]
        state = new_research_state(run_id=RUN, question="Q", mode="deep")
        state["sub_questions"] = [i.model_dump() for i in items]

        result = await run(runtime, state)

        assert result["claims"] == []
        assert llm_of(runtime).calls_for("ClaimExtractionOutput") == []

    async def test_empty_model_output(self) -> None:
        runtime = build_fake_runtime(responses={"ClaimExtractionOutput": claims_json()})
        state = await seed(runtime)

        result = await run(runtime, state)

        assert result["claims"] == []
        assert all(e["payload"]["claims"] == 0 for e in events(result, "claims_extracted"))


# ===========================================================================
# Events and output shape
# ===========================================================================


class TestEventsAndOutput:
    async def test_events_and_llm_calls(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)

        result = await run(runtime, state)

        types = [e["type"] for e in result["events"]]
        assert types[0] == "agent_started"
        assert types[-1] == "agent_finished"
        extracted = events(result, "claims_extracted")
        assert {e["payload"]["sub_question_id"] for e in extracted} == {
            sq["id"] for sq in state["sub_questions"]
        }
        assert sum(e["payload"]["claims"] for e in extracted) == len(result["claims"])
        finished = result["events"][-1]["payload"]
        assert finished["claims"] == len(result["claims"])
        assert finished["sub_questions"] == 2
        assert finished["passages"] == 2
        assert {e["agent"] for e in store_of(runtime).events} == {"claim_extractor"}
        assert len(result["llm_calls"]) == 2
        assert {c["agent"] for c in result["llm_calls"]} == {"claim_extractor"}
        assert {c["prompt_version"] for c in result["llm_calls"]} == {PROMPT_VERSION}

    async def test_output_is_plain_json(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)

        result = await run(runtime, state)

        json.dumps(result)

    async def test_claims_are_not_stored_yet(self) -> None:
        """The fact checker stores claims, once they carry verdicts."""
        runtime = build_fake_runtime()
        state = await seed(runtime)

        await run(runtime, state)

        assert store_of(runtime).claims == {}


class TestHelpers:
    def test_clean_claim_text(self) -> None:
        assert clean_claim_text("  - Heat pumps   work [P1, P2] in cold (P3). ") == (
            "Heat pumps work in cold."
        )
        assert clean_claim_text("* Bullet claim [p4]") == "Bullet claim"
        assert clean_claim_text("No markers here.") == "No markers here."

    def test_claim_key_ignores_case_and_punctuation(self) -> None:
        assert claim_key("Heat pumps, at -15 °C, work!") == claim_key("heat pumps at 15 c work")
        assert claim_key("A") != claim_key("B")


# ===========================================================================
# M6 — gap-filling rounds
# ===========================================================================


class TestRounds:
    async def test_a_later_round_reads_only_its_sub_questions_and_new_passages(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)
        first = await make_claim_extractor_node(runtime)(state)
        read = set(first["extracted_passage_ids"])
        assert read == {p["id"] for p in state["passages"]}

        # Round 1: one new sub-question, one new page for it.
        follow_up = SubQuestionItem(
            id=str(uuid.uuid4()), text="How noisy are heat pumps at night", queries=["noise"], iteration=1,
            parent_id=state["sub_questions"][0]["id"],
        )
        await runtime.store.save_plan(run_id=RUN, plan={}, sub_questions=[follow_up], iteration=1)
        source = SourceItem(id=str(uuid.uuid4()), url="https://c.example/3", title="Noise")
        await runtime.store.save_source(run_id=RUN, source=source)
        text = "Heat pumps are noisy at night when defrosting. Night noise from heat pumps is low."
        passage = PassageItem(id=str(uuid.uuid4()), source_id=source.id, sub_question_id=follow_up.id, text=text)
        [vector] = await runtime.embeddings.embed_documents([text])
        await runtime.store.save_passages(
            run_id=RUN, passages=[passage], embeddings=[vector.vector], embedding_model="test-model"
        )
        state["sub_questions"] = [*state["sub_questions"], follow_up.model_dump()]
        state["sources"] = [*state["sources"], source.model_dump()]
        state["passages"] = [*state["passages"], passage.model_dump()]
        state["claims"] = first["claims"]
        state["extracted_passage_ids"] = first["extracted_passage_ids"]
        state["iteration"] = 1

        second = await make_claim_extractor_node(runtime)(state)

        calls = llm_of(runtime).calls_for("ClaimExtractionOutput")
        assert len(calls) == 3, "two in round 0, one for the follow-up"
        assert "How noisy are heat pumps at night" in calls[-1]["prompt"]
        assert "minus fifteen" not in calls[-1]["prompt"], "round 0's passages are not read again"
        old = [c for c in second["claims"] if c["iteration"] == 0]
        new = [c for c in second["claims"] if c["iteration"] == 1]
        assert [c["id"] for c in old] == [c["id"] for c in first["claims"]]
        assert new and all(c["passage_ids"] == [passage.id] for c in new)
        assert all(c["sub_question_id"] == follow_up.id for c in new)
        assert set(second["extracted_passage_ids"]) == read | {passage.id}

    async def test_a_claim_worded_like_an_earlier_one_is_dropped(self) -> None:
        runtime = build_fake_runtime()
        state = await seed(runtime)
        earlier = ClaimItem(
            id="old", text="Heat pumps stay efficient in cold weather down to minus fifteen.",
            passage_ids=[state["passages"][0]["id"]], verdict="supported", confidence=0.6,
        )
        state["claims"] = [earlier.model_dump()]
        result = await make_claim_extractor_node(runtime)(state)
        texts = [c["text"] for c in result["claims"]]
        assert texts.count(earlier.text) == 1
        assert result["claims"][0]["id"] == "old"
        finished = result["events"][-1]["payload"]
        assert finished["duplicates_of_earlier"] == 1

    async def test_spent_budget_keeps_earlier_claims(self) -> None:
        runtime = build_fake_runtime(AgentConfig(max_tokens_per_run=10, budget_synthesis_reserve=0.0))
        runtime.budget.tokens_used = 10
        state = await seed(runtime)
        state["claims"] = [ClaimItem(id="old", text="An earlier claim.", passage_ids=["p"]).model_dump()]
        result = await make_claim_extractor_node(runtime)(state)
        assert [c["id"] for c in result["claims"]] == ["old"]
