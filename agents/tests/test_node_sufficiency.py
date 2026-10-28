"""
M6 tests — sufficiency checker node.

Rules first (claims and independent sites per sub-question), the model only
when the rules pass and another round could follow. The node loops — and
raises ``iteration`` — only when the run needs more and a round is allowed by
max_iterations, the budget, and the mode's deadline.
"""

from __future__ import annotations

import json
from typing import Any

from researchmind_agents.config import AgentConfig, SufficiencyConfig
from researchmind_agents.nodes.sufficiency import LOOP, SYNTHESIZE, make_sufficiency_node
from researchmind_agents.prompts.sufficiency import PROMPT_VERSION
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    ClaimItem,
    ContradictionItem,
    EvidenceItem,
    ResearchState,
    SubQuestionItem,
    new_research_state,
)
from researchmind_agents.testing import ScriptedLLM, build_fake_runtime

SITES = ["https://a.example/1", "https://b.example/2", "https://c.example/3"]


def state_with(
    claims_per_sub_question: list[int],
    *,
    iteration: int = 0,
    max_iterations: int = 3,
    mode: str = "deep",
) -> ResearchState:
    """Sub-questions with the given number of supported claims, each from a different site."""
    state = new_research_state(
        run_id="run-suff", question="Is it good?", mode=mode, max_iterations=max_iterations
    )
    state["iteration"] = iteration
    sub_questions, claims = [], []
    sources = [{"id": f"s{i}", "url": url} for i, url in enumerate(SITES)]
    for index, count in enumerate(claims_per_sub_question):
        item = SubQuestionItem(id=f"sq{index}", text=f"Sub-question {index}?")
        sub_questions.append(item.model_dump())
        for n in range(count):
            source = sources[n % len(sources)]
            claims.append(
                ClaimItem(
                    id=f"c{index}-{n}",
                    text=f"Claim {n} about sub-question {index}.",
                    sub_question_id=item.id,
                    passage_ids=[f"p{index}-{n}"],
                    evidence=[EvidenceItem(passage_id=f"p{index}-{n}", source_id=source["id"])],
                    verdict="supported",
                    confidence=0.6,
                ).model_dump()
            )
    state["sub_questions"] = sub_questions
    state["claims"] = claims
    state["sources"] = sources
    return state


async def run(runtime: AgentRuntime, state: ResearchState) -> dict[str, Any]:
    return await make_sufficiency_node(runtime)(state)


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    assert isinstance(runtime.llm, ScriptedLLM)
    return runtime.llm


def insufficient(*gaps: str) -> str:
    return json.dumps({"sufficient": False, "gaps": list(gaps), "reason": "Pricing is missing."})


class TestRules:
    async def test_thin_sub_questions_are_gaps_and_the_run_loops(self) -> None:
        runtime = build_fake_runtime()
        result = await run(runtime, state_with([3, 1]))

        sufficiency = result["sufficiency"]
        assert sufficiency["status"] == "needs_more"
        assert sufficiency["decision"] == LOOP
        assert sufficiency["checked_by"] == "rules"
        [gap] = sufficiency["gaps"]
        assert gap["label"] == "G1"
        assert gap["sub_question_id"] == "sq1"
        assert "1 verified claim from 1 site" in gap["description"]
        assert result["iteration"] == 1
        assert llm_of(runtime).calls_for("SufficiencyOutput") == [], "rules first: no model call"
        assert [c["covered"] for c in sufficiency["coverage"]] == [True, False]

    async def test_claims_from_one_site_are_not_enough(self) -> None:
        runtime = build_fake_runtime(
            AgentConfig(sufficiency=SufficiencyConfig(min_sites_per_sub_question=4))
        )
        result = await run(runtime, state_with([3]))
        assert result["sufficiency"]["status"] == "needs_more"

    async def test_thresholds_are_settings(self) -> None:
        config = AgentConfig(
            sufficiency=SufficiencyConfig(
                min_claims_per_sub_question=1, min_sites_per_sub_question=1, llm_check=False
            )
        )
        result = await run(build_fake_runtime(config), state_with([1, 1]))
        assert result["sufficiency"]["status"] == "sufficient"
        assert result["sufficiency"]["decision"] == SYNTHESIZE
        assert "iteration" not in result

    async def test_events(self) -> None:
        result = await run(build_fake_runtime(), state_with([3, 0]))
        [checked] = [e for e in result["events"] if e["type"] == "sufficiency_checked"]
        assert checked["payload"]["decision"] == LOOP
        assert len(checked["payload"]["gaps"]) == 1
        assert result["events"][-1]["type"] == "agent_finished"


class TestModelJudgement:
    async def test_asked_only_when_the_rules_pass(self) -> None:
        runtime = build_fake_runtime()
        result = await run(runtime, state_with([3, 3]))

        [call] = llm_of(runtime).calls_for("SufficiencyOutput")
        assert "Sub-question 0?" in call["prompt"] and "Claim 0 about sub-question 1." in call["prompt"]
        assert result["sufficiency"]["status"] == "sufficient"
        assert result["sufficiency"]["checked_by"] == "rules+llm"
        assert [c["prompt_version"] for c in result["llm_calls"]] == [PROMPT_VERSION]

    async def test_the_model_can_name_gaps(self) -> None:
        runtime = build_fake_runtime(
            responses={"SufficiencyOutput": insufficient("2026 list prices", "2026 list prices", "Support terms")}
        )
        result = await run(runtime, state_with([3, 3]))

        sufficiency = result["sufficiency"]
        assert sufficiency["decision"] == LOOP
        assert [g["description"] for g in sufficiency["gaps"]] == ["2026 list prices", "Support terms"]
        assert all(g["sub_question_id"] is None for g in sufficiency["gaps"])
        assert sufficiency["reason"] == "Pricing is missing."

    async def test_model_gaps_are_capped(self) -> None:
        runtime = build_fake_runtime(
            AgentConfig(sufficiency=SufficiencyConfig(max_gaps=1)),
            responses={"SufficiencyOutput": insufficient("one", "two")},
        )
        result = await run(runtime, state_with([3, 3]))
        assert len(result["sufficiency"]["gaps"]) == 1

    async def test_insufficient_without_gaps_is_sufficient(self) -> None:
        runtime = build_fake_runtime(responses={"SufficiencyOutput": insufficient()})
        result = await run(runtime, state_with([3, 3]))
        assert result["sufficiency"]["status"] == "sufficient"

    async def test_a_failed_judgement_leaves_the_rules_verdict(self) -> None:
        runtime = build_fake_runtime(responses={"SufficiencyOutput": RuntimeError("down")})
        result = await run(runtime, state_with([3, 3]))
        assert result["sufficiency"]["status"] == "sufficient"
        assert result["sufficiency"]["checked_by"] == "rules"
        assert any(e["type"] == "error" for e in result["events"])

    async def test_can_be_turned_off(self) -> None:
        runtime = build_fake_runtime(AgentConfig(sufficiency=SufficiencyConfig(llm_check=False)))
        await run(runtime, state_with([3, 3]))
        assert llm_of(runtime).calls_for("SufficiencyOutput") == []

    async def test_unresolved_contradictions_are_shown(self) -> None:
        runtime = build_fake_runtime()
        state = state_with([3, 3])
        state["contradictions"] = [
            ContradictionItem(
                id="x", claim_ids=["c0-0", "c1-0"], explanation="They differ.", likely_cause="unexplained"
            ).model_dump()
        ]
        await run(runtime, state)
        [call] = llm_of(runtime).calls_for("SufficiencyOutput")
        assert "Unresolved contradictions" in call["prompt"] and "They differ." in call["prompt"]


class TestWhenTheLoopStops:
    async def test_at_max_iterations(self) -> None:
        runtime = build_fake_runtime()
        result = await run(runtime, state_with([1], iteration=2, max_iterations=3))

        sufficiency = result["sufficiency"]
        assert sufficiency["status"] == "needs_more"
        assert sufficiency["decision"] == SYNTHESIZE
        assert sufficiency["stopped_by"] == "max_iterations"
        assert "iteration" not in result

    async def test_no_model_call_when_no_round_could_follow(self) -> None:
        runtime = build_fake_runtime()
        result = await run(runtime, state_with([3, 3], iteration=0, max_iterations=1))
        assert llm_of(runtime).calls_for("SufficiencyOutput") == []
        assert result["sufficiency"]["decision"] == SYNTHESIZE

    async def test_when_the_research_budget_is_spent(self) -> None:
        runtime = build_fake_runtime(AgentConfig(max_tokens_per_run=10, budget_synthesis_reserve=0.0))
        runtime.budget.tokens_used = 10
        result = await run(runtime, state_with([1]))
        assert result["sufficiency"]["stopped_by"] == "budget"
        assert result["sufficiency"]["decision"] == SYNTHESIZE

    async def test_after_the_modes_deadline(self) -> None:
        clock = [0.0]
        config = AgentConfig(sufficiency=SufficiencyConfig(loop_deadlines=(("deep", 100.0),)))
        runtime = build_fake_runtime(config, clock=lambda: clock[0])

        clock[0] = 99.0
        assert (await run(runtime, state_with([1])))["sufficiency"]["decision"] == LOOP
        clock[0] = 100.0
        late = await run(runtime, state_with([1]))
        assert late["sufficiency"]["decision"] == SYNTHESIZE
        assert late["sufficiency"]["stopped_by"] == "deadline"

    async def test_a_mode_without_a_deadline_is_not_timed(self) -> None:
        clock = [10_000.0]
        runtime = build_fake_runtime(clock=lambda: clock[0])
        runtime.started = 0.0
        result = await run(runtime, state_with([1], mode="deep"))
        assert result["sufficiency"]["stopped_by"] == "deadline", "deep has a default deadline"
        runtime_no = build_fake_runtime(
            AgentConfig(sufficiency=SufficiencyConfig(loop_deadlines=())), clock=lambda: clock[0]
        )
        runtime_no.started = 0.0
        assert (await run(runtime_no, state_with([1])))["sufficiency"]["decision"] == LOOP
