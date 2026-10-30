"""M3/M6 tests — planner node (mocked LLM, in-memory store)."""

from __future__ import annotations

import json
from typing import Any

from researchmind_agents.config import AgentConfig, PlannerConfig
from researchmind_agents.modes import MODES
from researchmind_agents.nodes.planner import fallback_plan, make_planner_node, normalize_plan
from researchmind_agents.prompts.planner import GAP_PROMPT_VERSION, PROMPT_VERSION
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    PlannedSubQuestion,
    PlanOutput,
    SubQuestionItem,
    new_research_state,
    stable_id,
)
from researchmind_agents.storage import InMemoryResearchStore
from researchmind_agents.testing import ScriptedLLM, build_fake_runtime

QUESTION = "Is RAG better than fine-tuning?"


def plan_json(sub_questions: list[dict[str, Any]], objective: str = "Compare them.") -> str:
    return json.dumps({"objective": objective, "sub_questions": sub_questions})


async def run_planner(runtime: AgentRuntime, **state_kwargs: Any) -> dict[str, Any]:
    node = make_planner_node(runtime)
    state = new_research_state(run_id="run-1", question=QUESTION, **state_kwargs)
    return await node(state)


def store_of(runtime: AgentRuntime) -> InMemoryResearchStore:
    assert isinstance(runtime.store, InMemoryResearchStore)
    return runtime.store


def llm_of(runtime: AgentRuntime) -> ScriptedLLM:
    assert isinstance(runtime.llm, ScriptedLLM)
    return runtime.llm


class TestHappyPath:
    async def test_plan_is_stored_with_state_ids(self) -> None:
        runtime = build_fake_runtime(
            responses={
                "PlanOutput": plan_json(
                    [
                        {"text": "What does RAG cost?", "queries": ["rag cost", "rag pricing"]},
                        {"text": "What does fine-tuning cost?", "queries": ["fine-tuning cost"]},
                    ]
                )
            }
        )

        update = await run_planner(runtime)

        store = store_of(runtime)
        ids = [sq["id"] for sq in update["sub_questions"]]
        assert len(ids) == 2
        assert set(store.sub_questions) == set(ids)
        assert store.plans["run-1"]["objective"] == "Compare them."
        assert [sq["id"] for sq in store.plans["run-1"]["sub_questions"]] == ids
        assert update["plan"]["prompt_version"] == PROMPT_VERSION
        assert update["plan"]["fallback_reason"] is None
        assert update["sub_questions"][0]["queries"] == ["rag cost", "rag pricing"]

    async def test_records_llm_call_and_events(self) -> None:
        runtime = build_fake_runtime(tokens=(120, 30))

        update = await run_planner(runtime)

        assert len(update["llm_calls"]) == 1
        call = update["llm_calls"][0]
        assert call["agent"] == "planner"
        assert call["prompt_version"] == PROMPT_VERSION
        assert call["tokens_in"] == 120
        assert call["tokens_out"] == 30

        types = [e["type"] for e in update["events"]]
        assert types == ["agent_started", "agent_finished"]
        persisted = [(e["agent"], e["type"]) for e in store_of(runtime).events]
        assert persisted == [("planner", "agent_started"), ("planner", "agent_finished")]
        assert update["events"][-1]["payload"]["fallback"] is False

    async def test_prompt_uses_config_and_context(self) -> None:
        config = AgentConfig(planner=PlannerConfig(min_sub_questions=3, max_sub_questions=6))
        runtime = build_fake_runtime(config)

        await run_planner(runtime, user_context={"region": "EU"})

        prompt = llm_of(runtime).calls[0]["prompt"]
        assert "between 3 and 6" in prompt
        assert "EU" in prompt

    async def test_budget_is_charged(self) -> None:
        runtime = build_fake_runtime(tokens=(100, 50))
        await run_planner(runtime)
        assert runtime.budget.tokens_used == 150

    async def test_below_minimum_is_flagged(self) -> None:
        runtime = build_fake_runtime(
            responses={"PlanOutput": plan_json([{"text": "Only one?", "queries": ["one"]}])}
        )
        update = await run_planner(runtime)
        assert update["events"][-1]["payload"]["below_minimum"] is True
        assert len(update["sub_questions"]) == 1


class TestNormalizePlan:
    def test_caps_counts(self) -> None:
        plan = PlanOutput(
            objective="o",
            sub_questions=[
                PlannedSubQuestion(text=f"Question {i}?", queries=[f"q{i}a", f"q{i}b", f"q{i}c"])
                for i in range(6)
            ],
        )
        _, items = normalize_plan(
            plan, config=PlannerConfig(max_sub_questions=3, max_queries_per_sub_question=2)
        )

        assert len(items) == 3
        assert all(len(item.queries) == 2 for item in items)

    def test_dedupes_case_insensitively(self) -> None:
        plan = PlanOutput(
            objective="o",
            sub_questions=[
                PlannedSubQuestion(text="What is RAG?", queries=["RAG", "rag", " RAG  "]),
                PlannedSubQuestion(text="what   is rag?", queries=["other"]),
            ],
        )
        _, items = normalize_plan(plan, config=PlannerConfig())

        assert len(items) == 1
        assert items[0].queries == ["RAG"]

    def test_empty_queries_use_sub_question_text(self) -> None:
        plan = PlanOutput(
            objective="o", sub_questions=[PlannedSubQuestion(text="What is HNSW?", queries=["", "  "])]
        )
        _, items = normalize_plan(plan, config=PlannerConfig())
        assert items[0].queries == ["What is HNSW?"]

    def test_blank_sub_questions_are_dropped(self) -> None:
        plan = PlanOutput(
            objective="o",
            sub_questions=[
                PlannedSubQuestion(text="   ", queries=["x"]),
                PlannedSubQuestion(text="Real?", queries=["y"]),
            ],
        )
        _, items = normalize_plan(plan, config=PlannerConfig())
        assert [item.text for item in items] == ["Real?"]

    def test_ids_are_unique(self) -> None:
        plan = PlanOutput(
            objective="o",
            sub_questions=[PlannedSubQuestion(text=f"Q{i}?", queries=["q"]) for i in range(4)],
        )
        _, items = normalize_plan(plan, config=PlannerConfig())
        assert len({item.id for item in items}) == 4

    async def test_empty_objective_falls_back_to_question(self) -> None:
        runtime = build_fake_runtime(
            responses={"PlanOutput": plan_json([{"text": "A?", "queries": ["a"]}], objective="  ")}
        )
        update = await run_planner(runtime)
        assert update["plan"]["objective"] == QUESTION

    def test_fallback_plan(self) -> None:
        objective, items = fallback_plan("  Is   RAG good? ")
        assert objective == "Is RAG good?"
        assert len(items) == 1
        assert items[0].text == "Is RAG good?"
        assert items[0].queries == ["Is RAG good?"]


class TestFallback:
    def assert_fallback(self, update: dict[str, Any], runtime: AgentRuntime) -> None:
        assert len(update["sub_questions"]) == 1
        assert update["sub_questions"][0]["text"] == QUESTION
        assert update["sub_questions"][0]["queries"] == [QUESTION]
        assert update["plan"]["fallback_reason"]
        # The fallback plan is still persisted so passages can reference it.
        assert set(store_of(runtime).sub_questions) == {update["sub_questions"][0]["id"]}
        types = [e["type"] for e in update["events"]]
        assert "error" in types
        assert update["events"][-1]["payload"]["fallback"] is True

    async def test_model_error(self) -> None:
        runtime = build_fake_runtime(responses={"PlanOutput": RuntimeError("boom")})
        update = await run_planner(runtime)

        self.assert_fallback(update, runtime)
        assert "boom" in update["plan"]["fallback_reason"]
        assert update["llm_calls"] == []

    async def test_invalid_json_exhausts_retries(self) -> None:
        runtime = build_fake_runtime(responses={"PlanOutput": "not json"})
        update = await run_planner(runtime)

        self.assert_fallback(update, runtime)
        assert "StructuredOutputError" in update["plan"]["fallback_reason"]
        # 1 call + structured_retries repair attempts, all billed.
        attempts = 1 + runtime.config.llm.structured_retries
        assert len(llm_of(runtime).calls) == attempts
        assert runtime.budget.calls == attempts

    async def test_plan_without_usable_sub_questions(self) -> None:
        runtime = build_fake_runtime(
            responses={"PlanOutput": plan_json([{"text": "   ", "queries": []}])}
        )
        update = await run_planner(runtime)
        self.assert_fallback(update, runtime)

    async def test_budget_exceeded_skips_the_llm(self) -> None:
        runtime = build_fake_runtime()
        runtime.budget.tokens_used = runtime.budget.max_tokens + 1

        update = await run_planner(runtime)

        self.assert_fallback(update, runtime)
        assert llm_of(runtime).calls == []
        assert "budget" in update["plan"]["fallback_reason"]


# ===========================================================================
# M5: research domains per mode
# ===========================================================================


def planned(*domains: str) -> PlanOutput:
    return PlanOutput(
        objective="o",
        sub_questions=[
            PlannedSubQuestion(text=f"Question {i}?", queries=[f"q{i}"], domain=domain)
            for i, domain in enumerate(domains)
        ],
    )


class TestDomains:
    def test_quick_mode_sends_everything_to_the_web_researcher(self) -> None:
        _, items = normalize_plan(
            planned("web", "technical", "market", "academic"),
            config=PlannerConfig(max_sub_questions=8),
            profile=MODES["quick"],
        )
        assert [i.domain for i in items] == ["web"] * 4

    def test_deep_mode_keeps_its_domains_and_maps_the_rest_to_web(self) -> None:
        _, items = normalize_plan(
            planned("technical", "market", "web", "academic", "astrology", " Market "),
            config=PlannerConfig(max_sub_questions=8),
            profile=MODES["deep"],
        )
        assert [i.domain for i in items] == ["technical", "market", "web", "web", "web", "market"]

    def test_aliases(self) -> None:
        _, items = normalize_plan(
            planned("engineering", "cost", "papers", "general"),
            config=PlannerConfig(max_sub_questions=8),
            profile=MODES["academic"],
        )
        assert [i.domain for i in items] == ["technical", "market", "academic", "web"]

    def test_a_plan_without_domains_defaults_to_web(self) -> None:
        plan = PlanOutput.model_validate_json(
            '{"objective": "o", "sub_questions": [{"text": "t?", "queries": ["q"]}]}'
        )
        _, items = normalize_plan(plan, config=PlannerConfig(), profile=MODES["deep"])
        assert items[0].domain == "web"

    def test_academic_mode_gives_the_first_sub_question_to_the_academic_researcher(self) -> None:
        _, items = normalize_plan(
            planned("web", "technical"), config=PlannerConfig(), profile=MODES["academic"]
        )
        assert [i.domain for i in items] == ["academic", "technical"]

    def test_academic_mode_keeps_an_academic_assignment(self) -> None:
        _, items = normalize_plan(
            planned("web", "academic"), config=PlannerConfig(), profile=MODES["academic"]
        )
        assert [i.domain for i in items] == ["web", "academic"]

    def test_fallback_plan_domain_per_mode(self) -> None:
        assert fallback_plan("Q?")[1][0].domain == "web"
        assert fallback_plan("Q?", profile=MODES["deep"])[1][0].domain == "web"
        assert fallback_plan("Q?", profile=MODES["academic"])[1][0].domain == "academic"

    async def test_node_assigns_and_stores_domains(self) -> None:
        runtime = build_fake_runtime(
            responses={
                "PlanOutput": plan_json(
                    [
                        {"text": "How fast is RAG?", "queries": ["rag latency"], "domain": "technical"},
                        {"text": "What does RAG cost?", "queries": ["rag cost"], "domain": "market"},
                        {"text": "Which papers compare them?", "queries": ["rag papers"], "domain": "academic"},
                    ]
                )
            }
        )

        update = await run_planner(runtime, mode="deep")

        assert [sq["domain"] for sq in update["sub_questions"]] == ["technical", "market", "web"]
        stored = store_of(runtime).sub_questions
        assert [stored[sq["id"]]["domain"] for sq in update["sub_questions"]] == [
            "technical",
            "market",
            "web",
        ]
        assert update["events"][-1]["payload"]["domains"] == {"technical": 1, "market": 1, "web": 1}

    async def test_fallback_in_academic_mode_goes_to_the_academic_researcher(self) -> None:
        runtime = build_fake_runtime(responses={"PlanOutput": RuntimeError("boom")})

        update = await run_planner(runtime, mode="academic")

        assert [sq["domain"] for sq in update["sub_questions"]] == ["academic"]

    async def test_prompt_lists_domains_only_when_the_mode_has_several(self) -> None:
        quick = build_fake_runtime()
        await run_planner(quick)
        quick_prompt = llm_of(quick).calls[0]["prompt"]
        assert '"domain"' not in quick_prompt
        assert "technical" not in quick_prompt

        deep = build_fake_runtime()
        await run_planner(deep, mode="deep")
        deep_prompt = llm_of(deep).calls[0]["prompt"]
        assert '"domain"' in deep_prompt
        assert '"technical"' in deep_prompt and '"market"' in deep_prompt
        assert '"academic"' not in deep_prompt
        assert "favours" not in deep_prompt

        academic = build_fake_runtime()
        await run_planner(academic, mode="academic")
        academic_prompt = llm_of(academic).calls[0]["prompt"]
        assert '"academic"' in academic_prompt
        assert 'favours "academic"' in academic_prompt

    async def test_unknown_mode_is_rejected(self) -> None:
        runtime = build_fake_runtime()
        try:
            await run_planner(runtime, mode="nonsense")
        except ValueError as exc:
            assert "not available" in str(exc)
        else:
            raise AssertionError("an unknown mode must be rejected")


# ===========================================================================
# M6 — per-mode bounds and gap-filling rounds
# ===========================================================================


class TestModeBounds:
    async def test_verified_modes_ask_for_four_to_eight(self) -> None:
        runtime = build_fake_runtime()
        await run_planner(runtime, mode="deep")
        prompt = llm_of(runtime).calls_for("PlanOutput")[0]["prompt"]
        assert "between 4 and 8 items" in prompt

    async def test_quick_mode_keeps_its_own_bounds(self) -> None:
        runtime = build_fake_runtime()
        await run_planner(runtime)
        assert "between 2 and 5 items" in llm_of(runtime).calls_for("PlanOutput")[0]["prompt"]

    def test_the_mode_cap_applies(self) -> None:
        plan = PlanOutput(
            objective="o",
            sub_questions=[PlannedSubQuestion(text=f"Question {i}?", queries=["q"]) for i in range(10)],
        )
        config = PlannerConfig(max_sub_questions=3, verified_max_sub_questions=7)
        assert len(normalize_plan(plan, config=config)[1]) == 3
        assert len(normalize_plan(plan, config=config, profile=MODES["deep"])[1]) == 7

    async def test_decision_mode_favours_market(self) -> None:
        runtime = build_fake_runtime()
        update = await run_planner(runtime, mode="decision")
        assert "market" in {sq["domain"] for sq in update["sub_questions"]}


def gap_json(*items: dict[str, Any]) -> str:
    return json.dumps({"sub_questions": list(items)})


async def run_gap_round(
    runtime: AgentRuntime,
    gaps: list[dict[str, Any]],
    *,
    existing: list[dict[str, Any]] | None = None,
    iteration: int = 1,
    mode: str = "deep",
) -> dict[str, Any]:
    """A planner call in round *iteration*, after a first plan of *existing* sub-questions."""
    items = [
        SubQuestionItem(id=f"sq{i}", text=sq["text"], queries=sq.get("queries", []), domain="web")
        for i, sq in enumerate(existing or [{"text": "What does RAG cost?", "queries": ["rag cost"]}])
    ]
    await runtime.store.save_plan(run_id="run-1", plan={"objective": "Compare."}, sub_questions=items)
    state = new_research_state(run_id="run-1", question=QUESTION, mode=mode, max_iterations=3)
    state["iteration"] = iteration
    state["plan"] = {"objective": "Compare.", "sub_questions": [i.model_dump() for i in items]}
    state["sub_questions"] = [i.model_dump() for i in items]
    state["sufficiency"] = {"status": "needs_more", "decision": "loop", "gaps": gaps}
    return await make_planner_node(runtime)(state)


SQ_GAP = {"label": "G1", "description": "Too little evidence for: What does RAG cost?", "sub_question_id": "sq0"}
TOPIC_GAP = {"label": "G2", "description": "2026 GPU prices for fine-tuning", "sub_question_id": None}


class TestGapRound:
    async def test_adds_new_sub_questions_for_the_gaps(self) -> None:
        runtime = build_fake_runtime(
            responses={
                "GapPlanOutput": gap_json(
                    {
                        "text": "What do vector databases cost per month?",
                        "queries": ["vector db pricing"],
                        "gap": "G1",
                        "domain": "market",
                    },
                    {"text": "What do GPUs for fine-tuning cost in 2026?", "queries": ["gpu prices 2026"], "gap": "G2"},
                )
            }
        )
        update = await run_gap_round(runtime, [SQ_GAP, TOPIC_GAP])

        old, filling, topic = update["sub_questions"]
        assert old["id"] == "sq0" and old["iteration"] == 0
        assert (filling["iteration"], filling["parent_id"], filling["domain"]) == (1, "sq0", "market")
        assert (topic["iteration"], topic["parent_id"]) == (1, None)
        # Only the new ones are stored as rows; the plan holds them all.
        store = store_of(runtime)
        assert {filling["id"], topic["id"]} <= set(store.sub_questions)
        assert store.iterations["run-1"] == 1
        assert [sq["id"] for sq in store.plans["run-1"]["sub_questions"]] == [old["id"], filling["id"], topic["id"]]
        [round_] = update["plan"]["rounds"]
        assert round_["iteration"] == 1 and round_["added"] == 2
        assert [c["prompt_version"] for c in update["llm_calls"]] == [GAP_PROMPT_VERSION]
        assert llm_of(runtime).calls_for("PlanOutput") == [], "a gap round never re-plans"

    async def test_the_prompt_shows_coverage_gaps_and_tried_queries(self) -> None:
        runtime = build_fake_runtime()
        await run_gap_round(runtime, [SQ_GAP, TOPIC_GAP])
        [call] = llm_of(runtime).calls_for("GapPlanOutput")
        prompt = call["prompt"]
        assert "What does RAG cost? (NOT covered: 0 verified claims from 0 sites)" in prompt
        assert "queries tried: rag cost" in prompt
        assert "[G2] 2026 GPU prices for fine-tuning" in prompt
        assert "UNTRUSTED_DATA" in prompt, "gap text derived from web pages is fenced"

    async def test_repeats_and_tried_queries_are_dropped(self) -> None:
        runtime = build_fake_runtime(
            responses={
                "GapPlanOutput": gap_json(
                    {"text": "what does  RAG cost?", "queries": ["x"], "gap": "G1"},
                    {"text": "What does RAG hosting cost?", "queries": ["RAG COST", "rag hosting price"], "gap": "G1"},
                    {"text": "What does RAG hosting cost?", "queries": ["dup"], "gap": "G1"},
                )
            }
        )
        update = await run_gap_round(runtime, [SQ_GAP])
        new = [sq for sq in update["sub_questions"] if sq["iteration"] == 1]
        assert [(sq["text"], sq["queries"]) for sq in new] == [("What does RAG hosting cost?", ["rag hosting price"])]

    async def test_at_most_max_gap_sub_questions(self) -> None:
        runtime = build_fake_runtime(
            AgentConfig(planner=PlannerConfig(max_gap_sub_questions=1)),
            responses={
                "GapPlanOutput": gap_json(
                    {"text": "A new question?", "queries": ["a"], "gap": "G1"},
                    {"text": "Another new question?", "queries": ["b"], "gap": "G1"},
                )
            },
        )
        update = await run_gap_round(runtime, [SQ_GAP])
        assert len(update["sub_questions"]) == 2

    async def test_an_unknown_gap_label_makes_a_top_level_sub_question(self) -> None:
        runtime = build_fake_runtime(
            responses={"GapPlanOutput": gap_json({"text": "Something new?", "queries": ["n"], "gap": "G9"})}
        )
        update = await run_gap_round(runtime, [SQ_GAP])
        assert update["sub_questions"][-1]["parent_id"] is None

    async def test_model_failure_adds_nothing(self) -> None:
        runtime = build_fake_runtime(responses={"GapPlanOutput": RuntimeError("down")})
        update = await run_gap_round(runtime, [SQ_GAP])
        assert [sq["id"] for sq in update["sub_questions"]] == ["sq0"]
        assert update["plan"]["rounds"][-1]["fallback_reason"].startswith("RuntimeError")
        assert update["events"][-1]["payload"]["fallback"] is True

    async def test_spent_budget_adds_nothing_without_a_call(self) -> None:
        runtime = build_fake_runtime(AgentConfig(max_tokens_per_run=10, budget_synthesis_reserve=0.0))
        runtime.budget.tokens_used = 10
        update = await run_gap_round(runtime, [SQ_GAP])
        assert llm_of(runtime).calls_for("GapPlanOutput") == []
        assert len(update["sub_questions"]) == 1

    async def test_no_gaps_no_call(self) -> None:
        runtime = build_fake_runtime()
        update = await run_gap_round(runtime, [])
        assert llm_of(runtime).calls_for("GapPlanOutput") == []
        assert update["plan"]["rounds"][-1]["fallback_reason"] == "no gaps to fill"

    async def test_a_store_failure_drops_the_new_sub_questions(self) -> None:
        runtime = build_fake_runtime(
            responses={"GapPlanOutput": gap_json({"text": "New?", "queries": ["n"], "gap": "G1"})}
        )
        original = runtime.store.save_plan
        calls = 0

        async def flaky(**kwargs: Any) -> None:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("db down")
            await original(**kwargs)

        runtime.store.save_plan = flaky  # type: ignore[method-assign]
        update = await run_gap_round(runtime, [SQ_GAP])
        assert [sq["id"] for sq in update["sub_questions"]] == ["sq0"]
        assert any(e["type"] == "error" for e in update["events"])


class TestStableIds:
    """M8: a planner re-run after a resume must not store a second set of sub-questions."""

    async def test_ids_come_from_run_round_and_position(self) -> None:
        runtime = build_fake_runtime(
            responses={"PlanOutput": plan_json([{"text": "A?", "queries": ["a"]}, {"text": "B?", "queries": ["b"]}])}
        )
        update = await run_planner(runtime)
        assert [sq["id"] for sq in update["sub_questions"]] == [
            stable_id("run-1", "sub_question", 0, 0),
            stable_id("run-1", "sub_question", 0, 1),
        ]

    async def test_a_re_run_replaces_its_rows(self) -> None:
        runtime = build_fake_runtime(
            responses={"PlanOutput": [plan_json([{"text": "A?", "queries": ["a"]}]), plan_json([{"text": "A again?", "queries": ["a"]}])]}
        )
        first = await run_planner(runtime)
        second = await run_planner(runtime)
        assert first["sub_questions"][0]["id"] == second["sub_questions"][0]["id"]
        rows = store_of(runtime).sub_questions
        assert len(rows) == 1 and next(iter(rows.values()))["text"] == "A again?"

    async def test_the_fallback_plan_is_keyed_the_same_way(self) -> None:
        runtime = build_fake_runtime(responses={"PlanOutput": RuntimeError("down")})
        update = await run_planner(runtime)
        assert update["sub_questions"][0]["id"] == stable_id("run-1", "sub_question", 0, 0)

    async def test_gap_round_ids_include_the_round(self) -> None:
        runtime = build_fake_runtime(
            responses={"GapPlanOutput": gap_json({"text": "New?", "queries": ["new"], "gap": "G2"})}
        )
        update = await run_gap_round(runtime, [TOPIC_GAP], iteration=2)
        assert update["sub_questions"][-1]["id"] == stable_id("run-1", "sub_question", 2, 0)
