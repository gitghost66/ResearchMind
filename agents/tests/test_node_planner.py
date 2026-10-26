"""M3 tests — planner node (mocked LLM, in-memory store)."""

from __future__ import annotations

import json
from typing import Any

from researchmind_agents.config import AgentConfig, PlannerConfig
from researchmind_agents.nodes.planner import fallback_plan, make_planner_node, normalize_plan
from researchmind_agents.prompts.planner import PROMPT_VERSION
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import PlannedSubQuestion, PlanOutput, new_research_state
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
