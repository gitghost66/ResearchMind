"""M3 tests — per-run token and cost budget."""

from __future__ import annotations

import pytest

from researchmind_agents.budget import BudgetTracker
from researchmind_agents.config import AgentConfig
from researchmind_agents.llm.base import LLMUsage


def usage(
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost: float = 0.0,
    agent: str = "planner",
    total: int = 0,
) -> LLMUsage:
    return LLMUsage(
        prompt_tokens=tokens_in,
        completion_tokens=tokens_out,
        total_tokens=total,
        cost_usd=cost,
        agent=agent,
    )


class TestRecording:
    def test_accumulates_totals_and_per_agent(self) -> None:
        budget = BudgetTracker()
        budget.record(usage(100, 50, 0.01, agent="planner"))
        budget.record(usage(200, 20, 0.02, agent="synthesizer"))
        budget.record(usage(10, 5, 0.005, agent="planner"))

        assert budget.calls == 3
        assert budget.tokens_used == 385
        assert budget.cost_usd == pytest.approx(0.035)
        assert budget.per_agent["planner"] == {
            "calls": 2,
            "tokens_in": 110,
            "tokens_out": 55,
            "cost_usd": pytest.approx(0.015),
        }
        assert budget.per_agent["synthesizer"]["calls"] == 1

    def test_prefers_reported_total_tokens(self) -> None:
        budget = BudgetTracker()
        budget.record(usage(10, 5, total=40))
        assert budget.tokens_used == 40

    def test_summary(self) -> None:
        budget = BudgetTracker(max_tokens=1000, max_cost_usd=1.0)
        budget.record(usage(100, 50, 0.1, agent="planner"))

        summary = budget.summary()

        assert summary["calls"] == 1
        assert summary["tokens_used"] == 150
        assert summary["cost_usd"] == pytest.approx(0.1)
        assert summary["max_tokens"] == 1000
        assert summary["max_cost_usd"] == 1.0
        assert summary["exceeded"] is False
        assert summary["research_exhausted"] is False
        assert summary["per_agent"]["planner"]["tokens_in"] == 100

    def test_summary_is_a_copy(self) -> None:
        budget = BudgetTracker()
        budget.record(usage(1, 1))
        summary = budget.summary()
        summary["per_agent"]["planner"]["calls"] = 99
        assert budget.per_agent["planner"]["calls"] == 1


class TestLimits:
    def test_token_limit(self) -> None:
        budget = BudgetTracker(max_tokens=1000, synthesis_reserve=0.2)
        budget.record(usage(700, 0))
        assert not budget.research_exhausted
        assert not budget.exceeded

        budget.record(usage(100, 0))  # 800 = 80% of the cap
        assert budget.research_exhausted
        assert not budget.exceeded

        budget.record(usage(200, 0))
        assert budget.exceeded

    def test_cost_limit(self) -> None:
        budget = BudgetTracker(max_cost_usd=1.0, synthesis_reserve=0.25)
        budget.record(usage(cost=0.5))
        assert not budget.research_exhausted

        budget.record(usage(cost=0.25))
        assert budget.research_exhausted
        assert not budget.exceeded

        budget.record(usage(cost=0.3))
        assert budget.exceeded

    def test_either_limit_trips(self) -> None:
        budget = BudgetTracker(max_tokens=10_000, max_cost_usd=0.1)
        budget.record(usage(10, 10, cost=0.2))
        assert budget.exceeded

    def test_zero_disables_a_limit(self) -> None:
        budget = BudgetTracker(max_tokens=0, max_cost_usd=0.0)
        budget.record(usage(10_000_000, 0, cost=1000.0))
        assert not budget.exceeded
        assert not budget.research_exhausted

    def test_zero_token_limit_still_enforces_cost(self) -> None:
        budget = BudgetTracker(max_tokens=0, max_cost_usd=0.5)
        budget.record(usage(10_000_000, 0, cost=0.1))
        assert not budget.exceeded
        budget.record(usage(cost=0.5))
        assert budget.exceeded

    def test_reserve_clamped_above_one(self) -> None:
        # A reserve >= 1 would make research exhausted before any spend.
        budget = BudgetTracker(max_tokens=100, synthesis_reserve=5.0)
        assert budget.research_exhausted  # 0 >= 100 * 0
        budget_ok = BudgetTracker(max_tokens=100, synthesis_reserve=1.0)
        assert budget_ok.research_exhausted == budget.research_exhausted

    def test_reserve_clamped_below_zero(self) -> None:
        budget = BudgetTracker(max_tokens=100, synthesis_reserve=-1.0)
        budget.record(usage(99, 0))
        # Negative reserve behaves like 0: research runs to the hard cap.
        assert not budget.research_exhausted
        budget.record(usage(1, 0))
        assert budget.research_exhausted
        assert budget.exceeded

    def test_fresh_tracker_is_not_exhausted(self) -> None:
        budget = BudgetTracker(max_tokens=100, max_cost_usd=1.0, synthesis_reserve=0.2)
        assert not budget.research_exhausted
        assert not budget.exceeded


class TestWrap:
    async def test_records_then_forwards(self) -> None:
        budget = BudgetTracker()
        seen: list[tuple[int, int]] = []

        async def downstream(item: LLMUsage) -> None:
            # Recording happens before forwarding.
            seen.append((item.prompt_tokens, budget.tokens_used))

        sink = budget.wrap(downstream)
        await sink(usage(10, 5))

        assert seen == [(10, 15)]
        assert budget.calls == 1

    async def test_without_downstream(self) -> None:
        budget = BudgetTracker()
        await budget.wrap()(usage(3, 4))
        assert budget.tokens_used == 7

    async def test_failing_downstream_still_counts(self) -> None:
        budget = BudgetTracker()

        async def broken(_: LLMUsage) -> None:
            raise RuntimeError("db down")

        sink = budget.wrap(broken)
        # The wrapped sink lets the error out; BaseLLMProvider._record swallows it.
        with pytest.raises(RuntimeError, match="db down"):
            await sink(usage(10, 10))

        assert budget.tokens_used == 20
        assert budget.calls == 1


class TestFromConfig:
    def test_reads_run_limits(self) -> None:
        config = AgentConfig(
            max_tokens_per_run=1234, max_cost_per_run_usd=0.5, budget_synthesis_reserve=0.3
        )
        budget = BudgetTracker.from_config(config)

        assert budget.max_tokens == 1234
        assert budget.max_cost_usd == 0.5
        assert budget.synthesis_reserve == 0.3
        assert budget.tokens_used == 0


class TestSeed:
    """M8: a resumed run's budget counts what earlier attempts spent."""

    def test_seeded_usage_counts_toward_the_limits(self) -> None:
        tracker = BudgetTracker(max_tokens=1000, synthesis_reserve=0.2)
        tracker.seed(tokens=850, cost_usd=0.25, calls=4)
        assert (tracker.tokens_used, tracker.cost_usd, tracker.calls) == (850, 0.25, 4)
        assert tracker.research_exhausted and not tracker.exceeded

    def test_negative_values_are_ignored(self) -> None:
        tracker = BudgetTracker()
        tracker.seed(tokens=-5, cost_usd=-1.0, calls=-2)
        assert (tracker.tokens_used, tracker.cost_usd, tracker.calls) == (0, 0.0, 0)
