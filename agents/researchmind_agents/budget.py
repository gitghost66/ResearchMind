"""
Per-run token and cost budget (PRD s10, cost control).

The tracker sits in front of the usage sink, so it sees every billed LLM call,
including structured-output retries and calls whose output later failed
validation. Nodes ask it two questions:

* :attr:`BudgetTracker.research_exhausted` — research should stop and the run
  should skip to synthesis. Trips at ``(1 - synthesis_reserve)`` of either
  limit, which leaves headroom for the report.
* :attr:`BudgetTracker.exceeded` — the hard cap is spent; no further LLM calls.
  The synthesizer then builds an extractive report without a model.

A single call can overshoot a limit (its size is unknown until it returns), so
the cap is enforced between calls, never mid-call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from researchmind_agents.config import AgentConfig
from researchmind_agents.llm.base import LLMUsage, UsageSink

logger = logging.getLogger(__name__)

__all__ = ["BudgetTracker"]


@dataclass
class BudgetTracker:
    """Accumulates LLM usage for one run and reports when limits are reached."""

    # A limit of 0 (or less) disables that limit.
    max_tokens: int = 0
    max_cost_usd: float = 0.0
    synthesis_reserve: float = 0.0

    tokens_used: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    per_agent: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: AgentConfig) -> BudgetTracker:
        return cls(
            max_tokens=config.max_tokens_per_run,
            max_cost_usd=config.max_cost_per_run_usd,
            synthesis_reserve=config.budget_synthesis_reserve,
        )

    # -- recording -----------------------------------------------------

    def record(self, usage: LLMUsage) -> None:
        tokens = usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)
        self.tokens_used += tokens
        self.cost_usd = round(self.cost_usd + usage.cost_usd, 10)
        self.calls += 1

        agent = self.per_agent.setdefault(
            usage.agent,
            {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0},
        )
        agent["calls"] += 1
        agent["tokens_in"] += usage.prompt_tokens
        agent["tokens_out"] += usage.completion_tokens
        agent["cost_usd"] = round(agent["cost_usd"] + usage.cost_usd, 10)

    def wrap(self, sink: UsageSink | None = None) -> UsageSink:
        """
        Return a sink that records usage here, then forwards it to *sink*.

        Recording happens first and cannot fail, so a broken downstream sink
        never hides spend from the budget.
        """

        async def tracking_sink(usage: LLMUsage) -> None:
            self.record(usage)
            if sink is not None:
                await sink(usage)

        return tracking_sink

    # -- limits --------------------------------------------------------

    @property
    def exceeded(self) -> bool:
        """The hard cap is spent."""
        return self._over(1.0)

    @property
    def research_exhausted(self) -> bool:
        """Research must stop so the reserve is left for synthesis."""
        reserve = min(max(self.synthesis_reserve, 0.0), 1.0)
        return self._over(1.0 - reserve)

    def _over(self, fraction: float) -> bool:
        if self.max_tokens > 0 and self.tokens_used >= self.max_tokens * fraction:
            return True
        return self.max_cost_usd > 0 and self.cost_usd >= self.max_cost_usd * fraction

    # -- reporting -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "research_exhausted": self.research_exhausted,
            "exceeded": self.exceeded,
            "per_agent": {name: dict(values) for name, values in self.per_agent.items()},
        }
