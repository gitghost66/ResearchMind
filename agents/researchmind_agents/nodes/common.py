"""Helpers shared by graph nodes: events and LLM-call records."""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Protocol

from researchmind_agents.llm.base import LLMUsage
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import ResearchState, WebResearchTask

logger = logging.getLogger(__name__)

__all__ = ["EventLog", "StateNode", "TaskNode", "llm_call_record"]


# LangGraph's node protocol names its parameter ``state``; a bare Callable type
# has no parameter names, so these spell it out for the type checker.
class StateNode(Protocol):
    def __call__(self, state: ResearchState) -> Awaitable[dict[str, Any]]: ...


class TaskNode(Protocol):
    def __call__(self, state: WebResearchTask) -> Awaitable[dict[str, Any]]: ...


class EventLog:
    """
    Collects a node's events for the graph state and persists each one as it
    happens, so progress is visible (and, from M4, streamable) mid-run.

    Persistence is telemetry: a store failure is logged, never raised.
    """

    def __init__(self, runtime: AgentRuntime, *, run_id: str, agent: str) -> None:
        self._runtime = runtime
        self._run_id = run_id
        self._agent = agent
        self.events: list[dict[str, Any]] = []

    async def emit(self, event_type: str, **payload: Any) -> None:
        event = {
            "ts": datetime.now(UTC).isoformat(),
            "agent": self._agent,
            "type": event_type,
            "payload": payload,
        }
        self.events.append(event)
        try:
            await self._runtime.store.log_event(
                run_id=self._run_id,
                agent=self._agent,
                event_type=event_type,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must not break research
            logger.warning(
                "Failed to persist %s event for run=%s: %s", event_type, self._run_id, exc
            )

    async def error(self, message: str, **payload: Any) -> None:
        await self.emit("error", message=message, **payload)


def llm_call_record(usage: LLMUsage, *, prompt_version: str) -> dict[str, Any]:
    """JSON-safe summary of one logical LLM call, for ``state['llm_calls']``."""
    return {
        "agent": usage.agent,
        "model": usage.model,
        "prompt_version": prompt_version,
        "tokens_in": usage.prompt_tokens,
        "tokens_out": usage.completion_tokens,
        "cost_usd": usage.cost_usd,
        "latency_ms": round(usage.latency_ms),
        "attempts": usage.attempts,
    }
