"""
Shared LangGraph state — the Pydantic model passed between all agent nodes.

This is a placeholder for M0. Full state schema will be implemented in M3.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    """Shared state for the LangGraph research workflow."""

    # Core
    run_id: str = ""
    question: str = ""
    mode: str = "quick"  # quick | deep | academic | decision
    user_context: dict[str, Any] = Field(default_factory=dict)

    # Plan (populated by Planner agent)
    plan: dict[str, Any] = Field(default_factory=dict)

    # Collected data (populated by research agents)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    passages: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)

    # Control
    iteration: int = 0
    max_iterations: int = 3

    # Output
    report: dict[str, Any] = Field(default_factory=dict)

    # Cost tracking
    usage: dict[str, Any] = Field(default_factory=dict)

    # Events log
    events: list[dict[str, Any]] = Field(default_factory=list)
