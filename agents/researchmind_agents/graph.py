"""
LangGraph assembly for the research workflow.

M3 (quick mode)::

    START -> planner --(one Send per sub-question)--> web_researcher --> synthesizer -> END
                    \\--(budget spent / no plan)----------------------->/

The web researcher runs as parallel branches, one per sub-question; LangGraph
waits for every branch before the synthesizer runs, and the list reducers in
:class:`ResearchState` merge their results. Branch concurrency is capped by
``concurrency_cap`` via the run config.

Nodes receive the runtime through closures, so the graph itself holds no
clients, sessions, or secrets, and its state stays checkpointable.

Tracing: every invocation carries a run name, tags, and ``run_id`` metadata
(:func:`build_run_config`). With LangSmith enabled through its environment
variables, a run is one trace and each node is a span under it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from researchmind_agents.nodes.planner import make_planner_node
from researchmind_agents.nodes.synthesizer import make_synthesizer_node
from researchmind_agents.nodes.web_researcher import make_web_researcher_node
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    SUPPORTED_MODES,
    ResearchState,
    WebResearchTask,
    new_research_state,
)

__all__ = [
    "PLANNER",
    "WEB_RESEARCHER",
    "SYNTHESIZER",
    "build_research_graph",
    "build_run_config",
    "make_research_router",
    "pipeline_steps",
    "run_research",
]

PLANNER = "planner"
WEB_RESEARCHER = "web_researcher"
SYNTHESIZER = "synthesizer"

# Nodes a run passes through, in order, per mode. Node names double as the
# ``agent`` of the events each node emits, so progress can be read from events.
_PIPELINE_STEPS: dict[str, tuple[str, ...]] = {
    "quick": (PLANNER, WEB_RESEARCHER, SYNTHESIZER),
}


def pipeline_steps(mode: str) -> tuple[str, ...]:
    """The agents a run in *mode* passes through, in order (empty if unknown)."""
    return _PIPELINE_STEPS.get(mode, ())


def make_research_router(runtime: AgentRuntime) -> Callable[[ResearchState], list[Send] | str]:
    """Fan out one web-researcher branch per sub-question, or skip to synthesis."""

    def route(state: ResearchState) -> list[Send] | str:
        sub_questions = state.get("sub_questions") or []
        if not sub_questions or runtime.budget.research_exhausted:
            return SYNTHESIZER
        return [
            Send(
                WEB_RESEARCHER,
                WebResearchTask(
                    run_id=state["run_id"],
                    question=state["question"],
                    sub_question=sub_question,
                ),
            )
            for sub_question in sub_questions
        ]

    return route


def build_research_graph(
    runtime: AgentRuntime,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the quick-mode graph. Uses an in-memory checkpointer by default."""
    graph = StateGraph(ResearchState)
    graph.add_node(PLANNER, make_planner_node(runtime))
    graph.add_node(WEB_RESEARCHER, make_web_researcher_node(runtime))
    graph.add_node(SYNTHESIZER, make_synthesizer_node(runtime))

    graph.add_edge(START, PLANNER)
    graph.add_conditional_edges(
        PLANNER, make_research_router(runtime), [WEB_RESEARCHER, SYNTHESIZER]
    )
    graph.add_edge(WEB_RESEARCHER, SYNTHESIZER)
    graph.add_edge(SYNTHESIZER, END)

    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


async def run_research(
    runtime: AgentRuntime,
    *,
    run_id: str,
    question: str,
    mode: str = "quick",
    user_context: dict[str, Any] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> ResearchState:
    """
    Run one research question to completion and return the final state.

    ``state["report"]`` holds the report JSON and ``state["report_markdown"]``
    its markdown; both are also stored through ``runtime.store``.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Mode {mode!r} is not available yet; supported: {', '.join(sorted(SUPPORTED_MODES))}"
        )
    cleaned = " ".join(question.split())
    if not cleaned:
        raise ValueError("question must not be empty")

    graph = build_research_graph(runtime, checkpointer=checkpointer)
    initial = new_research_state(
        run_id=run_id,
        question=cleaned,
        mode=mode,
        user_context=user_context,
        max_iterations=1,
    )
    result = await graph.ainvoke(initial, config=build_run_config(runtime, run_id=run_id, mode=mode))
    return cast(ResearchState, result)


def build_run_config(runtime: AgentRuntime, *, run_id: str, mode: str) -> RunnableConfig:
    """
    LangGraph config for one run: checkpoint thread, branch concurrency, and
    the name, tags, and metadata a tracer groups the run's spans under.
    """
    return {
        "configurable": {"thread_id": run_id},
        "max_concurrency": max(1, runtime.config.concurrency_cap),
        "run_name": f"research:{mode}",
        "tags": ["researchmind", f"mode:{mode}"],
        "metadata": {"run_id": run_id, "mode": mode},
    }
