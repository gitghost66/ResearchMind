"""
LangGraph assembly for the research workflow.

The graph is built for one mode (``modes.py``). Quick mode (M3)::

    START -> planner --(one Send per sub-question)--> web_researcher --> synthesizer -> END
                    \\--(budget spent / no plan)----------------------->/

Verified modes, deep, academic, and decision (M5, M6)::

    START -> planner --(Send per new sub-question, to its domain's researcher)-->
                 web_researcher | technical_researcher | market_researcher | academic_researcher
             --> claim_extractor -> fact_checker -> contradiction_detector
             -> sufficiency_checker --(needs more, round allowed)--> planner (gap-filling)
                                    \\--(sufficient, or no round allowed)--> synthesizer -> END

Researchers run as parallel branches, one per sub-question; LangGraph waits
for every branch before the next node runs, and the list reducers in
:class:`ResearchState` merge their results. Branch concurrency is capped by
``concurrency_cap`` via the run config.

Each pass through the loop is a research round (``iteration`` 0, 1, ...). The
planner then adds sub-questions for the gaps, and only those are researched;
the claim extractor, fact checker, and contradiction detector likewise work
on the round's new material. A gap round that adds nothing goes straight to
the synthesizer. A looping mode runs at most ``max_iterations`` rounds.

Nodes receive the runtime through closures, so the graph itself holds no
clients, sessions, or secrets, and its state stays checkpointable.

Tracing: every invocation carries a run name, tags, and ``run_id`` metadata
(:func:`build_run_config`). With LangSmith enabled through its environment
variables, a run is one trace and each node is a span under it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from researchmind_agents.modes import (
    ACADEMIC,
    MARKET,
    MODES,
    TECHNICAL,
    WEB,
    mode_profile,
    normalize_domain,
)
from researchmind_agents.nodes.academic_researcher import make_academic_researcher_node
from researchmind_agents.nodes.claim_extractor import make_claim_extractor_node
from researchmind_agents.nodes.common import TaskNode
from researchmind_agents.nodes.contradiction_detector import make_contradiction_detector_node
from researchmind_agents.nodes.fact_checker import make_fact_checker_node
from researchmind_agents.nodes.planner import make_planner_node
from researchmind_agents.nodes.sufficiency import LOOP, make_sufficiency_node
from researchmind_agents.nodes.synthesizer import make_synthesizer_node
from researchmind_agents.nodes.web_researcher import (
    make_market_researcher_node,
    make_technical_researcher_node,
    make_web_researcher_node,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import ResearchState, WebResearchTask, new_research_state

logger = logging.getLogger(__name__)

__all__ = [
    "PLANNER",
    "WEB_RESEARCHER",
    "TECHNICAL_RESEARCHER",
    "MARKET_RESEARCHER",
    "ACADEMIC_RESEARCHER",
    "CLAIM_EXTRACTOR",
    "FACT_CHECKER",
    "CONTRADICTION_DETECTOR",
    "SUFFICIENCY_CHECKER",
    "SYNTHESIZER",
    "RESEARCHER_NODES",
    "build_research_graph",
    "build_run_config",
    "make_research_router",
    "make_loop_router",
    "max_iterations_for",
    "pipeline_steps",
    "run_research",
]

PLANNER = "planner"
WEB_RESEARCHER = "web_researcher"
TECHNICAL_RESEARCHER = "technical_researcher"
MARKET_RESEARCHER = "market_researcher"
ACADEMIC_RESEARCHER = "academic_researcher"
CLAIM_EXTRACTOR = "claim_extractor"
FACT_CHECKER = "fact_checker"
CONTRADICTION_DETECTOR = "contradiction_detector"
SUFFICIENCY_CHECKER = "sufficiency_checker"
SYNTHESIZER = "synthesizer"

# Supersteps per research round (planner, researchers, extractor, checker,
# detector, sufficiency), for the recursion limit.
_STEPS_PER_ROUND = 6

# Research domain -> the node (and agent name) that researches it.
RESEARCHER_NODES: dict[str, str] = {
    WEB: WEB_RESEARCHER,
    TECHNICAL: TECHNICAL_RESEARCHER,
    MARKET: MARKET_RESEARCHER,
    ACADEMIC: ACADEMIC_RESEARCHER,
}

_RESEARCHER_FACTORIES: dict[str, Callable[[AgentRuntime], TaskNode]] = {
    WEB: make_web_researcher_node,
    TECHNICAL: make_technical_researcher_node,
    MARKET: make_market_researcher_node,
    ACADEMIC: make_academic_researcher_node,
}


def pipeline_steps(mode: str) -> tuple[str, ...]:
    """
    The agents a run in *mode* passes through, in order (empty if unknown).
    Node names double as the ``agent`` of the events each node emits, so
    progress can be read from events.
    """
    profile = MODES.get(mode)
    if profile is None:
        return ()
    steps = [PLANNER, *(RESEARCHER_NODES[d] for d in profile.domains)]
    if profile.verify:
        steps += [CLAIM_EXTRACTOR, FACT_CHECKER, CONTRADICTION_DETECTOR]
    if profile.loop:
        steps.append(SUFFICIENCY_CHECKER)
    steps.append(SYNTHESIZER)
    return tuple(steps)


def max_iterations_for(runtime: AgentRuntime, mode: str) -> int:
    """Research rounds a run in *mode* may take: 1 unless the mode loops."""
    return max(1, runtime.config.max_iterations) if mode_profile(mode).loop else 1


def make_research_router(
    runtime: AgentRuntime, mode: str = "quick"
) -> Callable[[ResearchState], list[Send] | str]:
    """
    Fan out one researcher branch per sub-question of the current round, or
    skip to synthesis (none, or the research budget is spent).
    """
    profile = mode_profile(mode)

    def route(state: ResearchState) -> list[Send] | str:
        iteration = int(state.get("iteration", 0))
        sub_questions = [
            sq for sq in state.get("sub_questions") or [] if int(sq.get("iteration", 0)) == iteration
        ]
        if not sub_questions or runtime.budget.research_exhausted:
            return SYNTHESIZER
        return [
            Send(
                RESEARCHER_NODES[normalize_domain(sub_question.get("domain", WEB), profile)],
                WebResearchTask(
                    run_id=state["run_id"],
                    question=state["question"],
                    sub_question=sub_question,
                ),
            )
            for sub_question in sub_questions
        ]

    return route


def make_loop_router() -> Callable[[ResearchState], str]:
    """After the sufficiency check: another round (planner) or the report."""

    def route(state: ResearchState) -> str:
        decision = (state.get("sufficiency") or {}).get("decision")
        return PLANNER if decision == LOOP else SYNTHESIZER

    return route


def build_research_graph(
    runtime: AgentRuntime,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    mode: str = "quick",
) -> Any:
    """Compile the graph for *mode*. Uses an in-memory checkpointer by default."""
    profile = mode_profile(mode)
    researchers = [RESEARCHER_NODES[d] for d in profile.domains]

    graph = StateGraph(ResearchState)
    graph.add_node(PLANNER, make_planner_node(runtime))
    for domain in profile.domains:
        graph.add_node(RESEARCHER_NODES[domain], _RESEARCHER_FACTORIES[domain](runtime))
    graph.add_node(SYNTHESIZER, make_synthesizer_node(runtime))

    after_research = SYNTHESIZER
    if profile.verify:
        graph.add_node(CLAIM_EXTRACTOR, make_claim_extractor_node(runtime))
        graph.add_node(FACT_CHECKER, make_fact_checker_node(runtime))
        graph.add_node(CONTRADICTION_DETECTOR, make_contradiction_detector_node(runtime))
        graph.add_edge(CLAIM_EXTRACTOR, FACT_CHECKER)
        graph.add_edge(FACT_CHECKER, CONTRADICTION_DETECTOR)
        if profile.loop:
            graph.add_node(SUFFICIENCY_CHECKER, make_sufficiency_node(runtime))
            graph.add_edge(CONTRADICTION_DETECTOR, SUFFICIENCY_CHECKER)
            graph.add_conditional_edges(
                SUFFICIENCY_CHECKER, make_loop_router(), [PLANNER, SYNTHESIZER]
            )
        else:
            graph.add_edge(CONTRADICTION_DETECTOR, SYNTHESIZER)
        after_research = CLAIM_EXTRACTOR

    graph.add_edge(START, PLANNER)
    graph.add_conditional_edges(
        PLANNER, make_research_router(runtime, mode), [*researchers, SYNTHESIZER]
    )
    # Every branch has finished before the next node runs: all Sends execute
    # in the same step, and an edge triggers its target once per step.
    for researcher in researchers:
        graph.add_edge(researcher, after_research)
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

    The run's checkpoints are kept under its ``run_id``. If *checkpointer*
    already holds some, the run resumes from the latest: steps that finished
    are not run again, and in a fan-out step only the branches that had not
    finished re-run. A run whose graph had already ended returns that final
    state without running anything.

    Each step's checkpoint is written before the next step starts
    (``durability="sync"``), so an interruption loses at most the step in
    progress.
    """
    mode_profile(mode)  # raises ValueError for an unknown mode
    cleaned = " ".join(question.split())
    if not cleaned:
        raise ValueError("question must not be empty")

    saver: BaseCheckpointSaver[Any] = checkpointer if checkpointer is not None else InMemorySaver()
    graph = build_research_graph(runtime, checkpointer=saver, mode=mode)
    config = build_run_config(runtime, run_id=run_id, mode=mode)

    if await saver.aget_tuple(config) is not None:
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            logger.info("Run %s had already finished; nothing to resume", run_id)
            return cast(ResearchState, dict(snapshot.values))
        logger.info("Resuming run %s at %s", run_id, ", ".join(snapshot.next))
        result = await graph.ainvoke(None, config=config, durability="sync")
        return cast(ResearchState, result)

    initial = new_research_state(
        run_id=run_id,
        question=cleaned,
        mode=mode,
        user_context=user_context,
        max_iterations=max_iterations_for(runtime, mode),
    )
    result = await graph.ainvoke(initial, config=config, durability="sync")
    return cast(ResearchState, result)


def build_run_config(runtime: AgentRuntime, *, run_id: str, mode: str) -> RunnableConfig:
    """
    LangGraph config for one run: checkpoint thread, branch concurrency, a
    recursion limit that fits the mode's rounds, and the name, tags, and
    metadata a tracer groups the run's spans under.
    """
    rounds = max_iterations_for(runtime, mode) if mode in MODES else 1
    return {
        "configurable": {"thread_id": run_id},
        "max_concurrency": max(1, runtime.config.concurrency_cap),
        "recursion_limit": 10 + _STEPS_PER_ROUND * rounds,
        "run_name": f"research:{mode}",
        "tags": ["researchmind", f"mode:{mode}"],
        "metadata": {"run_id": run_id, "mode": mode},
    }
