"""
Sufficiency checker node: is the research complete, or should another round
fill its gaps? (PRD s6.2: deterministic rules first, LLM judgement second.)

1. **Rules** (``coverage.py``): every top-level sub-question of the first
   plan needs at least ``min_claims_per_sub_question`` usable claims from at
   least ``min_sites_per_sub_question`` independent sites; claims from
   gap-filling sub-questions count toward the one they fill. Each one short of
   that is a gap, and the run ``needs_more``. (A later round's sub-question for
   a gap the model named is not held to the rules; the model judges it.)
2. **Model**: when the rules pass, the model judges whether the claims answer
   the question; it may name up to ``max_gaps`` missing aspects. A failed or
   empty judgement leaves the rules' verdict (``sufficient``) standing.

The decision is ``loop`` (to the planner, with the gaps) only when the run
needs more *and* another round may start: ``iteration + 1 < max_iterations``,
the research budget is not spent, and the mode's loop deadline has not
passed. Otherwise it is ``synthesize``; the model is not asked when no round
could follow. The node raises ``iteration`` when it loops.
"""

from __future__ import annotations

import asyncio
from typing import Any

from researchmind_agents.coverage import SubQuestionCoverage, assess_coverage
from researchmind_agents.nodes.common import EventLog, StateNode, llm_call_record
from researchmind_agents.prompts.sufficiency import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    PromptCoverage,
    build_sufficiency_prompt,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    ClaimItem,
    ContradictionItem,
    GapItem,
    ResearchState,
    SubQuestionItem,
    SufficiencyOutput,
)

__all__ = ["AGENT", "LOOP", "SYNTHESIZE", "make_sufficiency_node", "rule_gaps", "loop_blocker"]

AGENT = "sufficiency_checker"
LOOP = "loop"
SYNTHESIZE = "synthesize"

_MAX_GAP_CHARS = 300


def make_sufficiency_node(runtime: AgentRuntime) -> StateNode:
    async def sufficiency_node(state: ResearchState) -> dict[str, Any]:
        run_id = state["run_id"]
        iteration = int(state.get("iteration", 0))
        log = EventLog(runtime, run_id=run_id, agent=AGENT)
        await log.emit("agent_started", iteration=iteration)

        config = runtime.config.sufficiency
        sub_questions = [SubQuestionItem.model_validate(sq) for sq in state.get("sub_questions", [])]
        claims = [ClaimItem.model_validate(c) for c in state.get("claims", [])]
        coverage = assess_coverage(
            sub_questions,
            claims,
            {s["id"]: s.get("url", "") for s in state.get("sources", [])},
            min_claims=config.min_claims_per_sub_question,
            min_sites=config.min_sites_per_sub_question,
            min_confidence=runtime.config.synthesizer.min_claim_confidence,
        )
        blocker = loop_blocker(runtime, state)

        llm_calls: list[dict[str, Any]] = []
        gaps = rule_gaps(coverage)
        checked_by = "rules"
        reason = (
            f"{len(gaps)} sub-question(s) lack verified evidence"
            if gaps
            else "every sub-question has enough verified evidence"
        )
        if not coverage:
            reason = "the plan has no sub-questions"
        elif not gaps and blocker is None and config.llm_check:
            judged = await _judge(runtime, state, coverage, claims, log, llm_calls)
            if judged is not None:
                checked_by = "rules+llm"
                reason = " ".join(judged.reason.split())[:_MAX_GAP_CHARS] or reason
                if not judged.sufficient:
                    gaps = _model_gaps(judged.gaps, limit=config.max_gaps)

        status = "needs_more" if gaps else "sufficient"
        decision = LOOP if gaps and blocker is None else SYNTHESIZE
        result = {
            "iteration": iteration,
            "status": status,
            "decision": decision,
            "reason": reason,
            "stopped_by": blocker if gaps else None,
            "checked_by": checked_by,
            "gaps": [g.model_dump() for g in gaps],
            "coverage": [
                {
                    "sub_question_id": item.sub_question.id,
                    "claims": item.claims,
                    "sites": item.sites,
                    "covered": item.covered,
                }
                for item in coverage
            ],
        }
        await log.emit(
            "sufficiency_checked",
            iteration=iteration,
            status=status,
            decision=decision,
            stopped_by=result["stopped_by"],
            checked_by=checked_by,
            gaps=[g.description for g in gaps],
        )
        await log.emit("agent_finished", iteration=iteration, status=status, decision=decision)

        update: dict[str, Any] = {
            "sufficiency": result,
            "events": log.events,
            "llm_calls": llm_calls,
        }
        if decision == LOOP:
            update["iteration"] = iteration + 1
        return update

    return sufficiency_node


def loop_blocker(runtime: AgentRuntime, state: ResearchState) -> str | None:
    """Why no further round may start, or ``None`` if one may."""
    iteration = int(state.get("iteration", 0))
    if iteration + 1 >= int(state.get("max_iterations", 1)):
        return "max_iterations"
    if runtime.budget.research_exhausted:
        return "budget"
    deadline = runtime.config.sufficiency.loop_deadline(state.get("mode", "quick"))
    if deadline > 0 and runtime.elapsed() >= deadline:
        return "deadline"
    return None


def rule_gaps(coverage: list[SubQuestionCoverage]) -> list[GapItem]:
    """One gap per required sub-question the rules do not consider covered."""
    gaps: list[GapItem] = []
    for item in coverage:
        if item.covered or not item.required:
            continue
        claims = f"{item.claims} verified claim" + ("" if item.claims == 1 else "s")
        sites = f"{item.sites} site" + ("" if item.sites == 1 else "s")
        gaps.append(
            GapItem(
                label=f"G{len(gaps) + 1}",
                description=(
                    f"Sub-question {item.position} has too little verified evidence "
                    f"({claims} from {sites}): {item.sub_question.text}"
                ),
                sub_question_id=item.sub_question.id,
            )
        )
    return gaps


def _model_gaps(raw: list[str], *, limit: int) -> list[GapItem]:
    gaps: list[GapItem] = []
    seen: set[str] = set()
    for text in raw:
        if len(gaps) >= max(0, limit):
            break
        description = " ".join(text.split())[:_MAX_GAP_CHARS]
        if not description or description.lower() in seen:
            continue
        seen.add(description.lower())
        gaps.append(GapItem(label=f"G{len(gaps) + 1}", description=description))
    return gaps


async def _judge(
    runtime: AgentRuntime,
    state: ResearchState,
    coverage: list[SubQuestionCoverage],
    claims: list[ClaimItem],
    log: EventLog,
    llm_calls: list[dict[str, Any]],
) -> SufficiencyOutput | None:
    config = runtime.config.sufficiency
    by_id = {c.id: c for c in claims}
    shown = max(1, config.claims_shown_per_sub_question)
    contradictions = [
        ContradictionItem.model_validate(c) for c in state.get("contradictions", [])
    ]
    unresolved = [
        " vs ".join(f'"{by_id[cid].text[:160]}"' for cid in c.claim_ids if cid in by_id)
        + f" ({c.likely_cause}): {c.explanation[:200]}"
        for c in contradictions
        if not c.resolved
    ]
    prompt = build_sufficiency_prompt(
        question=state["question"],
        objective=str((state.get("plan") or {}).get("objective") or state["question"]),
        coverage=[
            PromptCoverage(
                position=item.position,
                text=item.sub_question.text,
                sites=item.sites,
                claims=[
                    (by_id[cid].text[:240], by_id[cid].verdict) for cid in item.claim_ids[:shown]
                ],
                total_claims=item.claims,
            )
            for item in coverage
        ],
        contradictions=unresolved,
        max_gaps=max(1, config.max_gaps),
    )
    try:
        async with asyncio.timeout(runtime.config.agent_timeout):
            response = await runtime.llm.generate_structured(
                prompt,
                SufficiencyOutput,
                system=SYSTEM_PROMPT,
                agent=AGENT,
                run_id=state["run_id"],
                max_output_tokens=config.max_output_tokens,
            )
    except Exception as exc:  # noqa: BLE001 - the rules' verdict stands
        await log.error("Sufficiency judgement failed", reason=f"{type(exc).__name__}: {exc}")
        return None
    llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
    return response.parsed
