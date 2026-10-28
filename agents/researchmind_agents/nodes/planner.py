"""
Planner node: question -> plan of sub-questions with search queries.

**First round.** Each sub-question is tagged with a research domain (web,
technical, market, academic) from those the run's mode allows; the graph
routes it to that domain's researcher. In a mode that favours a domain, at
least one sub-question gets it. Quick mode asks for ``min/max_sub_questions``
(2-5 by default), the verified modes for ``verified_min/max_sub_questions``
(4-8, PRD s6.2).

If the model fails (error, timeout, invalid output after retries) or the budget
is already spent, the run continues with a one-item fallback plan — the
question itself, searched verbatim — rather than failing (PRD s6.3).

**Gap-filling rounds** (looping modes, ``iteration`` > 0). The sufficiency
check found gaps; the planner gets the sub-questions so far with their
coverage, plus the gaps, and proposes at most ``max_gap_sub_questions`` *new*
sub-questions (PRD s6.2). A new sub-question that fills a sub-question's gap
records it as ``parent_id``, so its claims count toward that sub-question's
coverage. Repeats of existing sub-questions and of queries already tried are
dropped. If the model fails, the budget is spent, or the new sub-questions
cannot be stored, the round adds nothing and the run goes on to synthesis.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any

from researchmind_agents.config import PlannerConfig
from researchmind_agents.coverage import assess_coverage
from researchmind_agents.modes import (
    DOMAIN_DESCRIPTIONS,
    MODES,
    ModeProfile,
    mode_profile,
    normalize_domain,
)
from researchmind_agents.nodes.common import EventLog, StateNode, llm_call_record
from researchmind_agents.prompts.planner import (
    GAP_PROMPT_VERSION,
    GAP_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    PromptGap,
    PromptSubQuestion,
    build_gap_prompt,
    build_planner_prompt,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    ClaimItem,
    GapItem,
    GapPlanOutput,
    PlanOutput,
    ResearchState,
    SubQuestionItem,
)

__all__ = ["AGENT", "make_planner_node", "normalize_plan", "normalize_gap_plan", "fallback_plan"]

_QUICK = MODES["quick"]

AGENT = "planner"

# "G2", "[G2]", "G2: pricing", "gap G2" ...
_GAP_LABEL = re.compile(r"\bG(\d+)\b", re.IGNORECASE)


def make_planner_node(runtime: AgentRuntime) -> StateNode:
    async def planner_node(state: ResearchState) -> dict[str, Any]:
        if int(state.get("iteration", 0)) > 0:
            return await _gap_round(runtime, state)
        return await _first_round(runtime, state)

    return planner_node


# ---------------------------------------------------------------------------
# First round
# ---------------------------------------------------------------------------


async def _first_round(runtime: AgentRuntime, state: ResearchState) -> dict[str, Any]:
    run_id = state["run_id"]
    question = state["question"]
    config = runtime.config.planner
    profile = mode_profile(state.get("mode", "quick"))
    minimum, maximum = config.bounds(verified=profile.verify)
    log = EventLog(runtime, run_id=run_id, agent=AGENT)
    await log.emit("agent_started", question=question, iteration=0)

    llm_calls: list[dict[str, Any]] = []
    fallback_reason: str | None = None
    objective = question
    items: list[SubQuestionItem] = []

    if runtime.budget.exceeded:
        fallback_reason = "budget exceeded before planning"
    else:
        prompt = build_planner_prompt(
            question=question,
            min_sub_questions=minimum,
            max_sub_questions=maximum,
            max_queries=config.max_queries_per_sub_question,
            user_context=state.get("user_context"),
            domains={d: DOMAIN_DESCRIPTIONS[d] for d in profile.domains},
            favoured=profile.favoured,
        )
        try:
            async with asyncio.timeout(runtime.config.agent_timeout):
                response = await runtime.llm.generate_structured(
                    prompt,
                    PlanOutput,
                    system=SYSTEM_PROMPT,
                    agent=AGENT,
                    run_id=run_id,
                )
            llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
            if response.parsed is not None:
                objective, items = normalize_plan(response.parsed, config=config, profile=profile)
                objective = objective or question
            if not items:
                fallback_reason = "plan contained no usable sub-questions"
        except Exception as exc:  # noqa: BLE001 - degrade to the fallback plan
            fallback_reason = f"{type(exc).__name__}: {exc}"

    if fallback_reason is not None:
        await log.error("Planner fell back to a single-question plan", reason=fallback_reason)
        objective, items = fallback_plan(question, profile=profile)

    plan = {
        "objective": objective,
        "sub_questions": [item.model_dump() for item in items],
        "prompt_version": PROMPT_VERSION,
        "fallback_reason": fallback_reason,
    }
    # Passages reference sub-questions, so this write must succeed.
    await runtime.store.save_plan(run_id=run_id, plan=plan, sub_questions=items, iteration=0)

    domains: dict[str, int] = {}
    for item in items:
        domains[item.domain] = domains.get(item.domain, 0) + 1
    await log.emit(
        "agent_finished",
        iteration=0,
        sub_questions=len(items),
        domains=domains,
        below_minimum=len(items) < minimum,
        fallback=fallback_reason is not None,
    )
    return {
        "plan": plan,
        "sub_questions": [item.model_dump() for item in items],
        "events": log.events,
        "llm_calls": llm_calls,
    }


def normalize_plan(
    plan: PlanOutput,
    *,
    config: PlannerConfig,
    profile: ModeProfile = _QUICK,
) -> tuple[str, list[SubQuestionItem]]:
    """
    Clean the model's plan: collapse whitespace, drop duplicate sub-questions
    and queries, cap counts at the mode's maxima, give every sub-question at
    least one query (its own text), and map each domain to one the mode
    allows. In a mode that favours a domain, the first sub-question gets it
    when the model assigned it to none.
    """
    _, maximum = config.bounds(verified=profile.verify)
    items: list[SubQuestionItem] = []
    seen: set[str] = set()
    for planned in plan.sub_questions:
        text = _squash(planned.text)
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())

        queries = _dedupe([_squash(q) for q in planned.queries if _squash(q)])
        queries = queries[: max(1, config.max_queries_per_sub_question)] or [text]
        items.append(
            SubQuestionItem(
                id=str(uuid.uuid4()),
                text=text,
                queries=queries,
                domain=normalize_domain(planned.domain, profile),
            )
        )
        if len(items) >= maximum:
            break

    if items and profile.favoured and not any(i.domain == profile.favoured[0] for i in items):
        items[0] = items[0].model_copy(update={"domain": profile.favoured[0]})

    objective = _squash(plan.objective)
    return objective, items


def fallback_plan(
    question: str, *, profile: ModeProfile = _QUICK
) -> tuple[str, list[SubQuestionItem]]:
    """A one-item plan that researches the question as asked."""
    text = _squash(question)
    domain = profile.favoured[0] if profile.favoured else profile.default_domain
    return text, [SubQuestionItem(id=str(uuid.uuid4()), text=text, queries=[text], domain=domain)]


# ---------------------------------------------------------------------------
# Gap-filling rounds
# ---------------------------------------------------------------------------


async def _gap_round(runtime: AgentRuntime, state: ResearchState) -> dict[str, Any]:
    run_id = state["run_id"]
    question = state["question"]
    iteration = int(state.get("iteration", 0))
    profile = mode_profile(state.get("mode", "quick"))
    config = runtime.config.planner
    existing = [SubQuestionItem.model_validate(sq) for sq in state.get("sub_questions", [])]
    gaps = [GapItem.model_validate(g) for g in (state.get("sufficiency") or {}).get("gaps", [])]
    previous_plan = dict(state.get("plan") or {})
    objective = str(previous_plan.get("objective") or question)

    log = EventLog(runtime, run_id=run_id, agent=AGENT)
    await log.emit("agent_started", question=question, iteration=iteration, gaps=len(gaps))

    llm_calls: list[dict[str, Any]] = []
    added: list[SubQuestionItem] = []
    failure: str | None = None
    if not gaps:
        failure = "no gaps to fill"
    elif runtime.budget.research_exhausted:
        failure = "research budget exhausted"
    else:
        prompt = build_gap_prompt(
            question=question,
            objective=objective,
            sub_questions=_coverage_for_prompt(runtime, state, existing),
            gaps=[PromptGap(label=g.label, description=g.description) for g in gaps],
            max_new=max(1, config.max_gap_sub_questions),
            max_queries=config.max_queries_per_sub_question,
            user_context=state.get("user_context"),
            domains={d: DOMAIN_DESCRIPTIONS[d] for d in profile.domains},
        )
        try:
            async with asyncio.timeout(runtime.config.agent_timeout):
                response = await runtime.llm.generate_structured(
                    prompt,
                    GapPlanOutput,
                    system=GAP_SYSTEM_PROMPT,
                    agent=AGENT,
                    run_id=run_id,
                )
            llm_calls.append(llm_call_record(response.usage, prompt_version=GAP_PROMPT_VERSION))
            if response.parsed is not None:
                added = normalize_gap_plan(
                    response.parsed,
                    existing=existing,
                    gaps=gaps,
                    config=config,
                    profile=profile,
                    iteration=iteration,
                )
            if not added:
                failure = "no new sub-questions"
        except Exception as exc:  # noqa: BLE001 - the run goes on to synthesis
            failure = f"{type(exc).__name__}: {exc}"
            await log.error("Gap planning failed", reason=failure, iteration=iteration)

    rounds = list(previous_plan.get("rounds") or [])
    rounds.append(
        {
            "iteration": iteration,
            "gaps": [g.description for g in gaps],
            "added": len(added),
            "prompt_version": GAP_PROMPT_VERSION,
            "fallback_reason": failure,
        }
    )
    all_items = [*existing, *added]
    plan = {
        **previous_plan,
        "sub_questions": [item.model_dump() for item in all_items],
        "rounds": rounds,
    }
    try:
        # The run's iteration is the latest round that researched something:
        # a round that adds nothing is logged in the plan, not counted.
        await runtime.store.save_plan(
            run_id=run_id,
            plan=plan,
            sub_questions=added,
            iteration=iteration if added else max(sq.iteration for sq in existing),
        )
    except Exception as exc:  # noqa: BLE001 - unstored sub-questions cannot be researched
        await log.error(
            "Failed to store gap-filling sub-questions",
            reason=f"{type(exc).__name__}: {exc}",
            iteration=iteration,
        )
        added = []
        all_items = existing
        plan = {**plan, "sub_questions": [item.model_dump() for item in existing]}

    await log.emit(
        "agent_finished",
        iteration=iteration,
        sub_questions=len(added),
        gaps=len(gaps),
        fallback=not added,
    )
    return {
        "plan": plan,
        "sub_questions": [item.model_dump() for item in all_items],
        "events": log.events,
        "llm_calls": llm_calls,
    }


def _coverage_for_prompt(
    runtime: AgentRuntime, state: ResearchState, sub_questions: list[SubQuestionItem]
) -> list[PromptSubQuestion]:
    sufficiency = runtime.config.sufficiency
    coverage = assess_coverage(
        sub_questions,
        [ClaimItem.model_validate(c) for c in state.get("claims", [])],
        {s["id"]: s.get("url", "") for s in state.get("sources", [])},
        min_claims=sufficiency.min_claims_per_sub_question,
        min_sites=sufficiency.min_sites_per_sub_question,
        min_confidence=runtime.config.synthesizer.min_claim_confidence,
    )
    return [
        PromptSubQuestion(
            position=item.position,
            text=item.sub_question.text,
            covered=item.covered,
            claims=item.claims,
            sites=item.sites,
            queries=item.queries,
        )
        for item in coverage
    ]


def normalize_gap_plan(
    plan: GapPlanOutput,
    *,
    existing: list[SubQuestionItem],
    gaps: list[GapItem],
    config: PlannerConfig,
    profile: ModeProfile = _QUICK,
    iteration: int = 1,
) -> list[SubQuestionItem]:
    """
    The model's gap-filling sub-questions, cleaned: repeats of existing
    sub-questions and of queries already tried are dropped, counts are capped,
    and each one's gap becomes its ``parent_id`` when the gap concerns a
    sub-question.
    """
    seen = {sq.text.lower() for sq in existing}
    tried = {query.lower() for sq in existing for query in sq.queries}
    by_label = {gap.label.upper(): gap for gap in gaps}

    items: list[SubQuestionItem] = []
    for planned in plan.sub_questions:
        if len(items) >= max(0, config.max_gap_sub_questions):
            break
        text = _squash(planned.text)
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())

        queries = _dedupe([_squash(q) for q in planned.queries if _squash(q)])
        queries = [q for q in queries if q.lower() not in tried]
        queries = queries[: max(1, config.max_queries_per_sub_question)]
        if not queries:
            if text.lower() in tried:
                continue
            queries = [text]
        tried.update(q.lower() for q in queries)

        gap = by_label.get(_gap_label(planned.gap) or "")
        items.append(
            SubQuestionItem(
                id=str(uuid.uuid4()),
                text=text,
                queries=queries,
                domain=normalize_domain(planned.domain, profile),
                iteration=iteration,
                parent_id=gap.sub_question_id if gap is not None else None,
            )
        )
    return items


def _gap_label(raw: str) -> str | None:
    match = _GAP_LABEL.search(raw)
    return f"G{int(match.group(1))}" if match else None


def _squash(text: str) -> str:
    return " ".join(text.split())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
