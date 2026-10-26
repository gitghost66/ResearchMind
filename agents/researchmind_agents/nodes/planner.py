"""
Planner node: question -> plan of sub-questions with search queries.

If the model fails (error, timeout, invalid output after retries) or the budget
is already spent, the run continues with a one-item fallback plan — the
question itself, searched verbatim — rather than failing (PRD s6.3).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from researchmind_agents.config import PlannerConfig
from researchmind_agents.nodes.common import EventLog, StateNode, llm_call_record
from researchmind_agents.prompts.planner import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_planner_prompt,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import PlanOutput, ResearchState, SubQuestionItem

__all__ = ["AGENT", "make_planner_node", "normalize_plan", "fallback_plan"]

AGENT = "planner"


def make_planner_node(runtime: AgentRuntime) -> StateNode:
    async def planner_node(state: ResearchState) -> dict[str, Any]:
        run_id = state["run_id"]
        question = state["question"]
        config = runtime.config.planner
        log = EventLog(runtime, run_id=run_id, agent=AGENT)
        await log.emit("agent_started", question=question)

        llm_calls: list[dict[str, Any]] = []
        fallback_reason: str | None = None
        objective = question
        items: list[SubQuestionItem] = []

        if runtime.budget.exceeded:
            fallback_reason = "budget exceeded before planning"
        else:
            prompt = build_planner_prompt(
                question=question,
                min_sub_questions=config.min_sub_questions,
                max_sub_questions=config.max_sub_questions,
                max_queries=config.max_queries_per_sub_question,
                user_context=state.get("user_context"),
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
                    objective, items = normalize_plan(response.parsed, config=config)
                    objective = objective or question
                if not items:
                    fallback_reason = "plan contained no usable sub-questions"
            except Exception as exc:  # noqa: BLE001 - degrade to the fallback plan
                fallback_reason = f"{type(exc).__name__}: {exc}"

        if fallback_reason is not None:
            await log.error("Planner fell back to a single-question plan", reason=fallback_reason)
            objective, items = fallback_plan(question)

        plan = {
            "objective": objective,
            "sub_questions": [item.model_dump() for item in items],
            "prompt_version": PROMPT_VERSION,
            "fallback_reason": fallback_reason,
        }
        # Passages reference sub-questions, so this write must succeed.
        await runtime.store.save_plan(run_id=run_id, plan=plan, sub_questions=items)

        await log.emit(
            "agent_finished",
            sub_questions=len(items),
            below_minimum=len(items) < config.min_sub_questions,
            fallback=fallback_reason is not None,
        )
        return {
            "plan": plan,
            "sub_questions": [item.model_dump() for item in items],
            "events": log.events,
            "llm_calls": llm_calls,
        }

    return planner_node


def normalize_plan(
    plan: PlanOutput,
    *,
    config: PlannerConfig,
) -> tuple[str, list[SubQuestionItem]]:
    """
    Clean the model's plan: collapse whitespace, drop duplicate sub-questions
    and queries, cap counts at the configured maxima, and give every
    sub-question at least one query (its own text).
    """
    items: list[SubQuestionItem] = []
    seen: set[str] = set()
    for planned in plan.sub_questions:
        text = _squash(planned.text)
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())

        queries = _dedupe([_squash(q) for q in planned.queries if _squash(q)])
        queries = queries[: max(1, config.max_queries_per_sub_question)] or [text]
        items.append(SubQuestionItem(id=str(uuid.uuid4()), text=text, queries=queries))
        if len(items) >= config.max_sub_questions:
            break

    objective = _squash(plan.objective)
    return objective, items


def fallback_plan(question: str) -> tuple[str, list[SubQuestionItem]]:
    """A one-item plan that researches the question as asked."""
    text = _squash(question)
    return text, [SubQuestionItem(id=str(uuid.uuid4()), text=text, queries=[text])]


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
