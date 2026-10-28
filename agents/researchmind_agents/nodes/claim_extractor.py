"""
Claim extractor node: retrieved passages -> atomic claims linked to passages.

1. **Retrieve** the most relevant passages for each sub-question (run-scoped
   vector search, shared with the synthesizer). Each passage is read once,
   for the sub-question it matched best.
2. **Extract**, one model call per sub-question, run concurrently under the
   concurrency cap. Passages are shown as labels ``P1``, ``P2``, ...; each
   claim lists the labels it came from.
3. **Link**: labels map back to passage IDs. A claim without a valid label is
   dropped, so no claim exists without a supporting passage (PRD s6.2). The
   same fact found twice is kept once, with all of its passages.

Claims leave this node ``unverified``; the fact checker judges and stores
them. A failed call costs only its sub-question's claims, and with the
research budget spent nothing is extracted (the report then falls back to
passages).

**Rounds.** In a gap-filling round only that round's new sub-questions are
read, and only passages no earlier round read (a passage counts as read once
a model call over it succeeded; a failed call leaves its passages for a later
round). Earlier claims are kept as they are; a new claim worded like an
earlier usable one is dropped (one that failed verification may be found
again). Each new claim records its round (``ClaimItem.iteration``), which is
what the fact checker and contradiction detector act on.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from researchmind_agents.nodes.common import EventLog, StateNode, llm_call_record
from researchmind_agents.nodes.retrieval import Evidence, gather_evidence
from researchmind_agents.prompts.claim_extractor import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    PromptPassage,
    build_extraction_prompt,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    USABLE_VERDICTS,
    ClaimExtractionOutput,
    ClaimItem,
    EvidenceItem,
    ResearchState,
    SubQuestionItem,
)
from researchmind_agents.tools.urls import domain_of

__all__ = ["AGENT", "make_claim_extractor_node", "claim_key", "clean_claim_text"]

AGENT = "claim_extractor"

# Sanity bounds on one claim's text: shorter is a fragment, longer is not atomic.
_MIN_CLAIM_CHARS = 12
_MAX_CLAIM_CHARS = 600

_INLINE_LABELS = re.compile(r"\s*[\[(]\s*P\d+(?:\s*[,;]\s*P\d+)*\s*[\])]", re.IGNORECASE)
_LABEL = re.compile(r"^P(\d+)$", re.IGNORECASE)
_KEY_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class _Group:
    sub_question: SubQuestionItem
    passages: list[Evidence] = field(default_factory=list)


def make_claim_extractor_node(runtime: AgentRuntime) -> StateNode:
    async def claim_extractor_node(state: ResearchState) -> dict[str, Any]:
        run_id = state["run_id"]
        iteration = int(state.get("iteration", 0))
        log = EventLog(runtime, run_id=run_id, agent=AGENT)
        await log.emit("agent_started", iteration=iteration)

        previous = [ClaimItem.model_validate(c) for c in state.get("claims", [])]
        read = list(state.get("extracted_passage_ids", []))
        if runtime.budget.research_exhausted:
            await log.emit(
                "agent_finished", iteration=iteration, claims=0, skipped="research budget exhausted"
            )
            return {"claims": [c.model_dump() for c in previous], "events": log.events, "llm_calls": []}

        sub_questions = [
            sq
            for sq in (SubQuestionItem.model_validate(raw) for raw in state.get("sub_questions", []))
            if sq.iteration == iteration
        ]
        config = runtime.config.claims
        evidence = await gather_evidence(
            runtime,
            state,
            sub_questions,
            log,
            top_k=config.passages_per_sub_question,
            max_passages=config.max_passages,
            min_similarity=runtime.config.synthesizer.min_similarity,
            exclude=read,
        )
        groups = _group(sub_questions, evidence)

        semaphore = asyncio.Semaphore(max(1, runtime.config.concurrency_cap))
        llm_calls: list[dict[str, Any]] = []

        async def run(group: _Group) -> list[ClaimItem] | None:
            async with semaphore:
                return await _extract(runtime, state, group, log, llm_calls)

        per_group = await asyncio.gather(*(run(g) for g in groups))
        found = _merge_duplicates([c for claims in per_group if claims for c in claims])
        known = {claim_key(c.text) for c in previous if c.verdict in USABLE_VERDICTS}
        new = [
            c.model_copy(update={"iteration": iteration})
            for c in found
            if claim_key(c.text) not in known
        ]

        await log.emit(
            "agent_finished",
            iteration=iteration,
            claims=len(new),
            duplicates_of_earlier=len(found) - len(new),
            sub_questions=len(groups),
            passages=len(evidence),
        )
        seen = set(read)
        newly_read = [
            item.passage_id
            for group, claims in zip(groups, per_group, strict=True)
            if claims is not None
            for item in group.passages
            if item.passage_id not in seen
        ]
        return {
            "claims": [c.model_dump() for c in [*previous, *new]],
            "extracted_passage_ids": [*read, *newly_read],
            "events": log.events,
            "llm_calls": llm_calls,
        }

    return claim_extractor_node


def _group(sub_questions: list[SubQuestionItem], evidence: list[Evidence]) -> list[_Group]:
    """Passages per sub-question (each passage under its best match), in plan order."""
    groups = {sq.id: _Group(sub_question=sq) for sq in sub_questions}
    for item in evidence:
        group = groups.get(item.primary_sub_question_id or "")
        if group is not None:
            group.passages.append(item)
    return [g for g in groups.values() if g.passages]


async def _extract(
    runtime: AgentRuntime,
    state: ResearchState,
    group: _Group,
    log: EventLog,
    llm_calls: list[dict[str, Any]],
) -> list[ClaimItem] | None:
    """The group's claims; ``None`` when the model did not read it (failure, budget)."""
    run_id = state["run_id"]
    sub_question = group.sub_question
    labeled = {f"P{i}": item for i, item in enumerate(group.passages, start=1)}
    prompt = build_extraction_prompt(
        question=state["question"],
        sub_question=sub_question.text,
        passages=[
            PromptPassage(label=label, title=item.title, domain=domain_of(item.url), text=item.text)
            for label, item in labeled.items()
        ],
        max_claims=runtime.config.claims.max_claims_per_sub_question,
    )
    if runtime.budget.research_exhausted:
        return None
    try:
        async with asyncio.timeout(runtime.config.agent_timeout):
            response = await runtime.llm.generate_structured(
                prompt,
                ClaimExtractionOutput,
                system=SYSTEM_PROMPT,
                agent=AGENT,
                run_id=run_id,
                max_output_tokens=runtime.config.claims.max_output_tokens,
            )
    except Exception as exc:  # noqa: BLE001 - lose this sub-question's claims only
        await log.error(
            "Claim extraction failed",
            sub_question_id=sub_question.id,
            reason=f"{type(exc).__name__}: {exc}",
        )
        return None

    llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
    claims: list[ClaimItem] = []
    unlinked = 0
    for extracted in response.parsed.claims if response.parsed else []:
        text = clean_claim_text(extracted.text)
        if not _MIN_CLAIM_CHARS <= len(text) <= _MAX_CLAIM_CHARS:
            continue
        passages = _resolve(extracted.passages, labeled)
        if not passages:
            unlinked += 1
            continue
        claims.append(
            ClaimItem(
                id=str(uuid.uuid4()),
                text=text,
                sub_question_id=sub_question.id,
                passage_ids=[p.passage_id for p in passages],
                evidence=[
                    EvidenceItem(passage_id=p.passage_id, source_id=p.source_id, stance="supports")
                    for p in passages
                ],
            )
        )
    claims = _merge_duplicates(claims)[: runtime.config.claims.max_claims_per_sub_question]

    await log.emit(
        "claims_extracted",
        sub_question_id=sub_question.id,
        claims=len(claims),
        passages=len(labeled),
        dropped_without_passage=unlinked,
    )
    return claims


def _resolve(raw_labels: list[str], labeled: dict[str, Evidence]) -> list[Evidence]:
    found: list[Evidence] = []
    for raw in raw_labels:
        match = _LABEL.match(raw.strip().strip("[]()").strip())
        item = labeled.get(f"P{int(match.group(1))}") if match else None
        if item is not None and item not in found:
            found.append(item)
    return found


def clean_claim_text(text: str) -> str:
    """Collapse whitespace and drop passage labels the model wrote inline."""
    cleaned = _INLINE_LABELS.sub("", text)
    cleaned = " ".join(cleaned.split()).lstrip("-*• ").strip()
    return cleaned


def claim_key(text: str) -> str:
    """Identity of a claim's wording: lower-case words, punctuation ignored."""
    return " ".join(_KEY_WORD.findall(text.lower()))


def _merge_duplicates(claims: list[ClaimItem]) -> list[ClaimItem]:
    """Keep the first claim per wording; later copies add their passages to it."""
    merged: dict[str, ClaimItem] = {}
    for claim in claims:
        key = claim_key(claim.text)
        existing = merged.get(key)
        if existing is None:
            merged[key] = claim
            continue
        known = set(existing.passage_ids)
        for link in claim.evidence:
            if link.passage_id not in known:
                known.add(link.passage_id)
                existing.passage_ids.append(link.passage_id)
                existing.evidence.append(link)
    return list(merged.values())
