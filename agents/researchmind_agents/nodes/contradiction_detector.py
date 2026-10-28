"""
Contradiction detector node: verified claims -> contradiction records.

1. **Candidates** (deterministic, :func:`find_candidates`): pairs of usable
   claims (supported, weak, mixed), at least one of them from this round:

   * *linked* — the fact checker judged a passage that states one claim as
     contradicting the other claim;
   * *similar* — the claims' texts are at least ``min_similarity`` alike
     (cosine, one batched embedding call).

   Pairs whose claims came from the same passage are skipped (one passage
   stating two facts), and so are pairs already in a recorded contradiction.
   At most ``max_pairs``: linked first, then the most similar.
2. **Topics** (:func:`group_pairs`): candidate pairs are joined into groups of
   at most ``max_group_size`` claims; a pair that would overflow a group
   becomes a group of its own.
3. **Judge**: ``groups_per_call`` topics per model call, concurrently under the
   concurrency cap. Claims are labels (``C1``, ...); each is shown with its
   verdict and supporting excerpts, with their sources' type and date, so the
   model can name the cause.
4. **Record**: a contradiction needs two or more claims of one topic, and
   none of its claim pairs may already be in a recorded contradiction (so one
   conflict is never listed, or penalised, twice). Whether it is resolved
   comes from its cause (``ScoringConfig.resolved_causes``), never from the
   model. Records are stored and streamed as ``contradiction_found``.

A failed call loses only its topics; with the research budget spent nothing
is compared. The core, :func:`detect_contradictions`, needs no graph state, so
evals can call it.
"""

from __future__ import annotations

import asyncio
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from researchmind_agents.nodes.common import EventLog, StateNode, llm_call_record
from researchmind_agents.nodes.fact_checker import SourceInfo, sources_from_state
from researchmind_agents.prompts.contradiction_detector import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    PromptClaim,
    PromptExcerpt,
    PromptGroup,
    build_contradiction_prompt,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    USABLE_VERDICTS,
    ClaimItem,
    ContradictionItem,
    ContradictionOutput,
    ResearchState,
)
from researchmind_agents.tools.excerpts import best_excerpt
from researchmind_agents.tools.urls import domain_of

__all__ = [
    "AGENT",
    "CandidatePair",
    "Detection",
    "make_contradiction_detector_node",
    "find_candidates",
    "group_pairs",
    "detect_contradictions",
]

AGENT = "contradiction_detector"

_LABEL = re.compile(r"^C(\d+)$", re.IGNORECASE)
_MAX_EXPLANATION_CHARS = 600


@dataclass(frozen=True)
class CandidatePair:
    a: str
    b: str
    similarity: float
    # The fact checker found one claim's passage contradicting the other.
    linked: bool = False


@dataclass
class Detection:
    contradictions: list[ContradictionItem] = field(default_factory=list)
    candidates: list[CandidatePair] = field(default_factory=list)
    groups: int = 0
    failed_groups: int = 0


def make_contradiction_detector_node(runtime: AgentRuntime) -> StateNode:
    async def contradiction_detector_node(state: ResearchState) -> dict[str, Any]:
        run_id = state["run_id"]
        iteration = int(state.get("iteration", 0))
        log = EventLog(runtime, run_id=run_id, agent=AGENT)
        await log.emit("agent_started", iteration=iteration)

        claims = [ClaimItem.model_validate(c) for c in state.get("claims", [])]
        existing = [ContradictionItem.model_validate(c) for c in state.get("contradictions", [])]
        usable = [c for c in claims if c.verdict in USABLE_VERDICTS]
        if not any(c.iteration == iteration for c in usable) or len(usable) < 2:
            await log.emit("agent_finished", iteration=iteration, candidates=0, contradictions=0)
            return {"events": log.events, "llm_calls": []}
        if runtime.budget.research_exhausted:
            await log.emit(
                "agent_finished",
                iteration=iteration,
                contradictions=0,
                skipped="research budget exhausted",
            )
            return {"events": log.events, "llm_calls": []}

        llm_calls: list[dict[str, Any]] = []
        passages = {p["id"]: p.get("text", "") for p in state.get("passages", [])}
        result = await detect_contradictions(
            runtime,
            run_id=run_id,
            question=state["question"],
            claims=usable,
            sources=sources_from_state(state),
            passages=passages,
            existing=existing,
            iteration=iteration,
            log=log,
            llm_calls=llm_calls,
        )

        if result.contradictions:
            try:
                await runtime.store.save_contradictions(
                    run_id=run_id, contradictions=result.contradictions
                )
            except Exception as exc:  # noqa: BLE001 - the report still carries them
                await log.error(
                    "Failed to store contradictions", reason=f"{type(exc).__name__}: {exc}"
                )
        for item in result.contradictions:
            await log.emit(
                "contradiction_found",
                contradiction_id=item.id,
                claim_ids=item.claim_ids,
                likely_cause=item.likely_cause,
                resolved=item.resolved,
                explanation=item.explanation[:300],
            )
        await log.emit(
            "agent_finished",
            iteration=iteration,
            candidates=len(result.candidates),
            topics=result.groups,
            failed_topics=result.failed_groups,
            contradictions=len(result.contradictions),
            unresolved=sum(1 for c in result.contradictions if not c.resolved),
        )
        return {
            "contradictions": [c.model_dump() for c in [*existing, *result.contradictions]],
            "events": log.events,
            "llm_calls": llm_calls,
        }

    return contradiction_detector_node


# ---------------------------------------------------------------------------
# Candidates and topics
# ---------------------------------------------------------------------------


def find_candidates(
    claims: Sequence[ClaimItem],
    vectors: Mapping[str, list[float]],
    *,
    new_ids: set[str],
    min_similarity: float,
    max_pairs: int,
    known: Sequence[set[str]] = (),
) -> list[CandidatePair]:
    """
    Claim pairs worth comparing (see the module docstring). *vectors* maps
    claim IDs to embeddings; a claim without one can still be linked.
    """
    stated_by: dict[str, list[str]] = {}
    for claim in claims:
        for passage_id in claim.passage_ids:
            stated_by.setdefault(passage_id, []).append(claim.id)

    linked: set[frozenset[str]] = set()
    for claim in claims:
        for link in claim.evidence:
            if link.stance != "contradicts":
                continue
            for other in stated_by.get(link.passage_id, []):
                if other != claim.id:
                    linked.add(frozenset((claim.id, other)))

    def eligible(a: ClaimItem, b: ClaimItem) -> bool:
        if a.id not in new_ids and b.id not in new_ids:
            return False
        if set(a.passage_ids) & set(b.passage_ids):
            return False
        pair = {a.id, b.id}
        return not any(pair <= group for group in known)

    unit = {cid: _unit(vector) for cid, vector in vectors.items()}
    pairs: list[CandidatePair] = []
    ordered = list(claims)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if not eligible(a, b):
                continue
            similarity = _dot(unit.get(a.id), unit.get(b.id))
            is_linked = frozenset((a.id, b.id)) in linked
            if is_linked or similarity >= min_similarity:
                pairs.append(CandidatePair(a=a.id, b=b.id, similarity=similarity, linked=is_linked))

    pairs.sort(key=lambda p: (not p.linked, -p.similarity))
    return pairs[: max(0, max_pairs)]


def group_pairs(pairs: Sequence[CandidatePair], *, max_size: int) -> list[list[str]]:
    """Join candidate pairs into topics of at most *max_size* claims (in pair order)."""
    size_cap = max(2, max_size)
    parent: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    extras: list[list[str]] = []

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        for claim_id in (pair.a, pair.b):
            if claim_id not in parent:
                parent[claim_id] = claim_id
                members[claim_id] = [claim_id]
        ra, rb = find(pair.a), find(pair.b)
        if ra == rb:
            continue
        if len(members[ra]) + len(members[rb]) <= size_cap:
            parent[rb] = ra
            members[ra].extend(members.pop(rb))
        else:
            extras.append([pair.a, pair.b])

    groups = [m for root, m in members.items() if find(root) == root and len(m) >= 2]
    return groups + extras


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else []


def _dot(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity of two unit vectors (0 when either is missing)."""
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


async def detect_contradictions(
    runtime: AgentRuntime,
    *,
    run_id: str,
    question: str,
    claims: Sequence[ClaimItem],
    sources: Mapping[str, SourceInfo],
    log: EventLog,
    llm_calls: list[dict[str, Any]],
    passages: Mapping[str, str] | None = None,
    existing: Sequence[ContradictionItem] = (),
    iteration: int = 0,
) -> Detection:
    """
    Find contradictions among *claims* that involve at least one claim from
    *iteration*. *passages* (ID -> text) supplies excerpts for evidence that
    has none. Contradictions already in *existing* are not reported again.
    """
    config = runtime.config.contradictions
    usable = [c for c in claims if c.verdict in USABLE_VERDICTS]
    new_ids = {c.id for c in usable if c.iteration == iteration}
    detection = Detection()
    if not new_ids or len(usable) < 2:
        return detection

    vectors: dict[str, list[float]] = {}
    try:
        embedded = await runtime.embeddings.embed_documents([c.text for c in usable])
        vectors = {c.id: r.vector for c, r in zip(usable, embedded, strict=True) if r.ok}
    except Exception as exc:  # noqa: BLE001 - linked pairs are still compared
        await log.error("Claim embedding failed", reason=f"{type(exc).__name__}: {exc}")

    known = [set(c.claim_ids) for c in existing]
    # O(claims^2 x dimensions) in pure Python: off the event loop.
    detection.candidates = await asyncio.to_thread(
        find_candidates,
        usable,
        vectors,
        new_ids=new_ids,
        min_similarity=config.min_similarity,
        max_pairs=config.max_pairs,
        known=known,
    )
    groups = group_pairs(detection.candidates, max_size=config.max_group_size)
    detection.groups = len(groups)
    if not groups:
        return detection

    by_id = {c.id: c for c in usable}
    size = max(1, config.groups_per_call)
    batches = [groups[i : i + size] for i in range(0, len(groups), size)]
    semaphore = asyncio.Semaphore(max(1, runtime.config.concurrency_cap))

    async def run(batch: list[list[str]]) -> list[ContradictionItem] | None:
        async with semaphore:
            return await _judge(
                runtime,
                run_id,
                question,
                batch,
                by_id,
                sources,
                passages or {},
                iteration,
                log,
                llm_calls,
            )

    seen = [set(group) for group in known]
    results = await asyncio.gather(*(run(batch) for batch in batches))
    for batch, found in zip(batches, results, strict=True):
        if found is None:
            detection.failed_groups += len(batch)
            continue
        for item in found:
            ids = set(item.claim_ids)
            # A pair already in a recorded contradiction is not reported again,
            # even inside a larger set.
            if any(len(ids & other) >= 2 for other in seen):
                continue
            seen.append(ids)
            detection.contradictions.append(item)
    return detection


async def _judge(
    runtime: AgentRuntime,
    run_id: str,
    question: str,
    batch: list[list[str]],
    by_id: Mapping[str, ClaimItem],
    sources: Mapping[str, SourceInfo],
    passages: Mapping[str, str],
    iteration: int,
    log: EventLog,
    llm_calls: list[dict[str, Any]],
) -> list[ContradictionItem] | None:
    """One model call over *batch* topics; ``None`` when it failed."""
    if runtime.budget.research_exhausted:
        return None
    label_of: dict[str, str] = {}
    for group in batch:
        for claim_id in group:
            label_of.setdefault(claim_id, f"C{len(label_of) + 1}")
    id_of = {label: claim_id for claim_id, label in label_of.items()}

    prompt = build_contradiction_prompt(
        question=question,
        groups=[
            PromptGroup(
                label=f"T{index}",
                claims=[
                    _prompt_claim(runtime, label_of[cid], by_id[cid], sources, passages)
                    for cid in group
                ],
            )
            for index, group in enumerate(batch, start=1)
        ],
    )
    try:
        async with asyncio.timeout(runtime.config.agent_timeout):
            response = await runtime.llm.generate_structured(
                prompt,
                ContradictionOutput,
                system=SYSTEM_PROMPT,
                agent=AGENT,
                run_id=run_id,
                max_output_tokens=runtime.config.contradictions.max_output_tokens,
            )
    except Exception as exc:  # noqa: BLE001 - lose these topics only
        await log.error(
            "Contradiction check failed",
            topics=len(batch),
            reason=f"{type(exc).__name__}: {exc}",
        )
        return None

    llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
    if response.parsed is None:
        return []
    groups = [set(group) for group in batch]
    scoring = runtime.config.scoring
    found: list[ContradictionItem] = []
    for detected in response.parsed.contradictions:
        ids: list[str] = []
        for raw in detected.claims:
            known = id_of.get(_normalize(raw) or "")
            if known is not None and known not in ids:
                ids.append(known)
        if len(ids) < 2 or not any(set(ids) <= group for group in groups):
            continue
        explanation = " ".join(detected.explanation.split())[:_MAX_EXPLANATION_CHARS]
        found.append(
            ContradictionItem(
                id=str(uuid.uuid4()),
                claim_ids=sorted(ids, key=lambda cid: int(label_of[cid][1:])),
                explanation=explanation,
                likely_cause=detected.likely_cause,
                resolved=scoring.is_resolved(detected.likely_cause),
                iteration=iteration,
            )
        )
    return found


def _prompt_claim(
    runtime: AgentRuntime,
    label: str,
    claim: ClaimItem,
    sources: Mapping[str, SourceInfo],
    passages: Mapping[str, str],
) -> PromptClaim:
    config = runtime.config
    excerpts: list[PromptExcerpt] = []
    for link in claim.evidence:
        if link.stance != "supports":
            continue
        if len(excerpts) >= max(1, config.contradictions.excerpts_per_claim):
            break
        text = link.excerpt or best_excerpt(
            passages.get(link.passage_id, ""), claim.text, max_chars=config.fact_check.excerpt_chars
        )
        if not text:
            continue
        info = sources.get(link.source_id)
        details = []
        if info is not None and info.domain_type != "unknown":
            details.append(info.domain_type.replace("_", " "))
        if info is not None and info.published_at:
            details.append(info.published_at[:10])
        excerpts.append(
            PromptExcerpt(
                title=info.title if info is not None else "",
                domain=domain_of(info.url) if info is not None else "",
                details=", ".join(details),
                text=text,
            )
        )
    return PromptClaim(label=label, text=claim.text, verdict=claim.verdict, excerpts=excerpts)


def _normalize(raw: str) -> str | None:
    match = _LABEL.match(raw.strip().strip("[]()").strip())
    return f"C{int(match.group(1))}" if match else None
