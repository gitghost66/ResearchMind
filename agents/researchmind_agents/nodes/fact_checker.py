"""
Fact checker node: claims -> verdicts, stances, and confidence; claims stored.

1. **Collect evidence** per claim: the passages it was extracted from, then
   the passages most similar to it elsewhere in the run (one batched
   embedding call), preferring sources not already represented — that is
   where independent corroboration or contradiction comes from. At most
   ``evidence_per_claim`` passages, but never fewer than the claim's own.
2. **Excerpt** each passage around the claim (``tools/excerpts.py``), so a
   check reads a few sentences, not a whole chunk.
3. **Judge** in batches of ``batch_size`` claims, concurrently under the
   concurrency cap. The model gives a stance per excerpt and a verdict per
   claim; claims and excerpts are labels (``C1``, ``E3``), never IDs.
4. **Score** deterministically (``reliability.py``): the verdict is made to
   agree with the stances, and confidence comes from the verdict, the
   supporting and contradicting sources' reliability, and how many
   independent sites agree. The model never produces a number.
5. **Store** the claims with their evidence links and emit ``claim_verified``.

A claim the model did not judge (failed batch, missing from the output, spent
budget) stays ``unverified`` with confidence 0, linked to its own passages.
The core, :func:`check_claims`, needs no graph state, so evals can call it.

Each round checks and stores only its own claims (``ClaimItem.iteration``);
earlier rounds' claims keep their verdicts.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from researchmind_agents.nodes.common import EventLog, StateNode, llm_call_record
from researchmind_agents.prompts.fact_checker import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    PromptClaim,
    PromptEvidence,
    build_fact_check_prompt,
)
from researchmind_agents.reliability import (
    EvidenceSignal,
    SourceSignals,
    claim_confidence,
    classify_domain,
    reconcile_verdict,
    source_reliability,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import ClaimItem, EvidenceItem, FactCheckOutput, ResearchState
from researchmind_agents.tools.excerpts import best_excerpt
from researchmind_agents.tools.urls import domain_of

__all__ = [
    "AGENT",
    "CandidatePassage",
    "SourceInfo",
    "make_fact_checker_node",
    "collect_evidence",
    "check_claims",
    "sources_from_state",
]

AGENT = "fact_checker"

_LABEL = re.compile(r"^([CE])(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceInfo:
    """What scoring and the prompt need to know about a source."""

    source_id: str
    url: str
    title: str
    domain_type: str
    reliability: float
    published_at: str | None = None

    @property
    def site(self) -> str:
        return domain_of(self.url)


@dataclass(frozen=True)
class CandidatePassage:
    """A passage a claim may be checked against."""

    passage_id: str
    source_id: str
    text: str
    # From retrieval metadata; used when the source is not in ``sources``.
    url: str = ""
    title: str = ""


def make_fact_checker_node(runtime: AgentRuntime) -> StateNode:
    async def fact_checker_node(state: ResearchState) -> dict[str, Any]:
        run_id = state["run_id"]
        iteration = int(state.get("iteration", 0))
        log = EventLog(runtime, run_id=run_id, agent=AGENT)
        await log.emit("agent_started", iteration=iteration)

        everything = [ClaimItem.model_validate(c) for c in state.get("claims", [])]
        claims = [c for c in everything if c.iteration == iteration]
        if not claims:
            await log.emit("agent_finished", iteration=iteration, claims=0)
            return {"events": log.events, "llm_calls": []}

        sources = sources_from_state(state)
        candidates = await collect_evidence(runtime, state, claims, log)
        llm_calls: list[dict[str, Any]] = []
        if runtime.budget.research_exhausted:
            await log.emit("fact_check_skipped", reason="research budget exhausted")
            verified = [_unverified(claim) for claim in claims]
        else:
            verified = await check_claims(
                runtime,
                run_id=run_id,
                question=state["question"],
                claims=claims,
                candidates=candidates,
                sources=sources,
                log=log,
                llm_calls=llm_calls,
            )

        try:
            await runtime.store.save_claims(run_id=run_id, claims=verified)
        except Exception as exc:  # noqa: BLE001 - the report still carries them
            await log.error("Failed to store claims", reason=f"{type(exc).__name__}: {exc}")

        by_verdict: dict[str, int] = {}
        for claim in verified:
            by_verdict[claim.verdict] = by_verdict.get(claim.verdict, 0) + 1
            await log.emit(
                "claim_verified",
                claim_id=claim.id,
                sub_question_id=claim.sub_question_id,
                verdict=claim.verdict,
                confidence=claim.confidence,
                text=claim.text[:300],
            )
        await log.emit(
            "agent_finished", iteration=iteration, claims=len(verified), by_verdict=by_verdict
        )
        by_id = {c.id: c for c in verified}
        return {
            "claims": [by_id.get(c.id, c).model_dump() for c in everything],
            "events": log.events,
            "llm_calls": llm_calls,
        }

    return fact_checker_node


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def sources_from_state(state: ResearchState) -> dict[str, SourceInfo]:
    return {
        s["id"]: SourceInfo(
            source_id=s["id"],
            url=s.get("url", ""),
            title=s.get("title", ""),
            domain_type=s.get("domain_type", "unknown"),
            reliability=float(s.get("reliability_score", 0.5)),
            published_at=s.get("published_at"),
        )
        for s in state.get("sources", [])
    }


async def collect_evidence(
    runtime: AgentRuntime,
    state: ResearchState,
    claims: list[ClaimItem],
    log: EventLog,
) -> dict[str, list[CandidatePassage]]:
    """Per claim: its own passages, then similar passages from other sources."""
    passages = {p["id"]: p for p in state.get("passages", [])}
    limit = max(1, runtime.config.fact_check.evidence_per_claim)

    hits_by_claim: list[list[CandidatePassage]] = [[] for _ in claims]
    if runtime.vector_search is not None:
        try:
            hits_by_claim = await _similar_passages(runtime, state["run_id"], claims, limit)
        except Exception as exc:  # noqa: BLE001 - check against own passages only
            await log.error("Evidence retrieval failed", reason=f"{type(exc).__name__}: {exc}")

    result: dict[str, list[CandidatePassage]] = {}
    for claim, hits in zip(claims, hits_by_claim, strict=True):
        own = []
        for passage_id in claim.passage_ids:
            row = passages.get(passage_id)
            hit = next((h for h in hits if h.passage_id == passage_id), None)
            text = row["text"] if row else (hit.text if hit else "")
            source_id = row["source_id"] if row else _source_of(claim, passage_id)
            if text and source_id:
                own.append(
                    CandidatePassage(
                        passage_id=passage_id,
                        source_id=source_id,
                        text=text,
                        url=hit.url if hit else "",
                        title=hit.title if hit else "",
                    )
                )
        chosen = list(own)
        seen_sources = {c.source_id for c in own}
        others = [h for h in hits if h.passage_id not in {c.passage_id for c in own}]
        # New sources first: they are what corroborates or contradicts.
        others.sort(key=lambda h: h.source_id in seen_sources)
        for hit in others:
            if len(chosen) >= max(limit, len(own)):
                break
            chosen.append(hit)
            seen_sources.add(hit.source_id)
        result[claim.id] = chosen
    return result


async def _similar_passages(
    runtime: AgentRuntime, run_id: str, claims: list[ClaimItem], limit: int
) -> list[list[CandidatePassage]]:
    assert runtime.vector_search is not None
    vector_search = runtime.vector_search
    vectors = await runtime.embeddings.embed_queries([c.text for c in claims])

    async def one(vector: list[float]) -> list[CandidatePassage]:
        if not vector:
            return []
        hits = await vector_search.search(
            vector,
            # Extra hits leave room to skip the claim's own passages.
            top_k=limit + 2,
            run_id=run_id,
            min_score=runtime.config.fact_check.min_similarity,
        )
        return [
            CandidatePassage(
                passage_id=hit.passage_id,
                source_id=hit.source_id,
                text=hit.text,
                url=str(hit.metadata.get("source_url") or ""),
                title=str(hit.metadata.get("source_title") or ""),
            )
            for hit in hits
        ]

    return list(
        await asyncio.gather(*(one(result.vector if result.ok else []) for result in vectors))
    )


def _source_of(claim: ClaimItem, passage_id: str) -> str:
    return next((e.source_id for e in claim.evidence if e.passage_id == passage_id), "")


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    claim: ClaimItem
    # (evidence label, passage, excerpt)
    evidence: list[tuple[str, CandidatePassage, str]]


async def check_claims(
    runtime: AgentRuntime,
    *,
    run_id: str,
    question: str,
    claims: Sequence[ClaimItem],
    candidates: Mapping[str, list[CandidatePassage]],
    sources: Mapping[str, SourceInfo],
    log: EventLog,
    llm_calls: list[dict[str, Any]],
) -> list[ClaimItem]:
    """Judge *claims* against their *candidates*; returns verified copies, in order."""
    config = runtime.config.fact_check
    size = max(1, config.batch_size)
    batches = [list(claims[i : i + size]) for i in range(0, len(claims), size)]
    semaphore = asyncio.Semaphore(max(1, runtime.config.concurrency_cap))
    resolved = _SourceResolver(runtime, sources)

    async def run(batch: list[ClaimItem]) -> list[ClaimItem]:
        async with semaphore:
            return await _check_batch(
                runtime, run_id, question, batch, candidates, resolved, log, llm_calls
            )

    checked = await asyncio.gather(*(run(batch) for batch in batches))
    return [claim for batch in checked for claim in batch]


async def _check_batch(
    runtime: AgentRuntime,
    run_id: str,
    question: str,
    batch: list[ClaimItem],
    candidates: Mapping[str, list[CandidatePassage]],
    resolved: _SourceResolver,
    log: EventLog,
    llm_calls: list[dict[str, Any]],
) -> list[ClaimItem]:
    excerpt_chars = runtime.config.fact_check.excerpt_chars
    entries: dict[str, _Entry] = {}
    counter = 0
    for index, claim in enumerate(batch, start=1):
        evidence = []
        for passage in candidates.get(claim.id, []):
            counter += 1
            excerpt = best_excerpt(passage.text, claim.text, max_chars=excerpt_chars)
            evidence.append((f"E{counter}", passage, excerpt))
        entries[f"C{index}"] = _Entry(claim=claim, evidence=evidence)

    checkable = {label: e for label, e in entries.items() if e.evidence}
    judged: dict[str, ClaimItem] = {}
    if checkable and not runtime.budget.research_exhausted:
        prompt = build_fact_check_prompt(
            question=question,
            claims=[
                PromptClaim(
                    label=label,
                    text=entry.claim.text,
                    evidence=[
                        _prompt_evidence(elabel, passage, excerpt, resolved)
                        for elabel, passage, excerpt in entry.evidence
                    ],
                )
                for label, entry in checkable.items()
            ],
        )
        try:
            async with asyncio.timeout(runtime.config.agent_timeout):
                response = await runtime.llm.generate_structured(
                    prompt,
                    FactCheckOutput,
                    system=SYSTEM_PROMPT,
                    agent=AGENT,
                    run_id=run_id,
                    max_output_tokens=runtime.config.fact_check.max_output_tokens,
                )
            llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
            if response.parsed is not None:
                judged = _apply(response.parsed, checkable, resolved, runtime)
        except Exception as exc:  # noqa: BLE001 - this batch stays unverified
            await log.error(
                "Fact check failed; claims left unverified",
                claims=len(checkable),
                reason=f"{type(exc).__name__}: {exc}",
            )

    return [judged.get(label) or _unverified(entry.claim) for label, entry in entries.items()]


def _apply(
    output: FactCheckOutput,
    entries: Mapping[str, _Entry],
    resolved: _SourceResolver,
    runtime: AgentRuntime,
) -> dict[str, ClaimItem]:
    """Turn the model's checks into verified claims; unknown labels are ignored."""
    judged: dict[str, ClaimItem] = {}
    for check in output.checks:
        label = _normalize(check.claim, "C")
        if label is None or label in judged or label not in entries:
            continue
        entry = entries[label]
        own_labels = {elabel for elabel, _, _ in entry.evidence}
        stances: dict[str, str] = {}
        for judgement in check.evidence:
            elabel = _normalize(judgement.evidence, "E")
            if elabel in own_labels and elabel not in stances:
                stances[elabel] = judgement.stance
        if not stances:
            continue

        evidence = [
            EvidenceItem(
                passage_id=passage.passage_id,
                source_id=passage.source_id,
                stance=stances.get(elabel, "neutral"),  # type: ignore[arg-type]
                excerpt=excerpt,
            )
            for elabel, passage, excerpt in entry.evidence
        ]
        verdict = reconcile_verdict(check.verdict, [e.stance for e in evidence])
        signals = [
            EvidenceSignal(
                site=resolved.get(passage).site,
                reliability=resolved.get(passage).reliability,
                stance=stance,
            )
            for (_, passage, _), stance in zip(
                entry.evidence, [e.stance for e in evidence], strict=True
            )
        ]
        judged[label] = entry.claim.model_copy(
            update={
                "evidence": evidence,
                "verdict": verdict,
                "confidence": claim_confidence(verdict, signals, config=runtime.config.scoring),
                "reason": " ".join(check.reason.split())[:500],
            }
        )
    return judged


def _unverified(claim: ClaimItem) -> ClaimItem:
    """The claim as extracted: linked to its own passages, not judged."""
    own = set(claim.passage_ids)
    evidence = [e for e in claim.evidence if e.passage_id in own] or claim.evidence
    return claim.model_copy(
        update={"evidence": evidence, "verdict": "unverified", "confidence": 0.0}
    )


def _normalize(raw: str, prefix: str) -> str | None:
    match = _LABEL.match(raw.strip().strip("[]()").strip())
    if not match or match.group(1).upper() != prefix:
        return None
    return f"{prefix}{int(match.group(2))}"


def _prompt_evidence(
    label: str, passage: CandidatePassage, excerpt: str, resolved: _SourceResolver
) -> PromptEvidence:
    info = resolved.get(passage)
    details = [info.domain_type.replace("_", " ")] if info.domain_type != "unknown" else []
    if info.published_at:
        details.append(info.published_at[:4])
    return PromptEvidence(
        label=label,
        title=info.title,
        domain=info.site,
        details=", ".join(details),
        excerpt=excerpt,
    )


class _SourceResolver:
    """Source info by ID; sources missing from state are typed from their URL."""

    def __init__(self, runtime: AgentRuntime, sources: Mapping[str, SourceInfo]) -> None:
        self._runtime = runtime
        self._sources = dict(sources)

    def get(self, passage: CandidatePassage) -> SourceInfo:
        info = self._sources.get(passage.source_id)
        if info is None:
            domain_type = classify_domain(passage.url) if passage.url else "unknown"
            reliability = source_reliability(
                SourceSignals(domain_type=domain_type),
                now=self._now(),
                config=self._runtime.config.scoring,
            )
            info = SourceInfo(
                source_id=passage.source_id,
                url=passage.url,
                title=passage.title,
                domain_type=domain_type,
                reliability=reliability,
            )
            self._sources[passage.source_id] = info
        return info

    def _now(self) -> datetime:
        return self._runtime.now()
