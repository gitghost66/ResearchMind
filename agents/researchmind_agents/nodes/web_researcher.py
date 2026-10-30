"""
Web researchers: one sub-question -> stored, scored sources and embedded passages.

One pipeline serves three researchers, each a :class:`ResearcherProfile`:

* ``web_researcher`` — general web research;
* ``technical_researcher`` — documentation, specifications, benchmarks;
* ``market_researcher`` — pricing, adoption, vendor lock-in, talent.

A profile sets the agent name (events and usage are attributed to it), the
guidance the model gets when choosing what to read, and which source types
are read first when the model's picks run out. The academic researcher
(``academic_researcher.py``) reuses the storing half of this module.

Runs once per sub-question, in parallel branches (``Send`` fan-out). Steps:

1. search every planned query; merge results round-robin so each query
   contributes, dropping duplicate URLs, and type each result's site;
2. if there are more candidates than pages to read, ask the model which to
   read (labels only); on failure, or with the budget spent, keep search order
   (specialists move their preferred source types first);
3. fetch pages, replacing failures from the remaining candidates up to
   ``max_fetch_attempts``;
4. score each page's reliability (``reliability.py``) and store it as a
   source — the store de-duplicates by URL and content hash across branches,
   and a duplicate's passages are not stored twice;
5. chunk, embed in one batched call, and store the passages.

``max_sources_per_run`` caps stored sources across all branches. A branch that
finds the cap already reached stores nothing and is marked ``skipped``.

A source enters the state only after its passages are stored, so everything
downstream can cite it. A failure marks the sub-question ``failed`` and the run
continues (PRD s6.3).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from researchmind_agents.modes import MARKET, TECHNICAL, WEB
from researchmind_agents.nodes.common import EventLog, TaskNode, llm_call_record
from researchmind_agents.prompts.web_researcher import (
    PROMPT_VERSION,
    SELECTION_GUIDANCE,
    SYSTEM_PROMPT,
    build_selection_prompt,
)
from researchmind_agents.reliability import (
    PAPER_TYPES,
    SourceSignals,
    classify_domain,
    describes_methodology,
    source_reliability,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    PassageItem,
    SourceItem,
    SourceSelection,
    SubQuestionItem,
    WebResearchTask,
)
from researchmind_agents.tools.chunking import chunk_text, content_hash
from researchmind_agents.tools.page_extract import PageContent
from researchmind_agents.tools.search import SearchResult
from researchmind_agents.tools.urls import domain_of, is_http_url, normalize_url

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT",
    "Candidate",
    "PageToStore",
    "ResearchOutcome",
    "ResearcherProfile",
    "WEB_PROFILE",
    "TECHNICAL_PROFILE",
    "MARKET_PROFILE",
    "make_web_researcher_node",
    "make_technical_researcher_node",
    "make_market_researcher_node",
    "fetch_candidates",
    "merge_candidates",
    "prefer_types",
    "score_source",
    "source_slots_left",
    "store_pages",
]

AGENT = "web_researcher"

# Longest snippet shown to the model per search result.
_SNIPPET_CHARS = 300


@dataclass(frozen=True)
class ResearcherProfile:
    """What makes one researcher different from another."""

    agent: str
    domain: str
    # Source types read first when the model's picks run out; search order
    # otherwise. Empty keeps search order.
    preferred_types: tuple[str, ...] = ()

    @property
    def guidance(self) -> str:
        return SELECTION_GUIDANCE.get(self.domain, SELECTION_GUIDANCE[WEB])


WEB_PROFILE = ResearcherProfile(agent=AGENT, domain=WEB)
TECHNICAL_PROFILE = ResearcherProfile(
    agent="technical_researcher",
    domain=TECHNICAL,
    preferred_types=(
        "official_docs",
        "standards_body",
        "code_repository",
        "peer_reviewed",
        "preprint",
        "academic",
        "company_blog",
    ),
)
MARKET_PROFILE = ResearcherProfile(
    agent="market_researcher",
    domain=MARKET,
    preferred_types=("official_docs", "news", "government", "company_blog"),
)


@dataclass(frozen=True)
class Candidate:
    """A search result that may be read."""

    label: str
    url: str
    title: str
    snippet: str
    domain: str
    # The site's source type (reliability.classify_domain).
    kind: str = "unknown"


@dataclass(frozen=True)
class PageToStore:
    """Text to store as a source, from a fetched page or a paper's abstract."""

    url: str
    title: str
    text: str
    # None: classify from the URL.
    domain_type: str | None = None
    # ISO 8601 or "".
    published_at: str = ""
    citation_count: int | None = None


@dataclass
class ResearchOutcome:
    sources: list[SourceItem] = field(default_factory=list)
    passages: list[PassageItem] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    candidates: int = 0
    fetched: int = 0
    fetch_failures: int = 0
    duplicates: int = 0
    # The run-wide source cap stopped this branch from storing a page.
    capped: bool = False


# How a branch does its research, given a fresh outcome to fill in.
ResearchSteps = Callable[
    [AgentRuntime, str, SubQuestionItem, EventLog, ResearchOutcome], Awaitable[None]
]


def make_web_researcher_node(
    runtime: AgentRuntime, profile: ResearcherProfile = WEB_PROFILE
) -> TaskNode:
    async def steps(
        rt: AgentRuntime,
        run_id: str,
        sub_question: SubQuestionItem,
        log: EventLog,
        outcome: ResearchOutcome,
    ) -> None:
        await research_web(rt, run_id, sub_question, log, outcome, profile)

    return make_branch_node(runtime, agent=profile.agent, steps=steps)


def make_technical_researcher_node(runtime: AgentRuntime) -> TaskNode:
    return make_web_researcher_node(runtime, TECHNICAL_PROFILE)


def make_market_researcher_node(runtime: AgentRuntime) -> TaskNode:
    return make_web_researcher_node(runtime, MARKET_PROFILE)


def make_branch_node(runtime: AgentRuntime, *, agent: str, steps: ResearchSteps) -> TaskNode:
    """
    A research branch: status bookkeeping, budget check, timeout, and error
    containment around *steps*, plus the events and state update every
    researcher returns.
    """

    # "state" is the Send payload for this branch, not the full graph state.
    async def researcher_node(state: WebResearchTask) -> dict[str, Any]:
        run_id = state["run_id"]
        sub_question = SubQuestionItem.model_validate(state["sub_question"])
        log = EventLog(runtime, run_id=run_id, agent=agent)
        await log.emit(
            "agent_started", sub_question_id=sub_question.id, sub_question=sub_question.text
        )

        outcome = ResearchOutcome()
        error: str | None = None

        if runtime.budget.research_exhausted:
            status = "skipped"
            error = "research budget exhausted"
        else:
            await _set_status(runtime, run_id, sub_question.id, "researching")
            try:
                async with asyncio.timeout(runtime.config.agent_timeout):
                    await steps(runtime, run_id, sub_question, log, outcome)
                if not outcome.passages and not outcome.duplicates:
                    error = "run source limit reached" if outcome.capped else "no usable sources"
            except TimeoutError:
                error = f"timed out after {runtime.config.agent_timeout}s"
                await log.error(error, sub_question_id=sub_question.id)
            except Exception as exc:  # noqa: BLE001 - one sub-question must not end the run
                error = f"{type(exc).__name__}: {exc}"
                await log.error("Research failed", sub_question_id=sub_question.id, reason=error)
            # Pages another branch already stored still serve this sub-question:
            # retrieval searches the whole run.
            if outcome.passages or outcome.duplicates:
                status = "completed"
            elif error == "run source limit reached":
                status = "skipped"
            else:
                status = "failed"

        await _set_status(runtime, run_id, sub_question.id, status)
        result = {
            "sub_question_id": sub_question.id,
            "agent": agent,
            "status": status,
            "candidates": outcome.candidates,
            "fetched": outcome.fetched,
            "fetch_failures": outcome.fetch_failures,
            "duplicates": outcome.duplicates,
            "sources": len(outcome.sources),
            "passages": len(outcome.passages),
            "error": error,
        }
        await log.emit("agent_finished", **result)
        return {
            "sources": [s.model_dump() for s in outcome.sources],
            "passages": [p.model_dump() for p in outcome.passages],
            "research_results": [result],
            "events": log.events,
            "llm_calls": outcome.llm_calls,
        }

    return researcher_node


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


async def research_web(
    runtime: AgentRuntime,
    run_id: str,
    sub_question: SubQuestionItem,
    log: EventLog,
    outcome: ResearchOutcome,
    profile: ResearcherProfile = WEB_PROFILE,
) -> None:
    """Search, choose, fetch, and store pages for one sub-question."""
    queries = sub_question.queries or [sub_question.text]
    results = await runtime.search.search_multi(queries)
    candidates = merge_candidates(queries, results)
    outcome.candidates = len(candidates)
    await log.emit(
        "search_completed",
        sub_question_id=sub_question.id,
        queries=len(queries),
        candidates=len(candidates),
    )
    if not candidates:
        return

    if source_slots_left(runtime) == 0:
        await _note_cap(runtime, sub_question, log, outcome)
        return

    ordered = await _choose(runtime, run_id, sub_question, candidates, log, outcome, profile)
    fetched = await fetch_candidates(runtime, ordered, outcome)
    pages = [
        PageToStore(
            url=page.final_url or candidate.url,
            title=page.title or candidate.title,
            text=page.text,
            published_at=page.published_at,
        )
        for candidate, page in fetched
    ]
    await store_pages(runtime, run_id, sub_question, pages, log, outcome)


def source_slots_left(runtime: AgentRuntime) -> int | None:
    """Sources this run may still store, or ``None`` when there is no cap."""
    cap = runtime.config.research.max_sources_per_run
    if cap <= 0:
        return None
    return max(0, cap - runtime.source_slots)


def _reserve_source_slot(runtime: AgentRuntime) -> bool:
    """
    Take one slot under the run's source cap.

    Check and increment run with no ``await`` between them, so concurrent
    branches cannot both take the last slot.
    """
    left = source_slots_left(runtime)
    if left == 0:
        return False
    runtime.source_slots += 1
    return True


async def _note_cap(
    runtime: AgentRuntime, sub_question: SubQuestionItem, log: EventLog, outcome: ResearchOutcome
) -> None:
    if outcome.capped:
        return
    outcome.capped = True
    await log.emit(
        "source_limit_reached",
        sub_question_id=sub_question.id,
        limit=runtime.config.research.max_sources_per_run,
    )


def merge_candidates(
    queries: list[str],
    results: dict[str, list[SearchResult]],
) -> list[Candidate]:
    """
    Interleave each query's results by rank (round-robin) and drop repeats.

    Round-robin keeps one broad query from crowding out the others. Labels are
    assigned in the merged order: R1, R2, ...
    """
    lists = [results.get(query, []) for query in queries]
    seen: set[str] = set()
    merged: list[Candidate] = []
    for rank in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if rank >= len(lst):
                continue
            result = lst[rank]
            if not is_http_url(result.url):
                continue
            key = normalize_url(result.url)
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                Candidate(
                    label=f"R{len(merged) + 1}",
                    url=result.url.strip(),
                    title=" ".join(result.title.split()),
                    snippet=" ".join(result.snippet.split())[:_SNIPPET_CHARS],
                    domain=domain_of(result.url),
                    kind=classify_domain(result.url),
                )
            )
    return merged


def prefer_types(candidates: list[Candidate], preferred: tuple[str, ...]) -> list[Candidate]:
    """Candidates of the preferred types first (in that order), the rest after; stable."""
    if not preferred:
        return list(candidates)
    rank = {kind: index for index, kind in enumerate(preferred)}
    return sorted(candidates, key=lambda c: rank.get(c.kind, len(preferred)))


async def _choose(
    runtime: AgentRuntime,
    run_id: str,
    sub_question: SubQuestionItem,
    candidates: list[Candidate],
    log: EventLog,
    outcome: ResearchOutcome,
    profile: ResearcherProfile,
) -> list[Candidate]:
    """Order candidates for fetching: the model's picks first, the rest as backups."""
    want = max(1, runtime.config.research.pages_per_sub_question)
    backups = prefer_types(candidates, profile.preferred_types)
    if len(candidates) <= want or runtime.budget.research_exhausted:
        return backups

    prompt = build_selection_prompt(
        sub_question=sub_question.text,
        results=[(c.label, c.title, c.domain, c.snippet, c.kind) for c in candidates],
        max_selected=want,
        guidance=profile.guidance,
    )
    try:
        response = await runtime.llm.generate_structured(
            prompt,
            SourceSelection,
            system=SYSTEM_PROMPT,
            agent=profile.agent,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to search order
        await log.error(
            "Result selection failed; using search order",
            sub_question_id=sub_question.id,
            reason=f"{type(exc).__name__}: {exc}",
        )
        return backups

    outcome.llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
    by_label = {c.label: c for c in candidates}
    chosen: list[Candidate] = []
    for raw in response.parsed.selected if response.parsed else []:
        candidate = by_label.get(raw.strip().strip("[]").upper())
        if candidate is not None and candidate not in chosen:
            chosen.append(candidate)
    chosen = chosen[:want]
    if not chosen:
        return backups
    return chosen + [c for c in backups if c not in chosen]


async def fetch_candidates(
    runtime: AgentRuntime,
    ordered: list[Candidate],
    outcome: ResearchOutcome,
) -> list[tuple[Candidate, PageContent]]:
    """
    Fetch until enough pages have usable text or the attempt cap is hit.

    Also used by the single-agent RAG baseline (``evals/systems.py``).
    """
    config = runtime.config.research
    want = max(1, config.pages_per_sub_question)
    max_attempts = max(want, config.max_fetch_attempts)
    # No point reading more pages than the run may still store.
    left = source_slots_left(runtime)
    if left is not None:
        want = min(want, max(1, left))

    good: list[tuple[Candidate, PageContent]] = []
    index = 0
    while len(good) < want and index < len(ordered) and outcome.fetched < max_attempts:
        size = min(want - len(good), max_attempts - outcome.fetched)
        batch: list[Candidate] = []
        while index < len(ordered) and len(batch) < size:
            candidate = ordered[index]
            index += 1
            key = normalize_url(candidate.url)
            if key in runtime.claimed_urls:
                # Another branch is reading (or has read) this page.
                outcome.duplicates += 1
                continue
            runtime.claimed_urls.add(key)
            batch.append(candidate)
        if not batch:
            break
        pages = await runtime.page_extract.extract_many(
            [c.url for c in batch], concurrency=runtime.config.concurrency_cap
        )
        outcome.fetched += len(batch)
        for candidate, page in zip(batch, pages, strict=True):
            if page.ok and len(page.text) >= config.min_source_chars:
                good.append((candidate, page))
            else:
                outcome.fetch_failures += 1
    return good


# ---------------------------------------------------------------------------
# Storing (shared with the academic researcher)
# ---------------------------------------------------------------------------


def score_source(runtime: AgentRuntime, page: PageToStore) -> tuple[str, float, str | None]:
    """``(domain_type, reliability_score, published_at)`` for a page about to be stored."""
    domain_type = page.domain_type or classify_domain(page.url)
    published = _parse_iso(page.published_at)
    signals = SourceSignals(
        domain_type=domain_type,
        published_at=published,
        citation_count=page.citation_count,
        has_methodology=domain_type in PAPER_TYPES and describes_methodology(page.text),
    )
    score = source_reliability(signals, now=runtime.now(), config=runtime.config.scoring)
    return domain_type, score, published.isoformat() if published else None


async def store_pages(
    runtime: AgentRuntime,
    run_id: str,
    sub_question: SubQuestionItem,
    pages: list[PageToStore],
    log: EventLog,
    outcome: ResearchOutcome,
) -> None:
    """Store sources, then embed all new chunks in one call, then store passages."""
    config = runtime.config.research
    new_sources: list[tuple[SourceItem, list[str]]] = []

    for page in pages:
        url = normalize_url(page.url)
        domain_type, reliability, published_at = score_source(runtime, page)
        source = SourceItem(
            id=str(uuid.uuid4()),
            url=url,
            title=" ".join(page.title.split()),
            domain=domain_of(url),
            content_hash=content_hash(page.text),
            sub_question_id=sub_question.id,
            domain_type=domain_type,
            reliability_score=reliability,
            published_at=published_at,
        )
        if not _reserve_source_slot(runtime):
            await _note_cap(runtime, sub_question, log, outcome)
            break
        created = False
        try:
            source_id, created = await runtime.store.save_source(run_id=run_id, source=source)
        except Exception as exc:  # noqa: BLE001 - skip this source only
            await log.error("Failed to store source", url=url, reason=str(exc))
            continue
        finally:
            # Only a newly stored source keeps its slot. The finally also
            # covers a timeout or cancel while waiting on the store.
            if not created:
                runtime.source_slots -= 1
        if not created:
            known = {s.id for s in outcome.sources} | {s.id for s, _ in new_sources}
            reused = (
                []
                if source_id in known
                else await _earlier_passages(runtime, run_id, source_id, sub_question)
            )
            if reused:
                # Stored by this branch before the run was interrupted.
                outcome.sources.append(source.model_copy(update={"id": source_id}))
                outcome.passages.extend(reused)
            else:
                outcome.duplicates += 1
            continue

        chunks = chunk_text(
            page.text,
            chunk_tokens=config.chunk_size_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
            max_chunks=config.max_chunks_per_source,
        )
        if chunks:
            new_sources.append((source.model_copy(update={"id": source_id}), chunks))

    if not new_sources:
        return

    all_chunks = [chunk for _, chunks in new_sources for chunk in chunks]
    embedded = await runtime.embeddings.embed_documents(all_chunks)
    failed = sum(1 for result in embedded if not result.ok)
    if failed:
        await log.error(
            "Some passages could not be embedded",
            sub_question_id=sub_question.id,
            failed=failed,
            total=len(embedded),
        )

    offset = 0
    for source, chunks in new_sources:
        vectors = [r.vector if r.ok else None for r in embedded[offset : offset + len(chunks)]]
        offset += len(chunks)
        passages = [
            PassageItem(
                id=str(uuid.uuid4()),
                source_id=source.id,
                sub_question_id=sub_question.id,
                text=chunk,
                chunk_index=index,
            )
            for index, chunk in enumerate(chunks)
        ]
        try:
            await runtime.store.save_passages(
                run_id=run_id,
                passages=passages,
                embeddings=vectors,
                embedding_model=runtime.embeddings.model,
            )
        except Exception as exc:  # noqa: BLE001 - skip this source only
            await log.error("Failed to store passages", source_id=source.id, reason=str(exc))
            continue

        outcome.sources.append(source)
        outcome.passages.extend(passages)
        await log.emit(
            "source_found",
            sub_question_id=sub_question.id,
            source_id=source.id,
            url=source.url,
            title=source.title,
            domain_type=source.domain_type,
            reliability_score=source.reliability_score,
            passages=len(passages),
        )


async def _earlier_passages(
    runtime: AgentRuntime,
    run_id: str,
    source_id: str,
    sub_question: SubQuestionItem,
) -> list[PassageItem]:
    """
    Passages this sub-question already stored for an existing source.

    Empty for a true duplicate (another branch stored it). Non-empty when this
    branch ran before a resume: its sources are then stored but missing from
    the state, and are reused. (A page this call already stored under another
    URL is ruled out by the caller.)
    """
    try:
        return await runtime.store.list_passages(
            run_id=run_id, source_id=source_id, sub_question_id=sub_question.id
        )
    except Exception as exc:  # noqa: BLE001 - treat it as a duplicate
        logger.warning("Could not read passages of source %s: %s", source_id, exc)
        return []


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def _set_status(runtime: AgentRuntime, run_id: str, sub_question_id: str, status: str) -> None:
    try:
        await runtime.store.set_sub_question_status(
            run_id=run_id, sub_question_id=sub_question_id, status=status
        )
    except Exception as exc:  # noqa: BLE001 - status is bookkeeping
        logger.warning("Failed to set sub-question %s to %s: %s", sub_question_id, status, exc)
