"""
Academic researcher: one sub-question -> papers stored as scored sources.

Runs as a branch like the web researchers (and shares their storing code):

1. search the scholarly indexes (arXiv, Semantic Scholar) with every planned
   query; merge round-robin, dropping repeats and papers another branch has
   already claimed;
2. skip papers whose abstract is shorter than ``min_abstract_chars``;
3. if there are more papers than pages to read, ask the model which to read
   (labels only); on failure, or with the budget spent, keep index order;
4. store each paper as a source: its title and abstract are the text — the
   abstract is the authors' own summary, and the indexes return it without
   scraping a page. Venue, publication date, and citation count feed its
   reliability score (peer-reviewed with a venue, otherwise preprint).

If the indexes return nothing usable (no results, rate-limited, down), the
branch researches the web instead, preferring journals, archives, and
university sites.
"""

from __future__ import annotations

from typing import Any

from researchmind_agents.modes import ACADEMIC
from researchmind_agents.nodes.common import EventLog, TaskNode, llm_call_record
from researchmind_agents.nodes.web_researcher import (
    PageToStore,
    ResearcherProfile,
    ResearchOutcome,
    make_branch_node,
    research_web,
    source_slots_left,
    store_pages,
)
from researchmind_agents.prompts.web_researcher import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_paper_selection_prompt,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import SourceSelection, SubQuestionItem
from researchmind_agents.tools.scholarly import Paper
from researchmind_agents.tools.urls import normalize_url

__all__ = [
    "AGENT",
    "ACADEMIC_WEB_PROFILE",
    "make_academic_researcher_node",
    "merge_paper_lists",
    "paper_type",
]

AGENT = "academic_researcher"

# Used when the scholarly indexes return nothing usable.
ACADEMIC_WEB_PROFILE = ResearcherProfile(
    agent=AGENT,
    domain=ACADEMIC,
    preferred_types=("peer_reviewed", "preprint", "academic", "government", "standards_body"),
)

# Venues that mean "not peer reviewed (yet)".
_PREPRINT_VENUES = frozenset(
    {"arxiv", "arxiv.org", "corr", "biorxiv", "medrxiv", "ssrn", "research square", "preprints"}
)
_EXCERPT_CHARS = 300


def make_academic_researcher_node(runtime: AgentRuntime) -> TaskNode:
    return make_branch_node(runtime, agent=AGENT, steps=_research)


def paper_type(paper: Paper) -> str:
    """``peer_reviewed`` when the paper names a venue other than a preprint archive."""
    venue = " ".join(paper.venue.lower().split())
    if venue and venue not in _PREPRINT_VENUES and not venue.startswith("arxiv"):
        return "peer_reviewed"
    return "preprint"


def merge_paper_lists(queries: list[str], results: dict[str, list[Paper]]) -> list[Paper]:
    """Interleave each query's papers by rank and drop repeats (by paper key or URL)."""
    lists = [results.get(query, []) for query in queries]
    seen: set[str] = set()
    merged: list[Paper] = []
    for rank in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if rank >= len(lst):
                continue
            paper = lst[rank]
            keys = {paper.key, normalize_url(paper.url)} if paper.url else {paper.key}
            if not paper.url or keys & seen:
                continue
            seen |= keys
            merged.append(paper)
    return merged


async def _research(
    runtime: AgentRuntime,
    run_id: str,
    sub_question: SubQuestionItem,
    log: EventLog,
    outcome: ResearchOutcome,
) -> None:
    queries = sub_question.queries or [sub_question.text]
    results = await runtime.scholarly.search_multi(queries)
    found = merge_paper_lists(queries, results)
    min_chars = runtime.config.academic.min_abstract_chars
    papers = [p for p in found if len(" ".join(p.abstract.split())) >= min_chars]
    await log.emit(
        "search_completed",
        sub_question_id=sub_question.id,
        queries=len(queries),
        candidates=len(papers),
        papers_found=len(found),
        index="scholarly",
    )

    if not papers:
        await log.emit(
            "scholarly_fallback",
            sub_question_id=sub_question.id,
            reason="no papers with an abstract" if found else "no papers found",
        )
        await research_web(runtime, run_id, sub_question, log, outcome, ACADEMIC_WEB_PROFILE)
        return

    outcome.candidates = len(papers)
    ordered = await _choose(runtime, run_id, sub_question, papers, log, outcome)

    want = max(1, runtime.config.research.pages_per_sub_question)
    left = source_slots_left(runtime)
    if left is not None:
        want = min(want, max(1, left))
    picked: list[Paper] = []
    for paper in ordered:
        if len(picked) >= want:
            break
        key = normalize_url(paper.url)
        if key in runtime.claimed_urls:
            outcome.duplicates += 1
            continue
        runtime.claimed_urls.add(key)
        picked.append(paper)

    pages = [
        PageToStore(
            url=paper.url,
            title=paper.title,
            text=f"{paper.title}\n\n{' '.join(paper.abstract.split())}",
            domain_type=paper_type(paper),
            published_at=paper.published_at,
            citation_count=paper.citation_count,
        )
        for paper in picked
    ]
    await store_pages(runtime, run_id, sub_question, pages, log, outcome)


async def _choose(
    runtime: AgentRuntime,
    run_id: str,
    sub_question: SubQuestionItem,
    papers: list[Paper],
    log: EventLog,
    outcome: ResearchOutcome,
) -> list[Paper]:
    """The model's picks first, the rest in index order as backups."""
    want = max(1, runtime.config.research.pages_per_sub_question)
    if len(papers) <= want or runtime.budget.research_exhausted:
        return papers

    labeled = {f"R{i}": paper for i, paper in enumerate(papers, start=1)}
    prompt = build_paper_selection_prompt(
        sub_question=sub_question.text,
        papers=[
            (label, paper.title, _details(paper), _excerpt(paper.abstract))
            for label, paper in labeled.items()
        ],
        max_selected=want,
    )
    try:
        response = await runtime.llm.generate_structured(
            prompt, SourceSelection, system=SYSTEM_PROMPT, agent=AGENT, run_id=run_id
        )
    except Exception as exc:  # noqa: BLE001 - fall back to index order
        await log.error(
            "Paper selection failed; using index order",
            sub_question_id=sub_question.id,
            reason=f"{type(exc).__name__}: {exc}",
        )
        return papers

    outcome.llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
    chosen: list[Paper] = []
    for raw in response.parsed.selected if response.parsed else []:
        paper = labeled.get(raw.strip().strip("[]").upper())
        if paper is not None and paper not in chosen:
            chosen.append(paper)
    chosen = chosen[:want]
    if not chosen:
        return papers
    return chosen + [p for p in papers if p not in chosen]


def _details(paper: Paper) -> str:
    parts: list[Any] = [paper.venue or "preprint"]
    if paper.year:
        parts.append(paper.year)
    if paper.citation_count is not None:
        parts.append(f"{paper.citation_count} citations")
    return ", ".join(str(p) for p in parts)


def _excerpt(abstract: str) -> str:
    text = " ".join(abstract.split())
    if len(text) <= _EXCERPT_CHARS:
        return text
    cut = text[:_EXCERPT_CHARS]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut) + "..."
