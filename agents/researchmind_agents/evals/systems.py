"""
The three systems the evaluation compares (PRD s11). All use the same model,
tools, and per-run budget; each run gets a fresh runtime.

* ``baseline_llm`` — **closed book**: one prompt, no retrieval. The model
  lists URLs of pages it says state each sentence; those pages are fetched
  (SSRF-guarded, robots-aware) so the judge can check them, and so made-up
  references are counted.
* ``rag`` — **single-agent RAG**: one search for the question, the top
  ``EVAL_RAG_PAGES`` pages read, chunked, and embedded with the researchers'
  own code, the ``EVAL_RAG_TOP_K`` passages most similar to the question
  retrieved, and one prompt that answers citing source labels. No planner,
  no specialists, no verification, no citation retry.
* ``researchmind`` — the full graph (``graph.run_research``) in the
  question's mode.

Each returns a :class:`SystemAnswer`: its sentences, each with citations
resolved to the evidence the system itself had for that source (the passages
shown to its writer; for the closed book, the fetched page), plus the sources
it retrieved. A citation is

* ``resolved`` — the source exists and its text is known;
* ``missing`` — it does not exist: an unknown label, something that is not a
  URL, a host that does not resolve, or HTTP 404/410;
* ``unreachable`` — it may exist, but its text could not be read (403,
  robots.txt, timeout, or over the fetch cap).
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from researchmind_agents.config import AgentConfig
from researchmind_agents.evals.benchmark import BenchmarkQuestion
from researchmind_agents.evals.schemas import ClosedBookAnswer, RagAnswer
from researchmind_agents.graph import run_research
from researchmind_agents.nodes.common import EventLog
from researchmind_agents.nodes.retrieval import Evidence, gather_evidence
from researchmind_agents.nodes.web_researcher import (
    PageToStore,
    ResearchOutcome,
    fetch_candidates,
    merge_candidates,
    store_pages,
)
from researchmind_agents.prompts.baselines import (
    PROMPT_VERSION as CLOSED_BOOK_PROMPT_VERSION,
)
from researchmind_agents.prompts.baselines import (
    RAG_PROMPT_VERSION,
    RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_closed_book_prompt,
    build_rag_prompt,
)
from researchmind_agents.prompts.synthesizer import PromptSource
from researchmind_agents.report import clean_statement
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    ReportStatement,
    ResearchReport,
    ResearchState,
    SubQuestionItem,
)
from researchmind_agents.tools.page_extract import PageContent
from researchmind_agents.tools.urls import domain_of, is_http_url, normalize_url

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_LLM",
    "RAG",
    "RESEARCHMIND",
    "SYSTEMS",
    "SYSTEM_NAMES",
    "CitationStatus",
    "Citation",
    "AnswerStatement",
    "RetrievedSource",
    "SystemAnswer",
    "SystemRunner",
    "SYSTEM_RUNNERS",
    "system_config",
    "classify_page",
    "report_statements",
    "rank_sources",
    "run_closed_book",
    "run_rag",
    "run_researchmind",
]

BASELINE_LLM = "baseline_llm"
RAG = "rag"
RESEARCHMIND = "researchmind"
SYSTEMS: tuple[str, ...] = (BASELINE_LLM, RAG, RESEARCHMIND)
SYSTEM_NAMES: dict[str, str] = {
    BASELINE_LLM: "Baseline LLM",
    RAG: "Single-agent RAG",
    RESEARCHMIND: "ResearchMind",
}

CitationStatus = Literal["resolved", "missing", "unreachable"]

_NOT_FOUND = re.compile(r"error '(?:404|410)\b", re.IGNORECASE)
_NO_HOST = ("dns resolution failed", "dns returned no addresses")


@dataclass
class Citation:
    """One source an answer sentence cites, resolved to what the system knew of it."""

    # As the system wrote it: a source label (S3) or a URL.
    label: str
    url: str = ""
    title: str = ""
    status: CitationStatus = "resolved"
    reason: str = ""
    # The source's text the system had: passages shown to its writer, or the
    # fetched page. Kept in memory for the judge; never written out whole.
    evidence: list[str] = field(default_factory=list, repr=False)


@dataclass
class AnswerStatement:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedSource:
    source_id: str
    url: str


@dataclass
class SystemAnswer:
    system: str
    statements: list[AnswerStatement] = field(default_factory=list)
    # Sources the system stored, in discovery order (none for the closed book).
    sources: list[RetrievedSource] = field(default_factory=list)
    # False for a system that never retrieves: retrieval metrics do not apply.
    retrieves: bool = True
    # Research rounds (ResearchMind's loop); 1 for the baselines.
    iterations: int = 1
    # "llm", or ResearchMind's "extractive"/"empty" fallbacks.
    generation: str = "llm"
    prompt_version: str = ""
    notes: list[str] = field(default_factory=list)


SystemRunner = Callable[[AgentRuntime, BenchmarkQuestion, str], Awaitable[SystemAnswer]]


def system_config(config: AgentConfig, system: str) -> AgentConfig:
    """The config a system's runtime is built with: the RAG baseline reads ``rag_pages`` pages."""
    if system != RAG:
        return config
    pages = max(1, config.eval.rag_pages)
    research = replace(
        config.research,
        pages_per_sub_question=pages,
        max_fetch_attempts=max(config.research.max_fetch_attempts, 2 * pages),
    )
    return replace(config, research=research)


# ---------------------------------------------------------------------------
# Closed book
# ---------------------------------------------------------------------------


def classify_page(page: PageContent) -> tuple[CitationStatus, str]:
    """Did a cited URL resolve to readable text, not exist, or fail to load?"""
    if page.ok and page.text.strip():
        return "resolved", ""
    error = page.error or "no readable text"
    lowered = error.lower()
    if _NOT_FOUND.search(error) or any(marker in lowered for marker in _NO_HOST):
        return "missing", error.splitlines()[0]
    return "unreachable", error.splitlines()[0]


async def run_closed_book(
    runtime: AgentRuntime, question: BenchmarkQuestion, run_id: str
) -> SystemAnswer:
    prompt = build_closed_book_prompt(
        question=question.question, user_context=question.user_context or None
    )
    response = await runtime.llm.generate_structured(
        prompt,
        ClosedBookAnswer,
        system=SYSTEM_PROMPT,
        agent=BASELINE_LLM,
        run_id=run_id,
        max_output_tokens=runtime.config.eval.max_output_tokens,
    )
    if response.parsed is None:
        raise ValueError("model returned no parsed answer")

    written = [
        (" ".join(s.text.split()), list(dict.fromkeys(u.strip() for u in s.sources if u.strip())))
        for s in response.parsed.statements
    ]
    written = [(text, urls) for text, urls in written if text]

    # Fetch each distinct URL once, up to the cap.
    to_check: dict[str, str] = {}
    for _, urls in written:
        for url in urls:
            if is_http_url(url):
                to_check.setdefault(normalize_url(url), url)
    keys = list(to_check)[: max(0, runtime.config.eval.baseline_max_urls)]
    pages = await runtime.page_extract.extract_many(
        [to_check[key] for key in keys], concurrency=runtime.config.concurrency_cap
    )
    fetched = dict(zip(keys, pages, strict=True))

    statements = []
    for text, urls in written:
        citations = []
        for url in urls:
            if not is_http_url(url):
                citations.append(Citation(label=url, status="missing", reason="not a URL"))
                continue
            page = fetched.get(normalize_url(url))
            if page is None:
                citations.append(
                    Citation(
                        label=url,
                        url=url,
                        status="unreachable",
                        reason="not fetched: over EVAL_BASELINE_MAX_URLS",
                    )
                )
                continue
            status, reason = classify_page(page)
            citations.append(
                Citation(
                    label=url,
                    url=url,
                    title=page.title,
                    status=status,
                    reason=reason,
                    evidence=[page.text] if status == "resolved" else [],
                )
            )
        statements.append(AnswerStatement(text=text, citations=citations))

    return SystemAnswer(
        system=BASELINE_LLM,
        statements=statements,
        retrieves=False,
        prompt_version=CLOSED_BOOK_PROMPT_VERSION,
    )


# ---------------------------------------------------------------------------
# Single-agent RAG
# ---------------------------------------------------------------------------


@dataclass
class _LabeledSource:
    label: str
    source_id: str
    url: str
    title: str
    passages: list[Evidence] = field(default_factory=list)


def _label_sources(evidence: Sequence[Evidence]) -> dict[str, _LabeledSource]:
    """S1, S2, ... in order of first appearance; passages in page order."""
    by_source: dict[str, _LabeledSource] = {}
    for item in evidence:
        source = by_source.get(item.source_id)
        if source is None:
            source = _LabeledSource(
                label=f"S{len(by_source) + 1}",
                source_id=item.source_id,
                url=item.url,
                title=item.title,
            )
            by_source[item.source_id] = source
        source.passages.append(item)
    for source in by_source.values():
        source.passages.sort(key=lambda p: p.chunk_index)
    return {source.label: source for source in by_source.values()}


async def run_rag(runtime: AgentRuntime, question: BenchmarkQuestion, run_id: str) -> SystemAnswer:
    log = EventLog(runtime, run_id=run_id, agent=RAG)
    text = question.question
    sub_question = SubQuestionItem(id=str(uuid.uuid4()), text=text, queries=[text])

    # Retrieve: one query, pages in search order (no model picks them).
    results = await runtime.search.search_multi([text])
    candidates = merge_candidates([text], results)
    outcome = ResearchOutcome(candidates=len(candidates))
    fetched = await fetch_candidates(runtime, candidates, outcome) if candidates else []
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
    sources = [RetrievedSource(s.id, s.url) for s in outcome.sources]
    if not outcome.passages:
        return SystemAnswer(
            system=RAG,
            sources=sources,
            prompt_version=RAG_PROMPT_VERSION,
            notes=[f"no usable pages ({outcome.candidates} search results)"],
        )

    state: ResearchState = {
        "run_id": run_id,
        "sources": [s.model_dump() for s in outcome.sources],
        "passages": [p.model_dump() for p in outcome.passages],
    }
    top_k = max(1, runtime.config.eval.rag_top_k)
    evidence = await gather_evidence(
        runtime,
        state,
        [sub_question],
        log,
        top_k=top_k,
        max_passages=top_k,
        min_similarity=runtime.config.synthesizer.min_similarity,
    )
    labeled = _label_sources(evidence)

    # Generate: one call, no citation retry.
    prompt = build_rag_prompt(
        question=text,
        sources=[
            PromptSource(
                label=s.label,
                title=s.title,
                domain=domain_of(s.url),
                passages=[p.text for p in s.passages],
            )
            for s in labeled.values()
        ],
        user_context=question.user_context or None,
    )
    response = await runtime.llm.generate_structured(
        prompt,
        RagAnswer,
        system=RAG_SYSTEM_PROMPT,
        agent=RAG,
        run_id=run_id,
        max_output_tokens=runtime.config.eval.max_output_tokens,
    )
    if response.parsed is None:
        raise ValueError("model returned no parsed answer")

    statements = []
    for raw in response.parsed.statements:
        cleaned = clean_statement(ReportStatement(text=raw.text, citations=raw.citations), "S")
        if not cleaned.text:
            continue
        citations = []
        for label in cleaned.citations:
            source = labeled.get(label)
            if source is None:
                citations.append(Citation(label=label, status="missing", reason="unknown label"))
            else:
                citations.append(
                    Citation(
                        label=label,
                        url=source.url,
                        title=source.title,
                        evidence=[p.text for p in source.passages],
                    )
                )
        statements.append(AnswerStatement(text=" ".join(cleaned.text.split()), citations=citations))

    return SystemAnswer(
        system=RAG,
        statements=statements,
        sources=sources,
        prompt_version=RAG_PROMPT_VERSION,
    )


# ---------------------------------------------------------------------------
# ResearchMind
# ---------------------------------------------------------------------------


def report_statements(report: ResearchReport, passages: dict[str, str]) -> list[AnswerStatement]:
    """
    The report's sentences (summary, sections, recommendation) with their
    citations. A source's evidence is what the synthesizer relied on: in
    verified modes the passages behind the sentence's own claims from that
    source; otherwise the source's passages shown to the model.
    """
    by_label = {item.label: item for item in report.sources}
    findings = {finding.label: finding for finding in report.key_findings}
    statements = []
    for statement in [
        *report.executive_summary,
        *(s for section in report.sections for s in section.statements),
        *report.recommendation,
    ]:
        text = " ".join(statement.text.split())
        if not text:
            continue
        citations = []
        for label in statement.citations:
            item = by_label.get(label)
            if item is None:
                citations.append(Citation(label=label, status="missing", reason="unknown label"))
                continue
            ids = [
                evidence.passage_id
                for claim in statement.claims
                if claim in findings
                for evidence in findings[claim].evidence
                if evidence.source_label == label and evidence.stance != "neutral"
            ] or list(item.passage_ids)
            texts = [passages[pid] for pid in dict.fromkeys(ids) if pid in passages]
            citations.append(
                Citation(
                    label=label,
                    url=item.url,
                    title=item.title,
                    status="resolved" if texts else "unreachable",
                    reason="" if texts else "cited passages not in the run's state",
                    evidence=texts,
                )
            )
        statements.append(AnswerStatement(text=text, citations=citations))
    return statements


async def run_researchmind(
    runtime: AgentRuntime, question: BenchmarkQuestion, run_id: str
) -> SystemAnswer:
    state = await run_research(
        runtime,
        run_id=run_id,
        question=question.question,
        mode=question.mode,
        user_context=question.user_context or None,
    )
    if not state.get("report"):
        raise ValueError("the run produced no report")
    report = ResearchReport.model_validate(state["report"])
    passages = {p["id"]: p["text"] for p in state.get("passages", [])}
    notes = [] if report.generation == "llm" else [f"report generation: {report.generation}"]
    return SystemAnswer(
        system=RESEARCHMIND,
        statements=report_statements(report, passages),
        sources=[RetrievedSource(s["id"], s["url"]) for s in state.get("sources", [])],
        iterations=int(report.coverage.get("iterations") or 1),
        generation=report.generation,
        prompt_version=report.prompt_version,
        notes=notes,
    )


SYSTEM_RUNNERS: dict[str, SystemRunner] = {
    BASELINE_LLM: run_closed_book,
    RAG: run_rag,
    RESEARCHMIND: run_researchmind,
}


# ---------------------------------------------------------------------------
# Retrieval ranking
# ---------------------------------------------------------------------------


async def rank_sources(
    runtime: AgentRuntime, *, run_id: str, question: str, sources: Sequence[RetrievedSource]
) -> list[str]:
    """
    URLs of the sources a system stored, ranked by their best passage's
    similarity to the question; sources with no embedded passage follow in
    discovery order.

    One ranking for every system, so Precision@K and Recall@K compare what
    each one retrieved rather than how its report happens to be ordered.
    """
    by_id: dict[str, str] = {}
    for source in sources:
        by_id.setdefault(source.source_id, source.url)
    order: list[str] = []
    if by_id and runtime.vector_search is not None:
        try:
            [result] = await runtime.embeddings.embed_queries([question])
            if result.ok:
                # Ask for enough passages to reach every source.
                hits = await runtime.vector_search.search(
                    result.vector, top_k=max(50, 20 * len(by_id)), run_id=run_id
                )
                for hit in hits:
                    if hit.source_id in by_id and hit.source_id not in order:
                        order.append(hit.source_id)
        except Exception as exc:  # noqa: BLE001 - fall back to discovery order
            logger.warning("Ranking sources for run %s failed: %s", run_id, exc)
    order += [sid for sid in by_id if sid not in order]
    return [by_id[sid] for sid in order]


def answer_summary(answer: SystemAnswer) -> dict[str, Any]:
    """Counts for logs and progress lines."""
    citations = [c for s in answer.statements for c in s.citations]
    return {
        "statements": len(answer.statements),
        "citations": len(citations),
        "missing": sum(1 for c in citations if c.status == "missing"),
        "sources": len(answer.sources),
    }
