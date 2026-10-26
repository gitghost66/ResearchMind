"""
Synthesizer node: stored passages -> grounded, cited report.

1. **Retrieve** the passages most similar to each sub-question (vector search,
   scoped to the run), merged round-robin up to ``max_passages``. If vector
   search is unavailable or finds nothing, fall back to the passages the
   researchers put in the state.
2. **Verify** every passage's source is in the store; anything else is dropped
   before the model sees it.
3. **Label** sources ``S1``, ``S2``, ... in retrieval order. The model only ever
   sees labels; this node maps them back to source IDs.
4. **Write** a :class:`ReportOutput`, check its citations, retry once with the
   problems listed, then strip unknown labels and drop uncited statements.
5. **Fall back** to an extractive report (quoted passages, no model) if the
   budget is spent or the model fails, so a run always ends with a report.

Markdown is rendered from the validated structure, never written by the model.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from researchmind_agents.nodes.common import EventLog, StateNode, llm_call_record
from researchmind_agents.prompts.synthesizer import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    PromptSource,
    build_citation_retry_prompt,
    build_synthesis_prompt,
)
from researchmind_agents.report import (
    check_citations,
    cited_labels,
    enforce_citations,
    normalize_output,
    render_markdown,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    CitationItem,
    ReportOutput,
    ReportSection,
    ReportStatement,
    ResearchReport,
    ResearchState,
    SubQuestionItem,
)
from researchmind_agents.tools.urls import domain_of

logger = logging.getLogger(__name__)

__all__ = ["AGENT", "make_synthesizer_node"]

AGENT = "synthesizer"

# Extractive fallback: passages quoted per sub-question, and quote length.
_EXTRACTIVE_PASSAGES_PER_SECTION = 3
_EXCERPT_CHARS = 320


@dataclass
class _Evidence:
    passage_id: str
    source_id: str
    url: str
    title: str
    text: str
    chunk_index: int
    sub_question_ids: set[str] = field(default_factory=set)


@dataclass
class _LabeledSource:
    label: str
    source_id: str
    url: str
    title: str
    passages: list[_Evidence] = field(default_factory=list)


@dataclass
class _Draft:
    output: ReportOutput | None = None
    dropped: int = 0
    failure: str | None = None
    llm_calls: list[dict[str, Any]] = field(default_factory=list)


def make_synthesizer_node(runtime: AgentRuntime) -> StateNode:
    async def synthesizer_node(state: ResearchState) -> dict[str, Any]:
        run_id = state["run_id"]
        question = state["question"]
        log = EventLog(runtime, run_id=run_id, agent=AGENT)
        await log.emit("agent_started")

        sub_questions = [SubQuestionItem.model_validate(sq) for sq in state.get("sub_questions", [])]
        evidence = await _gather_evidence(runtime, state, sub_questions, log)

        # Citations may only point at stored sources (M3 acceptance).
        stored = await runtime.store.list_source_ids(run_id=run_id)
        evidence = [ev for ev in evidence if ev.source_id in stored and ev.url]
        sources = _label_sources(evidence)
        covered = {sq_id for ev in evidence for sq_id in ev.sub_question_ids}

        draft = _Draft()
        if not sources:
            generation = "empty"
        elif runtime.budget.exceeded:
            generation = "extractive"
            draft.failure = "the run's token/cost budget was spent"
        else:
            draft = await _write(runtime, run_id, state, sub_questions, sources, covered, log)
            generation = "llm" if draft.output is not None else "extractive"

        if generation == "extractive":
            await log.emit("extractive_fallback", reason=draft.failure)

        report = _assemble(
            runtime=runtime,
            state=state,
            question=question,
            sub_questions=sub_questions,
            evidence=evidence,
            sources=sources,
            covered=covered,
            stored_count=len(stored),
            draft=draft,
            generation=generation,
        )
        markdown = render_markdown(report)
        report_id = await runtime.store.save_report(run_id=run_id, report=report, markdown=markdown)

        statements = len(report.executive_summary) + sum(len(s.statements) for s in report.sections)
        await log.emit(
            "report_ready",
            report_id=report_id,
            generation=generation,
            sources=len(report.sources),
            statements=statements,
            dropped_statements=report.dropped_statements,
        )
        await log.emit("agent_finished", generation=generation)
        return {
            "report": report.model_dump(mode="json"),
            "report_markdown": markdown,
            "events": log.events,
            "llm_calls": draft.llm_calls,
        }

    return synthesizer_node


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def _gather_evidence(
    runtime: AgentRuntime,
    state: ResearchState,
    sub_questions: list[SubQuestionItem],
    log: EventLog,
) -> list[_Evidence]:
    config = runtime.config.synthesizer
    per_question: list[list[_Evidence]] = []
    if runtime.vector_search is not None and sub_questions:
        try:
            per_question = await _retrieve(runtime, state["run_id"], sub_questions)
        except Exception as exc:  # noqa: BLE001 - fall back to state passages
            await log.error("Vector retrieval failed", reason=f"{type(exc).__name__}: {exc}")
            per_question = []

    if not any(per_question):
        per_question = _from_state(state, sub_questions, config.top_k_per_sub_question)
        if any(per_question):
            await log.emit("retrieval_fallback", reason="vector search returned no passages")

    return _merge(per_question, config.max_passages)


async def _retrieve(
    runtime: AgentRuntime,
    run_id: str,
    sub_questions: list[SubQuestionItem],
) -> list[list[_Evidence]]:
    assert runtime.vector_search is not None
    vector_search = runtime.vector_search
    config = runtime.config.synthesizer
    vectors = await runtime.embeddings.embed_queries([sq.text for sq in sub_questions])

    async def one(sub_question: SubQuestionItem, vector: list[float]) -> list[_Evidence]:
        if not vector:
            return []
        hits = await vector_search.search(
            vector,
            top_k=config.top_k_per_sub_question,
            run_id=run_id,
            min_score=config.min_similarity,
        )
        return [
            _Evidence(
                passage_id=hit.passage_id,
                source_id=hit.source_id,
                url=str(hit.metadata.get("source_url") or ""),
                title=str(hit.metadata.get("source_title") or ""),
                text=hit.text,
                chunk_index=int(hit.metadata.get("chunk_index") or 0),
                sub_question_ids={sub_question.id},
            )
            for hit in hits
        ]

    return list(
        await asyncio.gather(
            *(
                one(sq, result.vector if result.ok else [])
                for sq, result in zip(sub_questions, vectors, strict=True)
            )
        )
    )


def _from_state(
    state: ResearchState,
    sub_questions: list[SubQuestionItem],
    top_k: int,
) -> list[list[_Evidence]]:
    """Passages the researchers stored, in discovery order, when retrieval is unavailable."""
    sources = {s["id"]: s for s in state.get("sources", [])}
    passages = state.get("passages", [])
    per_question = []
    for sq in sub_questions:
        picked = [
            p for p in passages if p.get("sub_question_id") == sq.id and p["source_id"] in sources
        ][: max(0, top_k)]
        per_question.append(
            [
                _Evidence(
                    passage_id=p["id"],
                    source_id=p["source_id"],
                    url=sources[p["source_id"]]["url"],
                    title=sources[p["source_id"]].get("title", ""),
                    text=p["text"],
                    chunk_index=int(p.get("chunk_index", 0)),
                    sub_question_ids={sq.id},
                )
                for p in picked
            ]
        )
    return per_question


def _merge(per_question: list[list[_Evidence]], max_passages: int) -> list[_Evidence]:
    """Round-robin across sub-questions so each gets evidence, de-duplicated by passage."""
    merged: dict[str, _Evidence] = {}
    for rank in range(max((len(hits) for hits in per_question), default=0)):
        for hits in per_question:
            if rank >= len(hits):
                continue
            evidence = hits[rank]
            existing = merged.get(evidence.passage_id)
            if existing is not None:
                existing.sub_question_ids |= evidence.sub_question_ids
            elif len(merged) < max(0, max_passages):
                merged[evidence.passage_id] = evidence
    return list(merged.values())


def _label_sources(evidence: list[_Evidence]) -> dict[str, _LabeledSource]:
    """Label sources S1, S2, ... in order of first appearance."""
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


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


async def _write(
    runtime: AgentRuntime,
    run_id: str,
    state: ResearchState,
    sub_questions: list[SubQuestionItem],
    sources: dict[str, _LabeledSource],
    covered: set[str],
    log: EventLog,
) -> _Draft:
    draft = _Draft()
    allowed = set(sources)
    objective = str(state.get("plan", {}).get("objective") or state["question"])
    prompt = build_synthesis_prompt(
        question=state["question"],
        objective=objective,
        sub_questions=[(sq.text, sq.id in covered) for sq in sub_questions],
        sources=[
            PromptSource(
                label=s.label,
                title=s.title,
                domain=domain_of(s.url),
                passages=[p.text for p in s.passages],
            )
            for s in sources.values()
        ],
    )

    try:
        async with asyncio.timeout(runtime.config.agent_timeout):
            output = await _generate(runtime, run_id, prompt, draft)
            check = check_citations(output, allowed)
            if not check.ok and not runtime.budget.exceeded:
                await log.emit(
                    "citation_retry",
                    unknown_labels=check.unknown_labels,
                    uncited_statements=len(check.uncited_statements),
                )
                retry_prompt = build_citation_retry_prompt(
                    prompt,
                    unknown_labels=check.unknown_labels,
                    uncited_statements=check.uncited_statements,
                    allowed_labels=sorted(allowed, key=lambda label: int(label[1:])),
                )
                try:
                    output = await _generate(runtime, run_id, retry_prompt, draft)
                except Exception as exc:  # noqa: BLE001 - keep the first draft
                    await log.error(
                        "Citation retry failed; keeping the first draft",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
    except Exception as exc:  # noqa: BLE001 - extractive fallback
        draft.failure = f"{type(exc).__name__}: {exc}"
        await log.error("Report generation failed", reason=draft.failure)
        return draft

    cleaned, dropped = enforce_citations(output, allowed)
    if dropped:
        await log.emit("citations_dropped", dropped_statements=dropped)
    if not cleaned.executive_summary and not cleaned.sections:
        draft.failure = "no statement in the model's report cited a valid source"
        draft.dropped = dropped
        return draft

    draft.output = cleaned
    draft.dropped = dropped
    return draft


async def _generate(runtime: AgentRuntime, run_id: str, prompt: str, draft: _Draft) -> ReportOutput:
    response = await runtime.llm.generate_structured(
        prompt,
        ReportOutput,
        system=SYSTEM_PROMPT,
        agent=AGENT,
        run_id=run_id,
        max_output_tokens=runtime.config.synthesizer.max_output_tokens,
    )
    draft.llm_calls.append(llm_call_record(response.usage, prompt_version=PROMPT_VERSION))
    if response.parsed is None:
        raise ValueError("model returned no parsed report")
    return normalize_output(response.parsed)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _assemble(
    *,
    runtime: AgentRuntime,
    state: ResearchState,
    question: str,
    sub_questions: list[SubQuestionItem],
    evidence: list[_Evidence],
    sources: dict[str, _LabeledSource],
    covered: set[str],
    stored_count: int,
    draft: _Draft,
    generation: str,
) -> ResearchReport:
    if draft.output is not None:
        title = draft.output.title or question
        summary = draft.output.executive_summary
        sections = draft.output.sections
        model_limitations = draft.output.limitations
    else:
        title = question
        summary = []
        sections = _extractive_sections(sub_questions, evidence, sources)
        model_limitations = []

    limitations = list(model_limitations)
    for sq in sub_questions:
        if sq.id not in covered:
            limitations.append(f"No usable sources were found for: {sq.text}")
    if runtime.budget.research_exhausted:
        limitations.append(
            "The run reached its token/cost budget, so research stopped early and some "
            "sub-questions may be under-covered."
        )
    if generation == "extractive":
        limitations.append(
            "The language model did not write this report "
            f"({draft.failure or 'unavailable'}); it quotes the most relevant passages instead."
        )
    limitations = list(dict.fromkeys(limitations))

    used = cited_labels(ReportOutput(title=title, executive_summary=summary, sections=sections, limitations=[]))
    citations = [
        CitationItem(
            label=label,
            source_id=sources[label].source_id,
            url=sources[label].url,
            title=sources[label].title,
            passage_ids=[p.passage_id for p in sources[label].passages],
        )
        for label in used
    ]

    return ResearchReport(
        title=title,
        question=question,
        mode=state.get("mode", "quick"),
        executive_summary=summary,
        sections=sections,
        limitations=limitations,
        methodology=_methodology(state, sub_questions, evidence, sources, generation),
        sources=citations,
        coverage={
            "sub_questions": len(sub_questions),
            "sub_questions_with_evidence": len(covered & {sq.id for sq in sub_questions}),
            "sources_stored": stored_count,
            "sources_shown": len(sources),
            "sources_cited": len(citations),
            "passages_shown": len(evidence),
            "tokens_used": runtime.budget.tokens_used,
            "cost_usd": runtime.budget.cost_usd,
            "budget_exhausted": runtime.budget.research_exhausted,
        },
        overall_confidence=None,
        generation=generation,  # type: ignore[arg-type]
        dropped_statements=draft.dropped,
        prompt_version=PROMPT_VERSION if generation == "llm" else "",
    )


def _extractive_sections(
    sub_questions: list[SubQuestionItem],
    evidence: list[_Evidence],
    sources: dict[str, _LabeledSource],
) -> list[ReportSection]:
    """Quote the top passages per sub-question, each cited to its source."""
    label_of = {s.source_id: label for label, s in sources.items()}
    sections = []
    for sq in sub_questions:
        picks = [ev for ev in evidence if sq.id in ev.sub_question_ids]
        statements = [
            ReportStatement(text=f'"{_excerpt(ev.text)}"', citations=[label_of[ev.source_id]])
            for ev in picks[:_EXTRACTIVE_PASSAGES_PER_SECTION]
            if ev.source_id in label_of
        ]
        if statements:
            sections.append(ReportSection(heading=sq.text, statements=statements))
    return sections


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end >= limit // 2:
        return cut[: end + 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut) + "..."


def _methodology(
    state: ResearchState,
    sub_questions: list[SubQuestionItem],
    evidence: list[_Evidence],
    sources: dict[str, _LabeledSource],
    generation: str,
) -> str:
    researched = len(state.get("sources", []))
    passages = len(state.get("passages", []))
    text = (
        f"Mode: {state.get('mode', 'quick')}. The question was split into "
        f"{len(sub_questions)} sub-question(s). Web search and page extraction stored "
        f"{researched} source(s) as {passages} passage(s). For each sub-question the "
        f"most similar passages were retrieved, {len(evidence)} in total from "
        f"{len(sources)} source(s). "
    )
    if generation == "llm":
        text += (
            "A language model wrote the report from those passages only; every statement "
            "had to cite a source label, and citations were checked against the stored "
            "sources. Confidence scores are not computed yet."
        )
    elif generation == "extractive":
        text += "No language model was used for the report; it quotes passages directly."
    else:
        text += "No usable passages were found, so the report has no findings."
    return text
