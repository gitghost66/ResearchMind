"""
Synthesizer node: evidence -> grounded, cited report.

Two paths, chosen by the run's mode (``modes.py``):

**Verified** (deep, academic, decision) — the report is written from
fact-checked claims only (PRD s6.2):

1. keep claims whose verdict is usable (supported, weak, mixed), whose
   confidence clears ``min_claim_confidence``, and that still have a
   supporting passage from a stored source; claims in a contradiction first,
   then the rest round-robin across sub-questions, at most ``max_claims``;
2. label claims ``C1``, ``C2``, ... (grouped under their top-level
   sub-question; a gap-filling sub-question's claims under the one it fills),
   their supporting (and, for mixed claims, contradicting) sources ``S1``,
   ``S2``, ..., and contradictions between labeled claims ``X1``, ``X2``, ...;
3. the model writes statements that cite claim labels — in decision mode
   also a recommendation, which is required; unknown labels, uncited
   statements, and a missing recommendation get one retry, then are stripped
   or dropped;
4. each statement's source citations are derived from its claims, and the
   report's confidence is computed from the cited claims, lowered for
   uncovered sub-questions and unresolved contradictions
   (``reliability.overall_confidence``) — never by the model.

**Passages** (quick, or a verified run with no usable claim) — the M3 path:

1. retrieve the passages most similar to each sub-question (run-scoped),
   drop any whose source is not stored, label sources ``S1``, ``S2``, ...;
2. the model writes statements citing source labels; same check and retry.

Either path falls back to an extractive report (claims or passages quoted
directly, no model) when the budget is spent or the model fails, so a run
always ends with a report. Markdown is rendered from the validated structure,
never written by the model.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from researchmind_agents.coverage import report_roots, root_ids, roots, rounds_run
from researchmind_agents.modes import ModeProfile, mode_profile
from researchmind_agents.nodes.common import EventLog, StateNode, llm_call_record
from researchmind_agents.nodes.retrieval import Evidence, gather_evidence
from researchmind_agents.prompts.synthesizer import (
    DECISION_PROMPT_VERSION,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    VERIFIED_PROMPT_VERSION,
    VERIFIED_SYSTEM_PROMPT,
    PromptClaim,
    PromptContradiction,
    PromptSource,
    build_citation_retry_prompt,
    build_synthesis_prompt,
    build_verified_synthesis_prompt,
)
from researchmind_agents.reliability import SCORING_VERSION, overall_confidence
from researchmind_agents.report import (
    check_citations,
    cited_labels,
    enforce_citations,
    label_sort_key,
    normalize_output,
    render_markdown,
)
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import (
    USABLE_VERDICTS,
    CitationItem,
    ClaimItem,
    ContradictionItem,
    DecisionReportOutput,
    GapItem,
    ReportClaim,
    ReportContradiction,
    ReportEvidence,
    ReportOutput,
    ReportSection,
    ReportStatement,
    ResearchReport,
    ResearchState,
    SubQuestionItem,
    VerifiedReportOutput,
)
from researchmind_agents.tools.urls import domain_of

logger = logging.getLogger(__name__)

__all__ = ["AGENT", "make_synthesizer_node", "select_claims"]

AGENT = "synthesizer"

# Extractive fallback: items quoted per sub-question, and quote length.
_EXTRACTIVE_PASSAGES_PER_SECTION = 3
_EXTRACTIVE_CLAIMS_PER_SECTION = 6
_EXCERPT_CHARS = 320


@dataclass
class _LabeledSource:
    label: str
    source_id: str
    url: str
    title: str
    passages: list[Evidence] = field(default_factory=list)
    # Verified path: passages behind the claims that cite this source.
    passage_ids: list[str] = field(default_factory=list)


@dataclass
class _LabeledClaim:
    label: str
    claim: ClaimItem
    # Plan position of its sub-question (1-based), 0 if none.
    position: int
    # Source labels shown with the claim.
    citations: list[str] = field(default_factory=list)
    supporting_sites: int = 0


@dataclass
class _Draft:
    output: ReportOutput | None = None
    dropped: int = 0
    failure: str | None = None
    llm_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _Writer:
    """How one path asks the model for a report."""

    schema: type[BaseModel]
    system: str
    version: str
    # Label prefix the model cites: "S" (sources) or "C" (claims).
    prefix: str
    # Model output -> ReportOutput whose ``citations`` hold the cited labels.
    to_output: Callable[[Any], ReportOutput]
    # The report must include a recommendation (decision mode).
    recommendation: bool = False


def _identity(output: ReportOutput) -> ReportOutput:
    return output


def _from_findings(output: VerifiedReportOutput) -> ReportOutput:
    def statement(s: Any) -> ReportStatement:
        return ReportStatement(text=s.text, citations=list(s.claims))

    return ReportOutput(
        title=output.title,
        executive_summary=[statement(s) for s in output.executive_summary],
        sections=[
            ReportSection(heading=sec.heading, statements=[statement(s) for s in sec.statements])
            for sec in output.sections
        ],
        limitations=output.limitations,
    )


def _from_decision(output: DecisionReportOutput) -> ReportOutput:
    converted = _from_findings(
        VerifiedReportOutput(
            title=output.title,
            executive_summary=output.executive_summary,
            sections=output.sections,
            limitations=output.limitations,
        )
    )
    converted.recommendation = [
        ReportStatement(text=s.text, citations=list(s.claims)) for s in output.recommendation
    ]
    return converted


_PASSAGE_WRITER = _Writer(ReportOutput, SYSTEM_PROMPT, PROMPT_VERSION, "S", _identity)
_CLAIM_WRITER = _Writer(
    VerifiedReportOutput, VERIFIED_SYSTEM_PROMPT, VERIFIED_PROMPT_VERSION, "C", _from_findings
)
_DECISION_WRITER = _Writer(
    DecisionReportOutput,
    VERIFIED_SYSTEM_PROMPT,
    DECISION_PROMPT_VERSION,
    "C",
    _from_decision,
    recommendation=True,
)


def make_synthesizer_node(runtime: AgentRuntime) -> StateNode:
    async def synthesizer_node(state: ResearchState) -> dict[str, Any]:
        run_id = state["run_id"]
        log = EventLog(runtime, run_id=run_id, agent=AGENT)
        await log.emit("agent_started")

        profile = mode_profile(state.get("mode", "quick"))
        sub_questions = [SubQuestionItem.model_validate(sq) for sq in state.get("sub_questions", [])]
        # Citations may only point at stored sources (M3 acceptance).
        stored = await runtime.store.list_source_ids(run_id=run_id)

        report: ResearchReport | None = None
        llm_calls: list[dict[str, Any]] = []
        claims = [ClaimItem.model_validate(c) for c in state.get("claims", [])]
        contradictions = [
            ContradictionItem.model_validate(c) for c in state.get("contradictions", [])
        ]
        if profile.verify:
            usable = select_claims(
                claims,
                sub_questions,
                stored=stored & set(_state_sources(state)),
                max_claims=runtime.config.synthesizer.max_claims,
                min_confidence=runtime.config.synthesizer.min_claim_confidence,
                required={cid for c in contradictions for cid in c.claim_ids},
            )
            if usable:
                report, llm_calls = await _verified_report(
                    runtime,
                    state,
                    profile,
                    sub_questions,
                    claims,
                    usable,
                    contradictions,
                    len(stored),
                    log,
                )
            else:
                await log.emit("verification_fallback", reason="no usable verified claims")

        if report is None:
            report, llm_calls = await _passage_report(
                runtime, state, sub_questions, stored, claims, profile.verify, log
            )

        markdown = render_markdown(report)
        report_id = await runtime.store.save_report(run_id=run_id, report=report, markdown=markdown)

        statements = len(report.executive_summary) + sum(len(s.statements) for s in report.sections)
        await log.emit(
            "report_ready",
            report_id=report_id,
            generation=report.generation,
            sources=len(report.sources),
            statements=statements,
            dropped_statements=report.dropped_statements,
            key_findings=len(report.key_findings),
            contradictions=len(report.contradictory_evidence),
            recommendation=len(report.recommendation),
            overall_confidence=report.overall_confidence,
        )
        await log.emit("agent_finished", generation=report.generation)
        return {
            "report": report.model_dump(mode="json"),
            "report_markdown": markdown,
            "events": log.events,
            "llm_calls": llm_calls,
        }

    return synthesizer_node


# ---------------------------------------------------------------------------
# Writing (both paths)
# ---------------------------------------------------------------------------


async def _write(
    runtime: AgentRuntime,
    run_id: str,
    prompt: str,
    allowed: set[str],
    writer: _Writer,
    log: EventLog,
) -> _Draft:
    draft = _Draft()
    output: ReportOutput | None = None
    try:
        async with asyncio.timeout(runtime.config.agent_timeout):
            output = await _generate(runtime, run_id, prompt, writer, draft)
            check = check_citations(output, allowed)
            missing = writer.recommendation and not _has_cited(output.recommendation, allowed)
            if (not check.ok or missing) and not runtime.budget.exceeded:
                await log.emit(
                    "citation_retry",
                    unknown_labels=check.unknown_labels,
                    uncited_statements=len(check.uncited_statements),
                    missing_recommendation=missing,
                )
                retry_prompt = build_citation_retry_prompt(
                    prompt,
                    unknown_labels=check.unknown_labels,
                    uncited_statements=check.uncited_statements,
                    allowed_labels=sorted(allowed, key=label_sort_key),
                    missing_recommendation=missing,
                )
                try:
                    retried = await _generate(runtime, run_id, retry_prompt, writer, draft)
                    output = _keep_better(output, retried, allowed, writer)
                except Exception as exc:  # noqa: BLE001 - keep the first draft
                    await log.error(
                        "Citation retry failed; keeping the first draft",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
    except Exception as exc:  # noqa: BLE001 - extractive fallback
        reason = f"{type(exc).__name__}: {exc}"
        if output is None:
            draft.failure = reason
            await log.error("Report generation failed", reason=reason)
            return draft
        # The timeout cancelled the citation retry, which the inner handler
        # cannot catch; the first draft is still usable.
        await log.error("Citation retry failed; keeping the first draft", reason=reason)

    cleaned, dropped = enforce_citations(output, allowed)
    if dropped:
        await log.emit("citations_dropped", dropped_statements=dropped)
    draft.dropped = dropped
    if not cleaned.executive_summary and not cleaned.sections:
        draft.failure = "no statement in the model's report cited a valid label"
        return draft
    draft.output = cleaned
    return draft


def _has_cited(statements: list[ReportStatement], allowed: set[str]) -> bool:
    return any(s.text and any(label in allowed for label in s.citations) for s in statements)


def _keep_better(
    first: ReportOutput, retry: ReportOutput, allowed: set[str], writer: _Writer
) -> ReportOutput:
    """
    The retry, unless it lost what the first draft had: with no validly cited
    statement left it falls back to the first draft, and a recommendation the
    retry dropped is taken from the first draft.
    """
    cleaned, _ = enforce_citations(retry, allowed)
    if not cleaned.executive_summary and not cleaned.sections:
        return first
    if (
        writer.recommendation
        and not _has_cited(retry.recommendation, allowed)
        and _has_cited(first.recommendation, allowed)
    ):
        return retry.model_copy(update={"recommendation": first.recommendation})
    return retry


async def _generate(
    runtime: AgentRuntime, run_id: str, prompt: str, writer: _Writer, draft: _Draft
) -> ReportOutput:
    response = await runtime.llm.generate_structured(
        prompt,
        writer.schema,
        system=writer.system,
        agent=AGENT,
        run_id=run_id,
        max_output_tokens=runtime.config.synthesizer.max_output_tokens,
    )
    draft.llm_calls.append(llm_call_record(response.usage, prompt_version=writer.version))
    if response.parsed is None:
        raise ValueError("model returned no parsed report")
    return normalize_output(writer.to_output(response.parsed), prefix=writer.prefix)


# ---------------------------------------------------------------------------
# Verified path
# ---------------------------------------------------------------------------


def select_claims(
    claims: list[ClaimItem],
    sub_questions: list[SubQuestionItem],
    *,
    stored: set[str],
    max_claims: int,
    min_confidence: float = 0.0,
    required: set[str] | None = None,
) -> list[ClaimItem]:
    """
    Claims a report may rest on, in plan order and best first within each
    top-level sub-question; evidence from sources outside *stored* is
    removed, and a claim left with no supporting passage is dropped. Within
    *max_claims*, each sub-question's best claim comes first, then the claims
    in *required* (those in a contradiction, so both sides can be shown), then
    the rest round-robin across sub-questions.
    """
    root_of = root_ids(sub_questions)
    top_level = roots(sub_questions)
    position = {sq.id: i for i, sq in enumerate(top_level)}

    def slot_of(claim: ClaimItem) -> int:
        return position.get(root_of.get(claim.sub_question_id or "", ""), len(top_level))

    groups: dict[int, list[ClaimItem]] = {}
    for claim in claims:
        if claim.verdict not in USABLE_VERDICTS or claim.confidence < min_confidence:
            continue
        evidence = [e for e in claim.evidence if e.source_id in stored]
        if not any(e.stance == "supports" for e in evidence):
            continue
        groups.setdefault(slot_of(claim), []).append(claim.model_copy(update={"evidence": evidence}))

    limit = max(0, max_claims)
    ordered_groups = [sorted(g, key=lambda c: -c.confidence) for _, g in sorted(groups.items())]
    must = required or set()
    picked: list[ClaimItem] = []
    taken: set[str] = set()

    def take(claim: ClaimItem) -> None:
        if len(picked) < limit and claim.id not in taken:
            picked.append(claim)
            taken.add(claim.id)

    for group in ordered_groups:
        take(group[0])
    for group in ordered_groups:
        for claim in group:
            if claim.id in must:
                take(claim)
    for rank in range(max((len(g) for g in ordered_groups), default=0)):
        for group in ordered_groups:
            if rank < len(group) and len(picked) < limit and group[rank].id not in taken:
                picked.append(group[rank])
                taken.add(group[rank].id)
    return sorted(picked, key=lambda c: (slot_of(c), -c.confidence))


def _state_sources(state: ResearchState) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in state.get("sources", [])}


def _label_claims(
    claims: list[ClaimItem],
    sub_questions: list[SubQuestionItem],
    state: ResearchState,
    top_level: list[SubQuestionItem] | None = None,
) -> tuple[dict[str, _LabeledClaim], dict[str, _LabeledSource]]:
    """
    Claims C1.. in order, each placed under its position in *top_level* (the
    roots the report covers; all roots by default); their sources S1.. by
    first appearance (support first).
    """
    root_of = root_ids(sub_questions)
    order = top_level if top_level is not None else roots(sub_questions)
    position = {sq.id: i for i, sq in enumerate(order, start=1)}
    sources_by_id = _state_sources(state)
    by_source: dict[str, _LabeledSource] = {}

    def source_label(source_id: str, passage_id: str) -> str:
        source = by_source.get(source_id)
        if source is None:
            row = sources_by_id[source_id]
            source = _LabeledSource(
                label=f"S{len(by_source) + 1}",
                source_id=source_id,
                url=row.get("url", ""),
                title=row.get("title", ""),
            )
            by_source[source_id] = source
        if passage_id not in source.passage_ids:
            source.passage_ids.append(passage_id)
        return source.label

    labeled: dict[str, _LabeledClaim] = {}
    for index, claim in enumerate(claims, start=1):
        entry = _LabeledClaim(
            label=f"C{index}",
            claim=claim,
            position=position.get(root_of.get(claim.sub_question_id or "", ""), 0),
        )
        cited_stances = ("supports", "contradicts") if claim.verdict == "mixed" else ("supports",)
        for stance in ("supports", "contradicts"):
            for link in claim.evidence:
                if link.stance != stance:
                    continue
                label = source_label(link.source_id, link.passage_id)
                if stance in cited_stances and label not in entry.citations:
                    entry.citations.append(label)
        entry.citations.sort(key=label_sort_key)
        entry.supporting_sites = len(
            {
                domain_of(sources_by_id[e.source_id].get("url", ""))
                for e in claim.evidence
                if e.stance == "supports"
            }
        )
        labeled[entry.label] = entry
    return labeled, {s.label: s for s in by_source.values()}


async def _verified_report(
    runtime: AgentRuntime,
    state: ResearchState,
    profile: ModeProfile,
    sub_questions: list[SubQuestionItem],
    all_claims: list[ClaimItem],
    usable: list[ClaimItem],
    all_contradictions: list[ContradictionItem],
    stored_count: int,
    log: EventLog,
) -> tuple[ResearchReport, list[dict[str, Any]]]:
    run_id = state["run_id"]
    root_of = root_ids(sub_questions)
    # Required sub-questions, plus any later-round one that found evidence:
    # a gap round that finds nothing does not count against the report.
    top_level = report_roots(
        sub_questions, {root_of.get(c.sub_question_id or "", "") for c in usable}
    )
    labeled, sources = _label_claims(usable, sub_questions, state, top_level)
    top_ids = {sq.id for sq in top_level}
    covered = {root_of.get(lc.claim.sub_question_id or "", "") for lc in labeled.values()} & top_ids
    contradictions = _report_contradictions(all_contradictions, labeled)
    writer = _DECISION_WRITER if profile.recommendation else _CLAIM_WRITER

    draft = _Draft()
    if runtime.budget.exceeded:
        generation = "extractive"
        draft.failure = "the run's token/cost budget was spent"
    else:
        objective = str(state.get("plan", {}).get("objective") or state["question"])
        prompt = build_verified_synthesis_prompt(
            question=state["question"],
            objective=objective,
            sub_questions=[(sq.text, sq.id in covered) for sq in top_level],
            contradictions=[
                PromptContradiction(
                    label=x.label,
                    claims=x.claims,
                    likely_cause=x.likely_cause,
                    explanation=x.explanation,
                    resolved=x.resolved,
                )
                for x in contradictions
            ],
            recommendation=profile.recommendation,
            user_context=state.get("user_context") or None,
            claims=[
                PromptClaim(
                    label=lc.label,
                    text=lc.claim.text,
                    verdict=lc.claim.verdict,
                    confidence=lc.claim.confidence,
                    sources=lc.supporting_sites,
                    sub_question=lc.position,
                )
                for lc in labeled.values()
            ],
        )
        draft = await _write(runtime, run_id, prompt, set(labeled), writer, log)
        generation = "llm" if draft.output is not None else "extractive"
    if generation == "extractive":
        await log.emit("extractive_fallback", reason=draft.failure)

    if draft.output is not None:
        title = draft.output.title or state["question"]
        summary = [_cite_claims(s, labeled) for s in draft.output.executive_summary]
        sections = [
            ReportSection(
                heading=sec.heading, statements=[_cite_claims(s, labeled) for s in sec.statements]
            )
            for sec in draft.output.sections
        ]
        recommendation = [_cite_claims(s, labeled) for s in draft.output.recommendation]
        model_limitations = list(draft.output.limitations)
    else:
        title = state["question"]
        summary = []
        sections = _extractive_claim_sections(top_level, labeled)
        recommendation = []
        model_limitations = []

    counts = Counter(
        label
        for statement in [
            *summary,
            *(s for sec in sections for s in sec.statements),
            *recommendation,
        ]
        for label in statement.claims
    )
    unresolved = [x for x in contradictions if not x.resolved]
    confidence = overall_confidence(
        [(labeled[label].claim.confidence, n) for label, n in counts.items()],
        sub_questions=len(top_level),
        covered_sub_questions=len(covered),
        unresolved_contradictions=len(unresolved),
        config=runtime.config.scoring,
    )

    verdicts = Counter(c.verdict for c in all_claims)
    limitations = list(model_limitations)
    for sq in top_level:
        if sq.id not in covered:
            limitations.append(f"No verified claim was found for: {sq.text}")
    limitations += _gap_limitations(state, covered)
    for item in unresolved:
        limitations.append(
            "Sources disagree, and the evidence does not explain why: " + item.explanation
        )
    if profile.recommendation and not recommendation:
        limitations.append("No recommendation could be grounded in the verified claims.")
    if verdicts.get("unsupported"):
        limitations.append(
            f"{verdicts['unsupported']} extracted claim(s) were not supported by their sources "
            "and were left out."
        )
    if verdicts.get("unverified"):
        limitations.append(
            f"{verdicts['unverified']} claim(s) could not be fact-checked and were left out."
        )
    limitations += _common_limitations(runtime, generation, draft)

    # Every labeled source backs (or, for mixed claims, disputes) a key finding.
    citations = [
        CitationItem(
            label=label,
            source_id=sources[label].source_id,
            url=sources[label].url,
            title=sources[label].title,
            passage_ids=list(sources[label].passage_ids),
        )
        for label in sorted(sources, key=label_sort_key)
    ]
    label_of_source = {s.source_id: label for label, s in sources.items()}
    key_findings = [
        ReportClaim(
            label=lc.label,
            claim_id=lc.claim.id,
            text=lc.claim.text,
            sub_question_id=lc.claim.sub_question_id,
            verdict=lc.claim.verdict,
            confidence=lc.claim.confidence,
            citations=lc.citations,
            evidence=[
                ReportEvidence(
                    passage_id=e.passage_id,
                    source_label=label_of_source[e.source_id],
                    stance=e.stance,
                    excerpt=e.excerpt,
                )
                for e in lc.claim.evidence
                if e.stance != "neutral" and e.source_id in label_of_source
            ],
        )
        for lc in labeled.values()
    ]
    evidence_passages = {e.passage_id for f in key_findings for e in f.evidence}

    report = ResearchReport(
        title=title,
        question=state["question"],
        mode=state.get("mode", "deep"),
        executive_summary=summary,
        sections=sections,
        recommendation=recommendation,
        limitations=list(dict.fromkeys(limitations)),
        methodology=_verified_methodology(
            state, top_level, all_claims, labeled, contradictions, generation, profile
        ),
        sources=citations,
        coverage={
            # The sub-questions the report covers (the confidence's denominator).
            "sub_questions": len(top_level),
            "follow_up_sub_questions": sum(1 for sq in sub_questions if sq.iteration > 0),
            "sub_questions_with_evidence": len(covered),
            "iterations": rounds_run(sub_questions),
            "sources_stored": stored_count,
            "sources_shown": len(sources),
            "sources_cited": len(citations),
            "passages_shown": len(evidence_passages),
            "tokens_used": runtime.budget.tokens_used,
            "cost_usd": runtime.budget.cost_usd,
            "budget_exhausted": runtime.budget.research_exhausted,
        },
        key_findings=key_findings,
        contradictory_evidence=contradictions,
        verification={
            "claims_extracted": len(all_claims),
            "claims_checked": sum(n for v, n in verdicts.items() if v != "unverified"),
            "by_verdict": dict(sorted(verdicts.items())),
            "claims_used": len(labeled),
            "claims_cited": len(counts),
            "contradictions_detected": len(all_contradictions),
            "contradictions": len(contradictions),
            "unresolved_contradictions": len(unresolved),
            "scoring_version": SCORING_VERSION,
        },
        overall_confidence=confidence,
        generation=generation,  # type: ignore[arg-type]
        dropped_statements=draft.dropped,
        prompt_version=writer.version if generation == "llm" else "",
    )
    return report, draft.llm_calls


def _report_contradictions(
    contradictions: list[ContradictionItem], labeled: dict[str, _LabeledClaim]
) -> list[ReportContradiction]:
    """Contradictions with at least two labeled claims, labeled X1, X2, ..."""
    label_of = {lc.claim.id: label for label, lc in labeled.items()}
    result: list[ReportContradiction] = []
    for item in contradictions:
        claims = sorted(
            {label_of[cid] for cid in item.claim_ids if cid in label_of}, key=label_sort_key
        )
        if len(claims) < 2:
            continue
        citations = sorted(
            {source for label in claims for source in labeled[label].citations},
            key=label_sort_key,
        )
        result.append(
            ReportContradiction(
                label=f"X{len(result) + 1}",
                contradiction_id=item.id,
                claims=claims,
                citations=citations,
                likely_cause=item.likely_cause,
                explanation=item.explanation,
                resolved=item.resolved,
            )
        )
    return result


def _gap_limitations(state: ResearchState, covered: set[str]) -> list[str]:
    """What the last sufficiency check still found missing, if the loop stopped there."""
    sufficiency = state.get("sufficiency") or {}
    if sufficiency.get("status") != "needs_more":
        return []
    limitations = []
    texts = {sq["id"]: sq.get("text", "") for sq in state.get("sub_questions", [])}
    for raw in sufficiency.get("gaps", []):
        gap = GapItem.model_validate(raw)
        if gap.sub_question_id is None:
            limitations.append(f"Not established: {gap.description}")
        elif gap.sub_question_id in covered:
            # Uncovered sub-questions already have their own limitation.
            limitations.append(
                f"Only limited verified evidence was found for: {texts.get(gap.sub_question_id, '')}"
            )
    return limitations


def _cite_claims(statement: ReportStatement, labeled: dict[str, _LabeledClaim]) -> ReportStatement:
    """A statement citing claim labels -> the same statement citing their sources."""
    claim_labels = [label for label in statement.citations if label in labeled]
    sources = {source for label in claim_labels for source in labeled[label].citations}
    return ReportStatement(
        text=statement.text,
        citations=sorted(sources, key=label_sort_key),
        claims=claim_labels,
    )


def _extractive_claim_sections(
    sub_questions: list[SubQuestionItem], labeled: dict[str, _LabeledClaim]
) -> list[ReportSection]:
    """The best claims per (top-level) sub-question, quoted as statements."""
    sections = []
    for position, sq in enumerate([*sub_questions, None], start=1):
        members = [
            lc
            for lc in labeled.values()
            if (lc.position == position if sq is not None else lc.position == 0)
        ][:_EXTRACTIVE_CLAIMS_PER_SECTION]
        statements = [
            ReportStatement(text=lc.claim.text, citations=lc.citations, claims=[lc.label])
            for lc in members
            if lc.citations
        ]
        if statements:
            sections.append(
                ReportSection(heading=sq.text if sq else "Other findings", statements=statements)
            )
    return sections


def _verified_methodology(
    state: ResearchState,
    sub_questions: list[SubQuestionItem],
    all_claims: list[ClaimItem],
    labeled: dict[str, _LabeledClaim],
    contradictions: list[ReportContradiction],
    generation: str,
    profile: ModeProfile,
) -> str:
    agents = sorted({r.get("agent", "web_researcher") for r in state.get("research_results", [])})
    planned = [SubQuestionItem.model_validate(sq) for sq in state.get("sub_questions", [])]
    rounds = rounds_run(planned)
    follow_ups = sum(1 for sq in planned if sq.iteration > 0)
    text = (
        f"Mode: {state.get('mode', 'deep')}. The question was split into "
        f"{len(planned) - follow_ups} sub-question(s)"
        + (f", plus {follow_ups} gap-filling sub-question(s)" if follow_ups > 0 else "")
        + ", researched by "
        f"{', '.join(a.replace('_', ' ') for a in agents) or 'no researcher'}, which stored "
        f"{len(state.get('sources', []))} source(s) as {len(state.get('passages', []))} "
        "passage(s). Each source's reliability was scored from its type, age, and, for "
        "papers, citations and methodology. "
        f"{len(all_claims)} atomic claim(s) were extracted from the most relevant passages, "
        "each linked to the passages it came from. A fact checker compared every claim with "
        "those passages and with similar passages from other sources, judging each passage "
        "as supporting, contradicting, or neutral. Claim confidence was computed from the "
        "verdict, the reliability of the supporting and contradicting sources, and how many "
        f"independent sites agree; {len(labeled)} claim(s) were usable. "
    )
    if profile.loop:
        unresolved = sum(1 for x in contradictions if not x.resolved)
        text += (
            "Similar claims from different sources were compared for contradictions, each "
            f"with a likely cause; {len(contradictions)} were found among the usable claims "
            f"({unresolved} unexplained by a difference in scope). After each round a "
            "sufficiency check counted the verified claims and independent sites per "
            "sub-question and planned new research for the gaps; "
            f"{rounds} research round(s) ran. "
        )
    if generation == "llm":
        text += (
            "A language model wrote the report from those claims only; every statement had "
            "to cite claims, and its sources are those of the claims it cites. The overall "
            "confidence is a weighted average of the cited claims' confidence, reduced for "
            "sub-questions without verified claims and for unresolved contradictions. No "
            "number in this report was produced by a language model."
        )
    else:
        text += "No language model was used for the report; it lists the claims directly."
    return text


# ---------------------------------------------------------------------------
# Passage path (quick mode, and the verified modes' fallback)
# ---------------------------------------------------------------------------


async def _passage_report(
    runtime: AgentRuntime,
    state: ResearchState,
    sub_questions: list[SubQuestionItem],
    stored: set[str],
    claims: list[ClaimItem],
    verify_mode: bool,
    log: EventLog,
) -> tuple[ResearchReport, list[dict[str, Any]]]:
    run_id = state["run_id"]
    config = runtime.config.synthesizer
    evidence = await gather_evidence(
        runtime,
        state,
        sub_questions,
        log,
        top_k=config.top_k_per_sub_question,
        max_passages=config.max_passages,
        min_similarity=config.min_similarity,
    )
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
        draft = await _write(runtime, run_id, prompt, set(sources), _PASSAGE_WRITER, log)
        generation = "llm" if draft.output is not None else "extractive"

    if generation == "extractive":
        await log.emit("extractive_fallback", reason=draft.failure)

    if draft.output is not None:
        title = draft.output.title or state["question"]
        summary = draft.output.executive_summary
        sections = draft.output.sections
        model_limitations = list(draft.output.limitations)
    else:
        title = state["question"]
        summary = []
        sections = _extractive_sections(sub_questions, evidence, sources)
        model_limitations = []

    limitations = list(model_limitations)
    for sq in sub_questions:
        if sq.id not in covered:
            limitations.append(f"No usable sources were found for: {sq.text}")
    verification: dict[str, Any] = {}
    if verify_mode:
        verdicts = Counter(c.verdict for c in claims)
        limitations.append(
            "No claim could be verified, so this report is written from retrieved passages "
            "and has no confidence score."
        )
        verification = {
            "claims_extracted": len(claims),
            "claims_checked": sum(n for v, n in verdicts.items() if v != "unverified"),
            "by_verdict": dict(sorted(verdicts.items())),
            "claims_used": 0,
            "claims_cited": 0,
            "scoring_version": SCORING_VERSION,
        }
    limitations += _common_limitations(runtime, generation, draft)

    used = cited_labels(
        ReportOutput(title=title, executive_summary=summary, sections=sections, limitations=[])
    )
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

    report = ResearchReport(
        title=title,
        question=state["question"],
        mode=state.get("mode", "quick"),
        executive_summary=summary,
        sections=sections,
        limitations=list(dict.fromkeys(limitations)),
        methodology=_methodology(state, sub_questions, evidence, sources, generation),
        sources=citations,
        coverage={
            "sub_questions": len(sub_questions),
            "sub_questions_with_evidence": len(covered & {sq.id for sq in sub_questions}),
            "sources_stored": len(stored),
            "sources_shown": len(sources),
            "sources_cited": len(citations),
            "passages_shown": len(evidence),
            "tokens_used": runtime.budget.tokens_used,
            "cost_usd": runtime.budget.cost_usd,
            "budget_exhausted": runtime.budget.research_exhausted,
        },
        verification=verification,
        overall_confidence=None,
        generation=generation,  # type: ignore[arg-type]
        dropped_statements=draft.dropped,
        prompt_version=PROMPT_VERSION if generation == "llm" else "",
    )
    return report, draft.llm_calls


def _common_limitations(runtime: AgentRuntime, generation: str, draft: _Draft) -> list[str]:
    limitations = []
    if runtime.budget.research_exhausted:
        limitations.append(
            "The run reached its token/cost budget, so research stopped early and some "
            "sub-questions may be under-covered."
        )
    if generation == "extractive":
        limitations.append(
            "The language model did not write this report "
            f"({draft.failure or 'unavailable'}); it quotes the evidence directly instead."
        )
    return limitations


def _label_sources(evidence: list[Evidence]) -> dict[str, _LabeledSource]:
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


def _extractive_sections(
    sub_questions: list[SubQuestionItem],
    evidence: list[Evidence],
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
    evidence: list[Evidence],
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
            "sources. Claims are not fact-checked in this mode, so there is no confidence score."
        )
    elif generation == "extractive":
        text += "No language model was used for the report; it quotes passages directly."
    else:
        text += "No usable passages were found, so the report has no findings."
    return text
