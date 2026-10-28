"""
Build a run's evidence graph (``GET /api/v1/research/{run_id}/graph``) from
what the run stored.

:func:`load_graph_data` reads the rows; :func:`build_evidence_graph` turns
them into nodes and edges (shapes and edge meanings in ``app.schemas.graph``).
What goes in:

* the question, every sub-question (gap-filling ones under their parent, as
  recorded in the plan), and every claim with its verdict and confidence;
* passages linked to a claim as supporting or contradicting evidence (neutral
  links are left out), plus passages the latest report cites — the only
  passages a quick-mode graph has;
* the sources of those passages, and every source the report cites;
* one ``contradicts`` edge per pair of claims in a contradiction record.

Sources no included passage comes from are left out; ``/sources`` lists them
all. Whether a contradiction is resolved is the agents' scoring rule
(``ScoringConfig.is_resolved``), passed in, never decided here.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.claims import Claim, ClaimEvidence, Contradiction, ContradictionClaim
from app.models.evidence import Passage, Source
from app.models.research import Report, ResearchRun, SubQuestion
from app.schemas.graph import (
    ClaimMeta,
    ClaimNode,
    EvidenceGraph,
    GraphEdge,
    GraphNode,
    PassageMeta,
    PassageNode,
    QuestionMeta,
    QuestionNode,
    SourceMeta,
    SourceNode,
    SubQuestionMeta,
    SubQuestionNode,
)

__all__ = ["GraphData", "build_evidence_graph", "load_graph_data"]

_LABEL_CHARS = 140
_EXCERPT_CHARS = 300


@dataclass
class GraphData:
    """The rows a graph is built from."""

    run: ResearchRun
    sub_questions: Sequence[SubQuestion] = ()
    claims: Sequence[Claim] = ()
    # Claim evidence links, each with its passage.
    links: Sequence[tuple[ClaimEvidence, Passage]] = ()
    sources: Sequence[Source] = ()
    # Each contradiction with the IDs of its claims.
    contradictions: Sequence[tuple[Contradiction, Sequence[uuid.UUID]]] = ()
    # Passages and sources the latest report cites.
    cited_passages: Sequence[Passage] = ()
    cited_source_ids: set[str] = field(default_factory=set)


async def load_graph_data(session: AsyncSession, run: ResearchRun) -> GraphData:
    sub_questions = (
        await session.execute(
            select(SubQuestion)
            .where(SubQuestion.run_id == run.id)
            .order_by(SubQuestion.created_at, SubQuestion.id)
        )
    ).scalars().all()
    claims = (
        await session.execute(
            select(Claim).where(Claim.run_id == run.id).order_by(Claim.created_at, Claim.id)
        )
    ).scalars().all()
    links = [
        (link, passage)
        for link, passage in (
            await session.execute(
                select(ClaimEvidence, Passage)
                .join(Passage, Passage.id == ClaimEvidence.passage_id)
                .join(Claim, Claim.id == ClaimEvidence.claim_id)
                .where(Claim.run_id == run.id)
                .order_by(Claim.created_at, Claim.id, Passage.chunk_index, Passage.id)
            )
        ).all()
    ]
    sources = (
        await session.execute(
            select(Source).where(Source.run_id == run.id).order_by(Source.created_at, Source.id)
        )
    ).scalars().all()

    members: dict[uuid.UUID, list[uuid.UUID]] = {}
    for contradiction_id, claim_id in (
        await session.execute(
            select(ContradictionClaim.contradiction_id, ContradictionClaim.claim_id)
            .join(Contradiction, Contradiction.id == ContradictionClaim.contradiction_id)
            .where(Contradiction.run_id == run.id)
        )
    ).all():
        members.setdefault(contradiction_id, []).append(claim_id)
    contradictions = [
        (row, members.get(row.id, []))
        for row in (
            await session.execute(
                select(Contradiction)
                .where(Contradiction.run_id == run.id)
                .order_by(Contradiction.created_at, Contradiction.id)
            )
        ).scalars()
    ]

    report = (
        await session.execute(
            select(Report)
            .where(Report.run_id == run.id)
            .order_by(Report.version.desc(), Report.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    cited_source_ids: set[str] = set()
    cited_passage_ids: set[uuid.UUID] = set()
    for item in ((report.content or {}).get("sources") or []) if report else []:
        if not isinstance(item, Mapping):
            continue
        cited_source_ids.add(str(item.get("source_id")))
        for passage_id in item.get("passage_ids") or []:
            try:
                cited_passage_ids.add(uuid.UUID(str(passage_id)))
            except ValueError:
                continue
    cited_passages: Sequence[Passage] = []
    if cited_passage_ids:
        cited_passages = (
            await session.execute(
                select(Passage)
                .join(Source, Source.id == Passage.source_id)
                .where(Source.run_id == run.id, Passage.id.in_(cited_passage_ids))
                .order_by(Passage.chunk_index, Passage.id)
            )
        ).scalars().all()

    return GraphData(
        run=run,
        sub_questions=sub_questions,
        claims=claims,
        links=links,
        sources=sources,
        contradictions=contradictions,
        cited_passages=cited_passages,
        cited_source_ids=cited_source_ids,
    )


def build_evidence_graph(data: GraphData, *, is_resolved: Callable[[str], bool]) -> EvidenceGraph:
    run = data.run
    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}

    def edge(kind: str, source: str, target: str, **meta: Any) -> None:
        key = f"{kind}:{source}->{target}"
        if "contradiction_id" in meta:
            key += f":{meta['contradiction_id']}"
        if key not in edges:
            edges[key] = GraphEdge.model_validate(
                {"id": key, "from": source, "to": target, "type": kind, "meta": meta}
            )

    question_id = f"question:{run.id}"
    nodes[question_id] = QuestionNode(
        id=question_id,
        label=_short(run.question),
        meta=QuestionMeta(
            question=run.question, mode=run.mode, status=run.status, iteration=run.iteration
        ),
    )

    planned = {
        str(item.get("id")): item
        for item in (run.plan or {}).get("sub_questions") or []
        if isinstance(item, Mapping)
    }
    for sq in data.sub_questions:
        info = planned.get(str(sq.id), {})
        nodes[f"sub_question:{sq.id}"] = SubQuestionNode(
            id=f"sub_question:{sq.id}",
            label=_short(sq.text),
            meta=SubQuestionMeta(
                text=sq.text,
                domain=sq.domain,
                status=sq.status,
                iteration=int(info.get("iteration") or 0),
                parent_id=str(info["parent_id"]) if info.get("parent_id") else None,
            ),
        )
    for sq in data.sub_questions:
        node_id = f"sub_question:{sq.id}"
        parent = nodes[node_id].meta.parent_id  # type: ignore[union-attr]
        parent_node = f"sub_question:{parent}" if parent else ""
        edge("decomposes", parent_node if parent_node in nodes else question_id, node_id)

    order: dict[uuid.UUID, int] = {}
    for index, claim in enumerate(data.claims):
        order[claim.id] = index
        node_id = f"claim:{claim.id}"
        nodes[node_id] = ClaimNode(
            id=node_id,
            label=_short(claim.text),
            meta=ClaimMeta(
                text=claim.text,
                verdict=claim.verdict,
                confidence=min(1.0, max(0.0, float(claim.confidence))),
                sub_question_id=str(claim.sub_question_id) if claim.sub_question_id else None,
            ),
        )
        parent = f"sub_question:{claim.sub_question_id}"
        edge("decomposes", parent if parent in nodes else question_id, node_id)

    sources = {str(s.id): s for s in data.sources}
    passages: dict[str, Passage] = {}
    for link, passage in data.links:
        if link.stance not in ("supports", "contradicts"):
            continue
        claim_node = f"claim:{link.claim_id}"
        if claim_node not in nodes or str(passage.source_id) not in sources:
            continue
        passages.setdefault(str(passage.id), passage)
        edge(link.stance, f"passage:{passage.id}", claim_node)
    for passage in data.cited_passages:
        if str(passage.source_id) in sources:
            passages.setdefault(str(passage.id), passage)

    shown_sources = {str(p.source_id) for p in passages.values()} | (
        data.cited_source_ids & set(sources)
    )
    for source_id, source in sources.items():
        if source_id not in shown_sources:
            continue
        nodes[f"source:{source_id}"] = SourceNode(
            id=f"source:{source_id}",
            label=_short(source.title or source.url),
            meta=SourceMeta(
                url=source.url,
                title=source.title,
                domain_type=source.domain_type,
                reliability_score=source.reliability_score,
                published_at=source.published_at,
                cited=source_id in data.cited_source_ids,
            ),
        )
    for passage_id, passage in passages.items():
        node_id = f"passage:{passage_id}"
        nodes[node_id] = PassageNode(
            id=node_id,
            label=_short(passage.text),
            meta=PassageMeta(
                source_id=str(passage.source_id),
                chunk_index=passage.chunk_index,
                excerpt=_short(passage.text, _EXCERPT_CHARS),
            ),
        )
        edge("cites", node_id, f"source:{passage.source_id}")

    for contradiction, claim_ids in data.contradictions:
        present = sorted(
            {cid for cid in claim_ids if f"claim:{cid}" in nodes}, key=lambda c: order[c]
        )
        for a, b in combinations(present, 2):
            edge(
                "contradicts",
                f"claim:{a}",
                f"claim:{b}",
                contradiction_id=str(contradiction.id),
                likely_cause=contradiction.likely_cause,
                resolved=is_resolved(contradiction.likely_cause),
                explanation=contradiction.explanation,
            )

    # Edges were added as their targets appeared; drop any whose end is missing.
    kept = [e for e in edges.values() if e.from_ in nodes and e.to in nodes]
    return EvidenceGraph(run_id=str(run.id), nodes=list(nodes.values()), edges=kept)


def _short(text: str, limit: int = _LABEL_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 3]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut) + "..."
