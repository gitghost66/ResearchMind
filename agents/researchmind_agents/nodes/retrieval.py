"""
Run-scoped passage retrieval, shared by the claim extractor and synthesizer.

For each sub-question, the most similar passages are found by vector search
over the whole run (a page one branch found can serve every sub-question),
then merged round-robin so every sub-question gets evidence, up to a cap.
Without vector search, or when it finds nothing, the passages the researchers
put in the state are used instead, in discovery order.

Passages in *exclude* (already read in an earlier round) are skipped; the
search asks for extra hits to make up for them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import dataclass, field

from researchmind_agents.nodes.common import EventLog
from researchmind_agents.runtime import AgentRuntime
from researchmind_agents.state import ResearchState, SubQuestionItem

__all__ = ["Evidence", "gather_evidence", "merge_round_robin"]


@dataclass
class Evidence:
    passage_id: str
    source_id: str
    url: str
    title: str
    text: str
    chunk_index: int
    # Sub-questions this passage was retrieved for, best match first.
    sub_question_ids: list[str] = field(default_factory=list)

    @property
    def primary_sub_question_id(self) -> str | None:
        return self.sub_question_ids[0] if self.sub_question_ids else None


async def gather_evidence(
    runtime: AgentRuntime,
    state: ResearchState,
    sub_questions: list[SubQuestionItem],
    log: EventLog,
    *,
    top_k: int,
    max_passages: int,
    min_similarity: float = 0.0,
    exclude: Collection[str] = (),
) -> list[Evidence]:
    """Top passages per sub-question, merged round-robin up to *max_passages*."""
    skip = set(exclude)
    per_question: list[list[Evidence]] = []
    if runtime.vector_search is not None and sub_questions:
        try:
            per_question = await _retrieve(
                runtime,
                state["run_id"],
                sub_questions,
                top_k=top_k,
                min_similarity=min_similarity,
                exclude=skip,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to state passages
            await log.error("Vector retrieval failed", reason=f"{type(exc).__name__}: {exc}")
            per_question = []

    if not any(per_question):
        per_question = _from_state(state, sub_questions, top_k, skip)
        if any(per_question):
            await log.emit("retrieval_fallback", reason="vector search returned no passages")

    return merge_round_robin(per_question, max_passages)


async def _retrieve(
    runtime: AgentRuntime,
    run_id: str,
    sub_questions: list[SubQuestionItem],
    *,
    top_k: int,
    min_similarity: float,
    exclude: set[str],
) -> list[list[Evidence]]:
    assert runtime.vector_search is not None
    vector_search = runtime.vector_search
    vectors = await runtime.embeddings.embed_queries([sq.text for sq in sub_questions])
    # Room for hits that are skipped as already read.
    extra = min(len(exclude), max(0, top_k) * 2)

    async def one(sub_question: SubQuestionItem, vector: list[float]) -> list[Evidence]:
        if not vector:
            return []
        hits = await vector_search.search(
            vector, top_k=top_k + extra, run_id=run_id, min_score=min_similarity
        )
        hits = [hit for hit in hits if hit.passage_id not in exclude][: max(0, top_k)]
        return [
            Evidence(
                passage_id=hit.passage_id,
                source_id=hit.source_id,
                url=str(hit.metadata.get("source_url") or ""),
                title=str(hit.metadata.get("source_title") or ""),
                text=hit.text,
                chunk_index=int(hit.metadata.get("chunk_index") or 0),
                sub_question_ids=[sub_question.id],
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
    exclude: set[str],
) -> list[list[Evidence]]:
    """Passages the researchers stored, in discovery order."""
    sources = {s["id"]: s for s in state.get("sources", [])}
    passages = state.get("passages", [])
    per_question = []
    for sq in sub_questions:
        picked = [
            p
            for p in passages
            if p.get("sub_question_id") == sq.id
            and p["source_id"] in sources
            and p["id"] not in exclude
        ][: max(0, top_k)]
        per_question.append(
            [
                Evidence(
                    passage_id=p["id"],
                    source_id=p["source_id"],
                    url=sources[p["source_id"]]["url"],
                    title=sources[p["source_id"]].get("title", ""),
                    text=p["text"],
                    chunk_index=int(p.get("chunk_index", 0)),
                    sub_question_ids=[sq.id],
                )
                for p in picked
            ]
        )
    return per_question


def merge_round_robin(per_question: list[list[Evidence]], max_passages: int) -> list[Evidence]:
    """Round-robin across sub-questions so each gets evidence, de-duplicated by passage."""
    merged: dict[str, Evidence] = {}
    for rank in range(max((len(hits) for hits in per_question), default=0)):
        for hits in per_question:
            if rank >= len(hits):
                continue
            evidence = hits[rank]
            existing = merged.get(evidence.passage_id)
            if existing is not None:
                for sq_id in evidence.sub_question_ids:
                    if sq_id not in existing.sub_question_ids:
                        existing.sub_question_ids.append(sq_id)
            elif len(merged) < max(0, max_passages):
                merged[evidence.passage_id] = evidence
    return list(merged.values())
