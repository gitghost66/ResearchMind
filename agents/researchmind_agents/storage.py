"""
Storage contract for the research graph.

Nodes persist through a :class:`ResearchStore` the caller passes in, so this
package never opens a database session (AGENTS.md rule 2). The backend supplies
a SQL-backed implementation; :class:`InMemoryResearchStore` serves tests,
evals, and DB-less script runs, and doubles as a vector search function.

Every method is keyword-only, the same rule as ``VectorSearchFn``: a positional
contract between two packages drifts silently.

Failure policy, which nodes rely on:

* ``save_plan``, ``save_report`` — the run cannot be grounded without them, so
  errors propagate.
* ``save_source``, ``save_passages`` — errors skip that source; a source that
  was not stored never reaches the report.
* ``save_claims``, ``save_contradictions`` — errors are logged as an
  ``error`` event; the records still reach the report from graph state.
* ``log_event`` — telemetry; errors are logged and swallowed by callers.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Protocol

from researchmind_agents.state import (
    ClaimItem,
    ContradictionItem,
    PassageItem,
    ResearchReport,
    SourceItem,
    SubQuestionItem,
)
from researchmind_agents.tools.vector_search import VectorSearchResult, similarity_from_distance

__all__ = ["ResearchStore", "InMemoryResearchStore"]


class ResearchStore(Protocol):
    """What the backend must provide for the graph to persist a run."""

    async def save_plan(
        self,
        *,
        run_id: str,
        plan: dict[str, Any],
        sub_questions: list[SubQuestionItem],
        iteration: int = 0,
    ) -> None:
        """
        Store the plan and create one row per sub-question, using the given IDs.

        *iteration* is the research round (0 first). A gap-filling round
        passes the whole plan so far, but only its new sub-questions, and the
        run's iteration is set to it.
        """
        ...

    async def set_sub_question_status(
        self,
        *,
        run_id: str,
        sub_question_id: str,
        status: str,
    ) -> None: ...

    async def save_source(self, *, run_id: str, source: SourceItem) -> tuple[str, bool]:
        """
        Get-or-create a source by URL or content hash within the run.

        Returns ``(source_id, created)``. When the source already exists the
        existing ID is returned with ``created=False`` and the caller must not
        store its passages again. Must be safe under concurrent calls.
        """
        ...

    async def save_passages(
        self,
        *,
        run_id: str,
        passages: list[PassageItem],
        embeddings: list[list[float] | None],
        embedding_model: str,
    ) -> None:
        """Store passages; ``embeddings[i]`` belongs to ``passages[i]`` (``None`` if it failed)."""
        ...

    async def save_claims(self, *, run_id: str, claims: list[ClaimItem]) -> None:
        """
        Store claims with their verdicts, confidence, and evidence links
        (``claims`` and ``claim_evidence``). Every evidence passage is a
        stored passage of the run; one link per (claim, passage).
        """
        ...

    async def save_contradictions(
        self, *, run_id: str, contradictions: list[ContradictionItem]
    ) -> None:
        """
        Store contradictions and their claim links (``contradictions`` and
        ``contradiction_claims``), using the given IDs. Every claim is a
        stored claim of the run.
        """
        ...

    async def save_report(self, *, run_id: str, report: ResearchReport, markdown: str) -> str:
        """Store the final report and return its ID."""
        ...

    async def log_event(
        self,
        *,
        run_id: str,
        agent: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None: ...

    async def list_source_ids(self, *, run_id: str) -> set[str]:
        """IDs of every source stored for the run (used to verify citations)."""
        ...


class InMemoryResearchStore:
    """
    A complete :class:`ResearchStore` held in process memory.

    Also implements ``VectorSearchFn`` through :meth:`vector_search`, with an
    exact cosine scan, so a graph can run end to end with no database.
    """

    def __init__(self) -> None:
        self.plans: dict[str, dict[str, Any]] = {}
        self.sub_questions: dict[str, dict[str, Any]] = {}
        self.sources: dict[str, dict[str, Any]] = {}
        self.passages: dict[str, dict[str, Any]] = {}
        self.claims: dict[str, dict[str, Any]] = {}
        self.contradictions: dict[str, dict[str, Any]] = {}
        self.iterations: dict[str, int] = {}
        self.reports: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self._source_lock = asyncio.Lock()

    async def save_plan(
        self,
        *,
        run_id: str,
        plan: dict[str, Any],
        sub_questions: list[SubQuestionItem],
        iteration: int = 0,
    ) -> None:
        self.plans[run_id] = dict(plan)
        self.iterations[run_id] = iteration
        for item in sub_questions:
            self.sub_questions[item.id] = {"run_id": run_id, **item.model_dump()}

    async def set_sub_question_status(
        self,
        *,
        run_id: str,
        sub_question_id: str,
        status: str,
    ) -> None:
        row = self.sub_questions.get(sub_question_id)
        if row is not None and row["run_id"] == run_id:
            row["status"] = status

    async def save_source(self, *, run_id: str, source: SourceItem) -> tuple[str, bool]:
        async with self._source_lock:
            for existing in self.sources.values():
                if existing["run_id"] != run_id:
                    continue
                same_url = existing["url"] == source.url
                same_hash = bool(source.content_hash) and (
                    existing["content_hash"] == source.content_hash
                )
                if same_url or same_hash:
                    return existing["id"], False
            self.sources[source.id] = {"run_id": run_id, **source.model_dump()}
            return source.id, True

    async def save_passages(
        self,
        *,
        run_id: str,
        passages: list[PassageItem],
        embeddings: list[list[float] | None],
        embedding_model: str,
    ) -> None:
        if len(passages) != len(embeddings):
            raise ValueError("passages and embeddings must have the same length")
        for passage, vector in zip(passages, embeddings, strict=True):
            if passage.source_id not in self.sources:
                raise ValueError(f"unknown source_id {passage.source_id!r}")
            self.passages[passage.id] = {
                "run_id": run_id,
                **passage.model_dump(),
                "embedding": list(vector) if vector else None,
                "embedding_model": embedding_model,
            }

    async def save_claims(self, *, run_id: str, claims: list[ClaimItem]) -> None:
        # Validate everything first, so a bad claim stores nothing (as a
        # rolled-back transaction would).
        for claim in claims:
            if not claim.evidence:
                raise ValueError(f"claim {claim.id!r} has no evidence")
            for link in claim.evidence:
                row = self.passages.get(link.passage_id)
                if row is None or row["run_id"] != run_id:
                    raise ValueError(f"unknown passage_id {link.passage_id!r}")
        for claim in claims:
            seen: set[str] = set()
            evidence = []
            for link in claim.evidence:
                if link.passage_id not in seen:
                    seen.add(link.passage_id)
                    evidence.append(link.model_dump())
            self.claims[claim.id] = {
                "run_id": run_id,
                **claim.model_dump(exclude={"evidence"}),
                "evidence": evidence,
            }

    async def save_contradictions(
        self, *, run_id: str, contradictions: list[ContradictionItem]
    ) -> None:
        for item in contradictions:
            if len(set(item.claim_ids)) < 2:
                raise ValueError(f"contradiction {item.id!r} needs two claims")
            for claim_id in item.claim_ids:
                row = self.claims.get(claim_id)
                if row is None or row["run_id"] != run_id:
                    raise ValueError(f"unknown claim_id {claim_id!r}")
        for item in contradictions:
            self.contradictions[item.id] = {"run_id": run_id, **item.model_dump()}

    async def save_report(self, *, run_id: str, report: ResearchReport, markdown: str) -> str:
        report_id = f"report-{len(self.reports) + 1}"
        self.reports[report_id] = {
            "id": report_id,
            "run_id": run_id,
            "content": report.model_dump(mode="json"),
            "markdown": markdown,
        }
        return report_id

    async def log_event(
        self,
        *,
        run_id: str,
        agent: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.events.append(
            {"run_id": run_id, "agent": agent, "type": event_type, "payload": dict(payload)}
        )

    async def list_source_ids(self, *, run_id: str) -> set[str]:
        return {sid for sid, row in self.sources.items() if row["run_id"] == run_id}

    # -- VectorSearchFn -------------------------------------------------

    async def vector_search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        run_id: str | None = None,
        sub_question_id: str | None = None,
    ) -> list[VectorSearchResult]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.passages.values():
            if row["embedding"] is None:
                continue
            if run_id is not None and row["run_id"] != run_id:
                continue
            if sub_question_id is not None and row["sub_question_id"] != sub_question_id:
                continue
            scored.append((_cosine_distance(query_vector, row["embedding"]), row))

        scored.sort(key=lambda pair: pair[0])
        results = []
        for distance, row in scored[: max(0, top_k)]:
            source = self.sources.get(row["source_id"], {})
            results.append(
                VectorSearchResult(
                    passage_id=row["id"],
                    source_id=row["source_id"],
                    text=row["text"],
                    score=similarity_from_distance(distance),
                    distance=distance,
                    metadata={
                        "chunk_index": row["chunk_index"],
                        "sub_question_id": row["sub_question_id"],
                        "source_url": source.get("url"),
                        "source_title": source.get("title"),
                    },
                )
            )
        return results


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return max(0.0, min(2.0, 1.0 - dot / (norm_a * norm_b)))
