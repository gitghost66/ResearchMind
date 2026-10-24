"""
Vector search interface for the agents layer.

The agents package never touches the database. It calls a
:class:`VectorSearchFn` that the backend wires to the real
``PassageRepository`` at startup, which keeps SQLAlchemy and DB sessions out of
this package (AGENTS.md rule 2).

The callable takes keyword arguments on purpose: the previous positional
contract silently mismatched the repository's parameter order, and a
keyword-only protocol makes that class of bug impossible.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "VectorSearchResult",
    "VectorSearchFn",
    "VectorSearchTool",
    "make_vector_search_fn",
    "similarity_from_distance",
]


# ---------------------------------------------------------------------------
# Result and protocol
# ---------------------------------------------------------------------------


class VectorSearchResult:
    """
    A passage returned by vector similarity search.

    ``score`` is a cosine similarity in ``[0, 1]`` (higher is closer);
    ``distance`` is the raw cosine distance the store reported.
    """

    __slots__ = ("passage_id", "source_id", "text", "score", "distance", "metadata")

    def __init__(
        self,
        passage_id: str,
        source_id: str,
        text: str,
        score: float,
        metadata: dict[str, Any] | None = None,
        distance: float | None = None,
    ) -> None:
        self.passage_id = passage_id
        self.source_id = source_id
        self.text = text
        self.score = score
        self.distance = distance if distance is not None else 1.0 - score
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"VectorSearchResult(passage_id={self.passage_id!r}, "
            f"score={self.score:.4f}, text={self.text[:40]!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorSearchResult):
            return NotImplemented
        return (
            self.passage_id == other.passage_id
            and self.source_id == other.source_id
            and self.text == other.text
            and self.score == other.score
            and self.metadata == other.metadata
        )


class VectorSearchFn(Protocol):
    """What the backend must provide for the agents layer to search passages."""

    async def __call__(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        run_id: str | None = None,
        sub_question_id: str | None = None,
    ) -> list[VectorSearchResult]: ...


def similarity_from_distance(distance: float) -> float:
    """
    Convert a cosine distance (``[0, 2]``) to a similarity in ``[0, 1]``.

    Clamped, so a near-opposite vector reads as 0 rather than a negative score
    that would corrupt downstream confidence arithmetic.
    """
    return max(0.0, min(1.0, 1.0 - distance))


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class VectorSearchTool:
    """
    Wraps an injected :class:`VectorSearchFn` so agent nodes call it uniformly.

    Usage (in the backend)::

        tool = VectorSearchTool(make_vector_search_fn(passage_repo))

    Usage (in an agent node)::

        hits = await tool.search(query_vector, top_k=5, run_id=state.run_id)
    """

    def __init__(self, search_fn: VectorSearchFn) -> None:
        self._search_fn = search_fn

    async def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        run_id: str | None = None,
        sub_question_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """
        Find the *top_k* passages most similar to *query_vector*.

        Scoped to *run_id* / *sub_question_id* when given. Returns an empty list
        on failure so retrieval degrades instead of killing the run.
        """
        if not query_vector:
            logger.warning("Vector search called with an empty query vector")
            return []

        try:
            results = await self._search_fn(
                query_vector,
                top_k=top_k,
                run_id=run_id,
                sub_question_id=sub_question_id,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash the run
            logger.error("Vector search failed: %s", exc)
            return []

        if min_score > 0.0:
            results = [r for r in results if r.score >= min_score]

        logger.debug(
            "Vector search top_k=%d run_id=%r -> %d results", top_k, run_id, len(results)
        )
        return results


# ---------------------------------------------------------------------------
# Backend adapter
# ---------------------------------------------------------------------------


def make_vector_search_fn(passage_repo: Any) -> VectorSearchFn:
    """
    Adapt a ``PassageRepository`` to :class:`VectorSearchFn`.

    Call this in the backend service layer::

        tool = VectorSearchTool(make_vector_search_fn(passage_repo))

    The repository yields ``(Passage, cosine_distance)`` pairs; this converts
    them into :class:`VectorSearchResult` with a similarity score and carries
    the source URL/title through so the synthesizer can cite without a second
    query.
    """

    async def search(
        query_vector: list[float],
        *,
        top_k: int = 5,
        run_id: str | None = None,
        sub_question_id: str | None = None,
    ) -> list[VectorSearchResult]:
        rows = await passage_repo.search_similar(
            query_embedding=query_vector,
            run_id=run_id,
            sub_question_id=sub_question_id,
            limit=top_k,
        )
        return [_to_result(passage, distance) for passage, distance in rows]

    return search


def _to_result(passage: Any, distance: float) -> VectorSearchResult:
    """Convert one ``(Passage, distance)`` row into a search result."""
    source = getattr(passage, "source", None)
    sub_question_id = getattr(passage, "sub_question_id", None)

    return VectorSearchResult(
        passage_id=str(passage.id),
        source_id=str(passage.source_id),
        text=passage.text,
        score=similarity_from_distance(float(distance)),
        distance=float(distance),
        metadata={
            "chunk_index": getattr(passage, "chunk_index", 0),
            "sub_question_id": str(sub_question_id) if sub_question_id else None,
            "source_url": getattr(source, "url", None),
            "source_title": getattr(source, "title", None),
            "reliability_score": getattr(source, "reliability_score", None),
        },
    )
