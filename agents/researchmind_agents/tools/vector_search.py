"""
Vector search interface for the agents layer.

The agents package never talks to the database directly. Instead, it
uses this interface, which accepts a callable conforming to VectorSearchFn.
The backend wires the real PassageRepository implementation at startup.

This keeps the agents layer free of SQLAlchemy / DB-session code.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol / type alias
# ---------------------------------------------------------------------------

@dataclass
class VectorSearchResult:
    """A single passage returned by vector similarity search."""

    passage_id: str
    source_id: str
    text: str
    score: float           # cosine similarity, 0-1
    metadata: dict[str, Any]


# Callable type that the backend provides:
#   async (query_vector, top_k, run_id?) -> list[VectorSearchResult]
VectorSearchFn = Callable[
    [list[float], int, str | None],
    Awaitable[list[VectorSearchResult]],
]


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class VectorSearchTool:
    """
    Wraps an injected VectorSearchFn so agent nodes can call it uniformly.

    Usage (inside an agent node)::

        tool = VectorSearchTool(search_fn=passage_repo.search_similar_wrapped)
        results = await tool.search(query_vector, top_k=5, run_id=state.run_id)
    """

    def __init__(self, search_fn: VectorSearchFn) -> None:
        self._search_fn = search_fn

    async def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        run_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """
        Find the *top_k* passages most similar to *query_vector*.

        Filters by *run_id* when provided (scoped search within a research run).
        """
        try:
            results = await self._search_fn(query_vector, top_k, run_id)
            logger.debug(
                "Vector search top_k=%d run_id=%r → %d results",
                top_k,
                run_id,
                len(results),
            )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.error("Vector search failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Helper: adapt PassageRepository.search_similar to VectorSearchFn
# ---------------------------------------------------------------------------


def make_vector_search_fn(passage_repo: Any) -> VectorSearchFn:
    """
    Adapt a PassageRepository instance to the VectorSearchFn interface.

    Call this in the backend service layer::

        search_fn = make_vector_search_fn(passage_repo)
        tool = VectorSearchTool(search_fn=search_fn)

    The returned coroutine converts ORM Passage objects to VectorSearchResult.
    """

    async def _search(
        query_vector: list[float],
        top_k: int,
        run_id: str | None,
    ) -> list[VectorSearchResult]:
        rows = await passage_repo.search_similar(
            embedding=query_vector,
            limit=top_k,
            run_id=run_id,
        )
        return [
            VectorSearchResult(
                passage_id=str(row.id),
                source_id=str(row.source_id),
                text=row.text,
                score=getattr(row, "_similarity_score", 0.0),
                metadata={
                    "chunk_index": row.chunk_index,
                    "token_count": row.token_count,
                },
            )
            for row in rows
        ]

    return _search
