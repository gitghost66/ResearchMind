"""
M2 tests — vector search tool and the repository adapter.

The adapter is tested against objects shaped like the real ORM rows (and, in
``backend/tests/test_vector_search_adapter.py``, against the real repository),
because a mock that accepts any call signature is exactly how the previous
adapter bug stayed invisible.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from researchmind_agents.tools.vector_search import (
    VectorSearchResult,
    VectorSearchTool,
    make_vector_search_fn,
    similarity_from_distance,
)


class FakeSource:
    """Shaped like app.models.evidence.Source."""

    def __init__(self, url: str = "https://example.com/a", title: str = "A") -> None:
        self.url = url
        self.title = title
        self.reliability_score = 0.8


class FakePassage:
    """Shaped like app.models.evidence.Passage — note: no token_count column."""

    def __init__(
        self,
        text: str = "passage text",
        chunk_index: int = 0,
        sub_question_id: uuid.UUID | None = None,
    ) -> None:
        self.id = uuid.uuid4()
        self.source_id = uuid.uuid4()
        self.text = text
        self.chunk_index = chunk_index
        self.sub_question_id = sub_question_id
        self.source = FakeSource()


class FakePassageRepository:
    """Mirrors PassageRepository.search_similar's real signature and return shape."""

    def __init__(self, rows: list[tuple[Any, float]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    async def search_similar(
        self,
        query_embedding: list[float],
        run_id: Any = None,
        sub_question_id: Any = None,
        limit: int = 5,
    ) -> list[tuple[Any, float]]:
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "run_id": run_id,
                "sub_question_id": sub_question_id,
                "limit": limit,
            }
        )
        return self.rows[:limit]


# ===========================================================================
# Distance / similarity conversion
# ===========================================================================


class TestSimilarity:
    @pytest.mark.parametrize(
        ("distance", "expected"),
        [
            (0.0, 1.0),   # identical
            (0.5, 0.5),
            (1.0, 0.0),   # orthogonal
            (2.0, 0.0),   # opposite, clamped rather than negative
        ],
    )
    def test_conversion(self, distance: float, expected: float) -> None:
        assert similarity_from_distance(distance) == pytest.approx(expected)

    def test_never_negative(self) -> None:
        """A negative score would corrupt downstream confidence arithmetic."""
        assert similarity_from_distance(1.9) >= 0.0


# ===========================================================================
# Adapter
# ===========================================================================


class TestAdapter:
    async def test_calls_repository_with_its_real_parameter_names(self) -> None:
        """Regression: the adapter previously passed embedding= and limit= only."""
        repo = FakePassageRepository([(FakePassage(), 0.1)])
        search = make_vector_search_fn(repo)

        run_id = str(uuid.uuid4())
        await search([0.1] * 768, top_k=3, run_id=run_id)

        assert repo.calls == [
            {
                "query_embedding": [0.1] * 768,
                "run_id": run_id,
                "sub_question_id": None,
                "limit": 3,
            }
        ]

    async def test_converts_tuple_rows_to_results(self) -> None:
        passage = FakePassage(text="RAG beats fine-tuning for freshness", chunk_index=4)
        repo = FakePassageRepository([(passage, 0.25)])
        search = make_vector_search_fn(repo)

        results = await search([0.1] * 768, top_k=5)

        assert len(results) == 1
        hit = results[0]
        assert isinstance(hit, VectorSearchResult)
        assert hit.passage_id == str(passage.id)
        assert hit.source_id == str(passage.source_id)
        assert hit.text == "RAG beats fine-tuning for freshness"
        assert hit.score == pytest.approx(0.75)
        assert hit.distance == pytest.approx(0.25)
        assert hit.metadata["chunk_index"] == 4

    async def test_carries_source_metadata_for_citations(self) -> None:
        repo = FakePassageRepository([(FakePassage(), 0.0)])
        results = await make_vector_search_fn(repo)([0.1] * 768)

        assert results[0].metadata["source_url"] == "https://example.com/a"
        assert results[0].metadata["source_title"] == "A"
        assert results[0].metadata["reliability_score"] == 0.8

    async def test_passes_sub_question_scope_through(self) -> None:
        repo = FakePassageRepository([])
        sub_question_id = str(uuid.uuid4())

        await make_vector_search_fn(repo)(
            [0.1] * 768, top_k=2, sub_question_id=sub_question_id
        )

        assert repo.calls[0]["sub_question_id"] == sub_question_id

    async def test_handles_detached_passage_without_source(self) -> None:
        passage = FakePassage()
        passage.source = None  # type: ignore[assignment]
        repo = FakePassageRepository([(passage, 0.3)])

        results = await make_vector_search_fn(repo)([0.1] * 768)

        assert results[0].metadata["source_url"] is None

    async def test_preserves_repository_ordering(self) -> None:
        near, mid, far = FakePassage("near"), FakePassage("mid"), FakePassage("far")
        repo = FakePassageRepository([(near, 0.01), (mid, 0.4), (far, 0.9)])

        results = await make_vector_search_fn(repo)([0.1] * 768, top_k=3)

        assert [r.text for r in results] == ["near", "mid", "far"]
        assert results[0].score > results[1].score > results[2].score


# ===========================================================================
# Tool wrapper
# ===========================================================================


class TestVectorSearchTool:
    async def test_returns_results(self) -> None:
        expected = [
            VectorSearchResult(
                passage_id="p1", source_id="s1", text="Some passage", score=0.95
            )
        ]

        async def fn(
            query_vector: list[float],
            *,
            top_k: int = 5,
            run_id: str | None = None,
            sub_question_id: str | None = None,
        ) -> list[VectorSearchResult]:
            return expected

        results = await VectorSearchTool(fn).search([0.1] * 768, top_k=5, run_id="run-123")

        assert len(results) == 1
        assert results[0].passage_id == "p1"
        assert results[0].score == pytest.approx(0.95)

    async def test_returns_empty_on_error(self) -> None:
        """Retrieval degrades rather than killing the run."""

        async def failing(
            query_vector: list[float],
            *,
            top_k: int = 5,
            run_id: str | None = None,
            sub_question_id: str | None = None,
        ) -> list[VectorSearchResult]:
            raise RuntimeError("db connection lost")

        assert await VectorSearchTool(failing).search([0.0] * 768, top_k=3) == []

    async def test_empty_query_vector_short_circuits(self) -> None:
        calls = 0

        async def fn(
            query_vector: list[float],
            *,
            top_k: int = 5,
            run_id: str | None = None,
            sub_question_id: str | None = None,
        ) -> list[VectorSearchResult]:
            nonlocal calls
            calls += 1
            return []

        assert await VectorSearchTool(fn).search([]) == []
        assert calls == 0

    async def test_min_score_filter(self) -> None:
        hits = [
            VectorSearchResult("p1", "s1", "strong", 0.9),
            VectorSearchResult("p2", "s1", "weak", 0.2),
        ]

        async def fn(
            query_vector: list[float],
            *,
            top_k: int = 5,
            run_id: str | None = None,
            sub_question_id: str | None = None,
        ) -> list[VectorSearchResult]:
            return hits

        results = await VectorSearchTool(fn).search([0.1], min_score=0.5)

        assert [r.passage_id for r in results] == ["p1"]

    async def test_forwards_scope_arguments(self) -> None:
        seen: dict[str, Any] = {}

        async def fn(
            query_vector: list[float],
            *,
            top_k: int = 5,
            run_id: str | None = None,
            sub_question_id: str | None = None,
        ) -> list[VectorSearchResult]:
            seen.update({"top_k": top_k, "run_id": run_id, "sub_question_id": sub_question_id})
            return []

        await VectorSearchTool(fn).search(
            [0.1], top_k=7, run_id="r1", sub_question_id="sq1"
        )

        assert seen == {"top_k": 7, "run_id": "r1", "sub_question_id": "sq1"}


class TestVectorSearchResult:
    def test_distance_defaults_from_score(self) -> None:
        result = VectorSearchResult("p", "s", "t", 0.8)
        assert result.distance == pytest.approx(0.2)

    def test_equality_and_repr(self) -> None:
        a = VectorSearchResult("p", "s", "t", 0.5)
        b = VectorSearchResult("p", "s", "t", 0.5)
        assert a == b
        assert "p" in repr(a)
