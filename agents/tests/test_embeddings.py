"""
M2 tests — embedding service.

Focus on the properties the research pipeline depends on: real batching,
per-text ordering, dimension safety, retries, and graceful degradation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from researchmind_agents.config import EmbeddingConfig
from researchmind_agents.llm.embeddings import EmbeddingResult, EmbeddingService


def make_embed_response(vectors: list[list[float]], total_tokens: int = 0) -> MagicMock:
    """Build a fake google-genai EmbedContentResponse for a batch."""
    embeddings = []
    for vector in vectors:
        embedding = MagicMock()
        embedding.values = vector
        embeddings.append(embedding)

    response = MagicMock()
    response.embeddings = embeddings
    if total_tokens:
        usage = MagicMock()
        usage.total_token_count = total_tokens
        response.usage_metadata = usage
    else:
        response.usage_metadata = None
    return response


class FakeModels:
    """Records every embed_content call and replays queued responses."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = responses or []

    def embed_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._responses:
            result = self._responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        count = len(kwargs["contents"])
        return make_embed_response([[0.1] * 768 for _ in range(count)])


class FakeClient:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.models = FakeModels(responses)


def service(responses: list[Any] | None = None, **overrides: Any) -> EmbeddingService:
    config = EmbeddingConfig(api_key="k", base_delay=0.0, **overrides)
    return EmbeddingService(config, client=FakeClient(responses))


class TestSingleEmbeddings:
    async def test_embed_query_uses_query_task_type(self) -> None:
        svc = service()
        result = await svc.embed_query("What is RAG?")

        assert isinstance(result, EmbeddingResult)
        assert result.ok
        assert len(result.vector) == 768
        assert result.model == "text-embedding-004"
        config = svc._client.models.calls[0]["config"]
        assert config.task_type == "RETRIEVAL_QUERY"

    async def test_embed_document_uses_document_task_type(self) -> None:
        svc = service()
        result = await svc.embed_document("Some long document")

        assert result.ok
        config = svc._client.models.calls[0]["config"]
        assert config.task_type == "RETRIEVAL_DOCUMENT"

    def test_exposes_configured_dimension_and_model(self) -> None:
        svc = service(dimension=768, model="text-embedding-004")
        assert svc.dimension == 768
        assert svc.model == "text-embedding-004"


class TestBatching:
    async def test_sends_one_request_per_batch(self) -> None:
        """PRD s10: embeddings are batched, not one call per text."""
        svc = service(batch_size=2)
        texts = [f"doc{i}" for i in range(5)]

        results = await svc.embed_documents(texts)

        assert len(results) == 5
        # 5 texts at batch_size=2 -> 3 requests, not 5.
        assert len(svc._client.models.calls) == 3
        assert [len(c["contents"]) for c in svc._client.models.calls] == [2, 2, 1]

    async def test_preserves_input_order_across_batches(self) -> None:
        responses = [
            make_embed_response([[0.0] * 768, [1.0] * 768]),
            make_embed_response([[2.0] * 768]),
        ]
        svc = service(responses, batch_size=2)

        results = await svc.embed_documents(["a", "b", "c"])

        assert [r.vector[0] for r in results] == [0.0, 1.0, 2.0]

    async def test_empty_input_makes_no_calls(self) -> None:
        svc = service()
        assert await svc.embed_documents([]) == []
        assert svc._client.models.calls == []

    async def test_token_count_spread_across_batch(self) -> None:
        svc = service([make_embed_response([[0.1] * 768, [0.2] * 768], total_tokens=10)])
        results = await svc.embed_documents(["a", "b"])
        assert [r.token_count for r in results] == [5, 5]

    async def test_concurrency_is_bounded(self) -> None:
        """An unbounded fan-out would be rate-limited immediately on a free tier."""
        svc = service(batch_size=1, max_concurrency=2)
        assert svc._semaphore._value == 2

        results = await svc.embed_documents([f"d{i}" for i in range(6)])
        assert len(results) == 6


class TestFailureHandling:
    async def test_retries_transient_error_then_succeeds(self) -> None:
        responses: list[Any] = [
            Exception("429 resource exhausted"),
            make_embed_response([[0.3] * 768]),
        ]
        svc = service(responses, max_attempts=3)

        result = await svc.embed_document("doc")

        assert result.ok
        assert len(svc._client.models.calls) == 2

    async def test_failed_batch_degrades_instead_of_raising(self) -> None:
        svc = service([Exception("invalid api key")], max_attempts=1)

        results = await svc.embed_documents(["a", "b"])

        assert len(results) == 2
        assert all(not r.ok for r in results)
        assert all("invalid api key" in r.error for r in results)

    async def test_dimension_mismatch_is_flagged_not_stored(self) -> None:
        """A wrong-width vector would corrupt every pgvector insert."""
        svc = service([make_embed_response([[0.1] * 512])], dimension=768)

        result = await svc.embed_document("doc")

        assert not result.ok
        assert "dim=512" in result.error
        assert "expected 768" in result.error
        assert result.vector == []

    async def test_empty_vector_is_an_error(self) -> None:
        svc = service([make_embed_response([[]])])
        result = await svc.embed_document("doc")
        assert not result.ok
        assert result.error == "empty embedding"

    async def test_short_response_fills_missing_results(self) -> None:
        """Provider returned fewer embeddings than inputs — stay aligned."""
        svc = service([make_embed_response([[0.1] * 768])], batch_size=2)

        results = await svc.embed_documents(["a", "b"])

        assert len(results) == 2
        assert results[0].ok
        assert not results[1].ok
        assert "fewer embeddings" in results[1].error

    async def test_partial_failure_reports_per_text(self) -> None:
        responses = [
            make_embed_response([[0.1] * 768]),
            Exception("quota exceeded"),
        ]
        svc = service(responses, batch_size=1, max_attempts=1)

        results = await svc.embed_documents(["good", "bad"])

        assert results[0].ok
        assert not results[1].ok


class TestLazyClient:
    def test_construction_needs_no_api_key(self) -> None:
        """
        Embeddings use their own provider key, which may be absent when the LLM
        runs elsewhere. Building the service must not require it.
        """
        EmbeddingService(EmbeddingConfig(api_key=""))

    def test_missing_key_surfaces_on_use_not_on_build(self) -> None:
        svc = EmbeddingService(EmbeddingConfig(api_key=""))
        with pytest.raises(ValueError, match="API key"):
            _ = svc._client

    def test_clients_are_shared_per_key(self) -> None:
        from researchmind_agents.llm.google_client import (
            reset_genai_clients,
            shared_genai_client,
        )

        reset_genai_clients()
        try:
            first = shared_genai_client("key-a")
            second = shared_genai_client("key-a")
            third = shared_genai_client("key-b")

            assert first is second
            assert first is not third
        finally:
            reset_genai_clients()


class TestEmbeddingResult:
    @pytest.mark.parametrize(
        ("vector", "error", "expected"),
        [
            ([0.1], "", True),
            ([], "", False),
            ([0.1], "boom", False),
        ],
    )
    def test_ok_property(self, vector: list[float], error: str, expected: bool) -> None:
        assert EmbeddingResult(vector=vector, error=error).ok is expected
